import copy
import math
import multiprocessing as mp
import traceback
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Type, Union

import numpy as np
import setproctitle
import torch
import torch.distributed as dist
import trainer.utils.file as futils
import trainer.utils.train as tutils
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainer.config.run import (DDPConfig, ParallelConfig, RunConfig,
                                TrainMode)
from trainer.modules.logging.main import SummaryLogger
from trainer.modules.main import (LossDivergedError, Metrics)
from trainer.modules.model.main import ModelBase
from trainer.modules.schedulers import Scheduler


class EvaluationError(Exception):
    pass


class TrainPlateauError(Exception):
    pass


class LogStepError(Exception):
    pass


class DuplicateRunError(Exception):
    pass


class ModelWrapper(ModelBase):
    def __init__(
        self,
        model_class: Type[nn.Module],
    ):
        super().__init__(
            model_class=model_class,
        )

    def _init_logger(self, allow_resume=True):

        self._init_model_dir()
        if (
            not allow_resume
            and not self.train_config.debug
            and self.model_dir.exists()
            and not self.train_config.resume
            and not self.is_slave
        ):
            raise DuplicateRunError(f"{self.model_dir} exists and resume is false")
        self.logger = SummaryLogger(
            run_config=self.run_config, model_dir=self.model_dir, is_slave=self.is_slave
        )
        if tutils.debugger_is_active() and not self.train_config.debug:
            self.logger.warn("Debug flag is False but running in debug mode.")

    def _init_model_dir(self):
        self.experiment_dir = experiment_dir = Path(self.run_config.experiment_dir)
        if self.run_config.model_dir is not None:
            self.model_dir = Path(self.run_config.model_dir)
        else:
            self.run_config.model_dir = self.model_dir = experiment_dir.joinpath(
                self.uid
            )

    def _init_log_artifacts(self):

        self._init_logger(allow_resume=False)
        self.logger.info(f"Model directory: {self.model_dir}")
        self.logger.write_config(self.run_config)

    def _init_state_attributes(self):

        train_config = self.train_config
        self.device = tutils.parse_device(train_config.device)

        self.train_mode = self.run_config.train_mode
        self.random_seed = train_config.random_seed
        self.amp = train_config.amp
        if self.device == "cpu" and self.amp:
            raise ValueError(
                "AMP is not supported for CPU. You will need to set `train_config.amp` to False."
            )
        if self.train_mode == TrainMode.dist_data_parallel:
            assert dist.is_initialized()
            assert issubclass(type(self.run_config), DDPConfig)
            self.is_master = self.run_config.rank == 0
            self.is_slave = not self.is_master
            assert (
                self.device != "cpu"
            ), f"Invalid device {self.device} in distributed mode."
            self.autocast = torch.autocast(
                enabled=self.amp,
                device_type="cuda",
            )
        elif self.train_mode == TrainMode.data_parallel:
            self.autocast = torch.autocast(
                enabled=self.amp,
                device_type="cuda",
            )
        else:
            self.autocast = torch.autocast(
                enabled=self.amp,
                device_type="cuda" if "cuda" in self.device else "cpu",
            )

        self.verbose = False if self.is_slave else train_config.tqdm
        self.verbose_tqdm = False if self.is_slave else train_config.tqdm

        if not self.verbose:
            warnings.filterwarnings("ignore")

        self.keep_n_checkpoints = train_config.keep_n_checkpoints

        if (
            train_config.early_stopping_iter is not None
            and train_config.early_stopping_iter > 0
        ):
            assert (
                self.val_dataloader is not None
            ), "dataloader function has to return validation set when setting early stopping to True"
        self.metrics = Metrics(
            batch_limit=train_config.metrics_n_batches,
            memory_limit=train_config.metrics_byte_limit,
            epochs=train_config.epochs,
            total_steps=self._total_steps(),
            moving_average_limit=self.epoch_len,
            lr=self.run_config.train_config.optimizer_config.lr,
            evaluation_functions=self.evaluation_functions,
        )

        setproctitle.setproctitle(self.get_process_name())
        # Directory setup

    def _total_steps(self):
        return self.epoch_len * self.train_config.epochs

    def _init_model_state(self):

        if self.train_config.init_chkpt is not None and self.train_config.resume:
            self.current_checkpoint = Path(self.train_config.init_chkpt)
            (
                self.model,
                self.optimizer,
                self.scaler,
                self.scheduler,
                self.save_dict,
            ) = self.load_checkpoint(self.current_checkpoint)

        elif self.train_config.init_chkpt is not None and not self.train_config.resume:
            # Loads only the weights
            self.current_checkpoint = self.train_config.init_chkpt
            self._make_model_from_chkpt(self.current_checkpoint)

        elif self.train_config.resume and not self.smoke_test:
            recent_checkpoint_dir = self.logger.CHKPT_DIRS["recent"]
            self._make_model_from_valid_checkpoint(recent_checkpoint_dir)
        else:
            self._make_new_model()

    def init_experiment_state(
        self,
        run_config: Union[RunConfig, ParallelConfig],
        make_log_artifacts=True,
        make_logger=True,
    ):
        if run_config.train_config.random_seed is not None:
            tutils.set_seed(run_config.train_config.random_seed)
        self.run_config = run_config
        # TODO assert  none of the stateful attributes are changed
        _run_config = copy.deepcopy(run_config)
        self.init_data_state()
        self.run_config = self.config_parser(run_config)
        # TODO check all derived attributes are complete
        self._init_state_attributes()
        # Does not create log artifacts during smoke test
        if make_log_artifacts:
            # TODO remove this flag and derive it beyond the resume flag. i.e. evaluate
            self._init_log_artifacts()
        elif make_logger:
            self._init_logger()
        else:
            self.logger = tutils.Dummy()
        self._init_model_state()
        _run_config.assert_state(self.run_config)

    def _make_model_from_chkpt(self, chkpt):
        self.logger.info(
            (
                "Initializing model weights ONLY (not optimizer, scheduler or scaler)"
                f" from checkpoint. {self.current_checkpoint}"
            )
        )
        (
            self.model,
            self.optimizer,
            self.scaler,
            self.scheduler,
        ) = self.create_model(init_chkpt=chkpt)

    def _make_new_model(self):
        self.logger.info("Creating new model")
        (
            self.model,
            self.optimizer,
            self.scaler,
            self.scheduler,
        ) = self.create_model()
        self.update_save_dict()

    def _make_model_from_valid_checkpoint(self, chkpt_dir):

        latest_checkpoints = futils.get_latest_chkpts(chkpt_dir)
        current_checkpoint = None
        if len(latest_checkpoints) > 0:
            # Try to load first valid chkpt in case there was a crash and some checkpoint is unrecoverable
            for i, _checkpoint in enumerate(latest_checkpoints):
                try:
                    self.logger.info(f"Loading checkpoint {_checkpoint}")
                    (
                        model,
                        optimizer,
                        scaler,
                        scheduler,
                        save_dict,
                    ) = self.load_checkpoint(_checkpoint)
                    current_checkpoint = _checkpoint
                    break
                except Exception as e:
                    if i == len(latest_checkpoints) - 1:
                        # if it is the last checkpoint raise exception
                        raise e
                    else:
                        # ignore exception
                        self.logger.error(
                            f"Error loading checkpoint {_checkpoint}. Skipping....\n{traceback.format_exc()}"
                        )
        if current_checkpoint is None:
            raise FileNotFoundError(f"Could not find a valid checkpoint in {chkpt_dir}")
        self.current_checkpoint = current_checkpoint

        (
            self.model,
            self.optimizer,
            self.scaler,
            self.scheduler,
            self.save_dict,
        ) = (model, optimizer, scaler, scheduler, save_dict)

    def load_checkpoint(
        self, checkpoint_path: Path
    ) -> Tuple[
        nn.Module,
        torch.optim.Optimizer,
        torch.cuda.amp.GradScaler,
        Scheduler,
        Dict[str, Any],
    ]:
        device = self.device
        use_amp = self.amp

        save_dict = torch.load(checkpoint_path, map_location="cpu")

        (model, optimizer, scaler, scheduler, save_dict) = self.load_model(
            save_dict, device=device, use_amp=use_amp
        )

        return model, optimizer, scaler, scheduler, save_dict

    def to_device(self, data: Iterable, device=None):
        if device is None:
            device = self.device
        return tutils.iter_to_device(data, device)

    def model_step(
        self, model: nn.Module, batch: Iterable
    ) -> Tuple[Any, Optional[torch.Tensor], torch.Tensor]:
        batch = self.to_device(batch)
        labels = None
        with self.autocast:

            if isinstance(batch, list):
                pred, loss = model(*batch)
                if len(batch) > 1:
                    labels = batch[1]
            elif isinstance(batch, dict):
                pred, loss = model(**batch)
                # TODO find a more flexible way to return labels for different problems (i.e. RL, or other methods)?
                # We could derive it from the dataset during initialization
                if "labels" in batch:
                    labels = batch["labels"]
                elif "y" in batch:
                    labels = batch["y"]
                elif "label" in batch:
                    labels = batch["label"]

        return pred, labels, loss

    def train_step(
        self, batch: Iterable
    ) -> Tuple[Any, Optional[torch.Tensor], torch.Tensor, Optional[Dict[str, Any]]]:

        model = self.model
        optimizer = self.optimizer
        scaler = self.scaler
        scheduler = self.scheduler
        # Ensure no left-over grads are in the model's parameters from custom evaluation or what-not
        optimizer.zero_grad()
        pred, labels, loss = self.model_step(model=model, batch=batch)
        loss_value = self.apply_loss(model, loss, optimizer, scaler, scheduler)
        if (
            scheduler is not None
            and getattr(scheduler, "step_when", None) == "epoch"
            and self._is_step(self.epoch_len)
        ):
            scheduler.step()  # type: ignore
        self._inc_iter()
        self._update_learning_rate()
        return pred, labels, loss_value, None  # {}  # aux metrics

    def log_step(self):

        self.logger.update(self.metrics)
        msg = self.metrics.get_msg()
        to_console = self.verbose and not self.verbose_tqdm
        self.logger.info(msg, to_console=to_console)

    def mock_train(
        self,
        run_config: Optional[RunConfig] = None,
        run_async=True,
        block: bool = True,
    ) -> mp.Process:
        mock_model = copy.deepcopy(self)

        if run_config is None:
            run_config = mock_model.run_config
        if run_async:
            p = mp.Process(target=mock_model.train, args=(run_config, True))
            p.start()
            if block:
                p.join()
            return p
        else:
            return mock_model.train(run_config=run_config, smoke_test=True)

    def update_tqdm(self):
        if not self.verbose_tqdm:
            return
        rate = self.train_tqdm.format_dict["rate"]
        time_remaining = "??"
        if rate is not None and isinstance(rate, (int, float)):
            time_remaining = self.train_tqdm.format_interval(
                (self.total_steps - self.iteration) / rate
            )
        msg = self.metrics.get_msg()
        self.train_tqdm.set_description(msg)
        self.train_tqdm.set_postfix_str(f"Remaining: {time_remaining}")
        self.train_tqdm.update(1)

    @cached_property
    def epoch_len(self):
        return len(self.train_dataloader)

    @cached_property
    def eval_itr(self):
        return math.ceil(self.train_config.eval_epoch * self.epoch_len)

    @cached_property
    def log_itr(self):
        return math.ceil(self.train_config.log_epoch * self.epoch_len)

    def log(self):
        # Log step
        if self._is_step(self.log_itr) and not self.is_slave:
            self.metrics.eval_train()
            self.log_step()

    def eval(self):
        # Evaluation step
        if self._is_step(self.eval_itr) and not self.is_slave:
            """
            Related to:
                https://discuss.pytorch.org/t/proper-distributeddataparallel-usage/74564

                How can I make it so that I can evaluate the model in async mode on the master node.

                Looking at the API `join` would be the most obvious way.
            """
            try:
                self.evaluation_step()
            except (LossDivergedError, TrainPlateauError) as e:
                error = traceback.format_exc()
                self.logger.error(error)
                raise e
            finally:
                eval_step = (
                    self.iteration
                    if self.eval_itr == 0
                    else self.iteration // self.eval_itr
                )
                msg = self.metrics.get_msg()
                self.logger.info(
                    f"Evaluation Step [{eval_step}] {msg}", to_console=False
                )

    def train_loop(self):
        train_dataloader = self.train_dataloader
        generator = iter(train_dataloader)

        for i in range(self.iteration, self.total_steps):
            self.model.train()
            try:
                batch = next(generator)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                generator = iter(train_dataloader)
                batch = next(generator)
                self.metrics.reset_train_preds()
                self.train_tqdm.reset()
            pred, labels, loss, aux_metrics = self.train_step(batch)
            self.metrics.append_train(pred, labels, loss, aux_metrics)
            if loss is not None and not np.isfinite(self.metrics.current_loss):
                self.logger.error(f"Loss Diverged. Terminating. loss: {loss}")
                break

            self.update_tqdm()
            self.log()
            self.eval()

            if self.smoke_test and i > 20:
                break
        return self.metrics

    def train(
        self,
        run_config: Union[RunConfig, ParallelConfig],
        smoke_test: bool = False,
    ):
        self.smoke_test = smoke_test

        self.init_experiment_state(
            run_config=run_config,
            make_log_artifacts=not smoke_test,
            make_logger=not smoke_test,
        )

        if self.verbose_tqdm and not smoke_test:
            self.train_tqdm = tqdm(
                total=self.epoch_len,
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                position=0,
                leave=True,
                dynamic_ncols=True,
            )
        else:
            self.train_tqdm = tutils.Dummy()

        # NOTE self.cur_epoch can be different than cur_epoch if resuming training
        # We use total steps for training to avoid incosistency between runs.

        try:
            return self.train_loop()
        except KeyboardInterrupt:
            self.checkpoint()

        return self.metrics

    def sync(self):
        self.logger.sync()

    def evaluation_step(self):
        is_best = False
        val_loss = None
        if self.val_dataloader is not None:
            self.run_evaluate(self.val_dataloader, tag="val")
            div_warm_up_steps = self.train_config.stop_div_step_frac
            is_best = self.metrics.eval_best(
                div_factor=self.train_config.stop_div_factor,
                div_warm_up_steps=div_warm_up_steps,
            )
            val_loss = self.metrics.val_loss
        # Save best and latest checkpoints

        if (
            self.scheduler is not None
            and hasattr(self.scheduler, "step_when")
            and self.scheduler.step_when == "val"
        ):
            if val_loss is None:
                scheduler_name = self.train_config.scheduler_config.name
                raise EvaluationError(
                    f"A validation dataset is rquired with {scheduler_name} scheduler"
                )
            self.scheduler.step(val_loss)

        self.checkpoint()
        if is_best:
            self.checkpoint(is_best=True)

        # Early stopping
        early_stopping_iter = self.train_config.early_stopping_iter

        if (
            early_stopping_iter is not None
            and (self.iteration - self.metrics.best_iteration) > early_stopping_iter
        ):
            raise TrainPlateauError(
                f"Early stopping, no improvement for {early_stopping_iter} iterations."
            )

    @torch.no_grad()
    def validation_step(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        tag: str,
        subsample: float = 1.0,
    ) -> Dict[str, Any]:

        self.metrics.init_preds(
            tag,
            batch_limit=len(dataloader) + 1,
            memory_limit=float("inf"),
            moving_average_limit=len(dataloader) + 1,
        )

        was_training = model.training
        cutoff_itr = len(dataloader) * subsample
        model.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                pred, labels, loss = self.model_step(model=model, batch=batch)
                self.metrics.append(pred, labels, tag, loss=loss)
                if i > cutoff_itr or self.smoke_test:
                    break

        self.metrics.eval_metrics(tag)

        custom_metrics = self.custom_evaluation(model, dataloader=dataloader)
        if was_training:
            model.train()
        return_dict = self.metrics.get_added_metrics(tag)
        if custom_metrics is not None:
            self.metrics.update(custom_metrics)
            return_dict.update(custom_metrics)
        return return_dict

    @torch.no_grad()
    def run_evaluate(self, dataloader, tag: Optional[str] = None) -> Dict[str, Any]:

        model: nn.Module = (
            self.model.module  # type: ignore
            if self.train_mode == TrainMode.dist_data_parallel
            or self.train_mode == TrainMode.data_parallel
            else self.model
        )
        if tag is None:
            tag = "val"
        metrics = self.validation_step(
            model, dataloader, tag, self.train_config.eval_subsample
        )
        return metrics

    def evaluate(
        self,
        run_config: RunConfig,
        model_dir: Optional[str] = None,
        chkpt: Optional[str] = None,
    ):

        if model_dir is not None:
            run_config.load(list(Path(model_dir).glob("config.yaml"))[0])
            run_config.model_dir = model_dir
            run_config.train_config.resume = True
        elif (
            run_config.model_dir is None
            and getattr(self, "model_dir", None) is not None
        ):
            run_config.load(list(Path(self.model_dir).glob("config.yaml"))[0])
            run_config.model_dir = self.model_dir.as_posix()
            run_config.train_config.resume = True
        elif run_config.experiment_dir is not None:
            run_config.train_config.resume = True

        if chkpt is not None:
            run_config.train_config.init_chkpt = chkpt
            run_config.train_config.resume = False

        if (
            run_config.model_dir is None
            and run_config.train_config.init_chkpt is not None
            and not run_config.train_config.resume
        ):
            raise RuntimeError(
                "Must provide a model dir or init_chkpt as an argument or through run_config."
            )
        if (
            run_config.train_mode == TrainMode.data_parallel
            or run_config.train_mode == TrainMode.dist_data_parallel
        ):
            run_config.train_mode = TrainMode.vanilla
        self.init_experiment_state(
            run_config, make_log_artifacts=False, make_logger=True
        )
        self.logger.info(f"Evaluating {self.current_checkpoint}")

        msg = self.metrics.get_msg()
        self.logger.info(f"Current metrics: {msg}")
        for loader, tag in zip(
            [self.test_dataloader, self.val_dataloader], ["test", "val"]
        ):
            if loader is not None:
                self.run_evaluate(loader, tag=tag)
                msg = self.metrics.get_msg_preds(tag)
                self.logger.info(f"{tag} metrics: {msg}")

import copy
import multiprocessing as mp
import traceback
import typing as ty
from abc import abstractmethod
from collections.abc import Callable, Iterable

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import ablator.utils.base as butils
from ablator.main.configs import ModelConfig, RunConfig, TrainConfig
from ablator.main.model.main import EvaluationError, ModelBase, TrainPlateauError
from ablator.modules.metrics.main import LossDivergedError, TrainMetrics
from ablator.modules.optimizer import OptimizerConfig
from ablator.modules.scheduler import Scheduler, SchedulerConfig


class ModelWrapper(ModelBase):
    """
    A wrapper around model_class that removes training boiler-plate code, with over-writable functions
    with support for custom use-cases.
    """

    def __init__(
        self,
        model_class: type[nn.Module],
    ):
        super().__init__(
            model_class=model_class,
        )
        # Will be loaded or created from checkpoint
        self.model: nn.Module
        self.optimizer: Optimizer
        self.scaler: GradScaler
        self.scheduler: Scheduler | None

    @property
    def train_config(self) -> TrainConfig:
        return self.run_config.train_config

    @property
    def model_config(self) -> ModelConfig:
        return self.run_config.model_config

    def create_model(
        self,
        save_dict: dict[str, ty.Any] | None = None,
        strict_load: bool = True,
    ) -> None:
        save_dict = {} if save_dict is None else save_dict
        scheduler_state = save_dict["scheduler"] if "scheduler" in save_dict else None
        optimizer_state = save_dict["optimizer"] if "optimizer" in save_dict else None
        scaler_state = save_dict["scaler"] if "scaler" in save_dict else None

        model_class = self.model_class
        model: nn.Module
        if (model_config := self.model_config) is not None:
            model = model_class(model_config)
        else:
            # Support of decleartive paradigm without model over-writing
            model = model_class()

        if "model" in save_dict:
            model.load_state_dict(save_dict["model"], strict=strict_load)
        elif self.train_config.rand_weights_init:
            model.apply(butils.init_weights)

        model = model.to(self.device)
        optimizer = self.create_optimizer(
            model=model,
            optimizer_config=self.train_config.optimizer_config,
            optimizer_state=optimizer_state,
        )
        scheduler = self.create_scheduler(
            model=model,
            optimizer=optimizer,
            scheduler_config=self.train_config.scheduler_config,
            scheduler_state=scheduler_state,
        )
        scaler = self.create_scaler(scaler_state=scaler_state)
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler

    def create_scheduler(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler_config: SchedulerConfig | None = None,
        scheduler_state: dict | None = None,
    ) -> Scheduler | None:
        scheduler: ty.Optional[Scheduler] = None
        if scheduler_config is not None:
            scheduler = scheduler_config.make_scheduler(model, optimizer)

        if scheduler_state is not None:
            if scheduler is None:
                self.logger.warn(
                    "Supplied `scheduler_state` without `scheduler_config`. Ignoring scheduler."
                )
                return None
            scheduler.load_state_dict(scheduler_state)
        return scheduler

    def create_optimizer(
        self,
        model: nn.Module,
        optimizer_config: OptimizerConfig | None = None,
        optimizer_state: dict[str, ty.Any] | None = None,
    ) -> Optimizer:
        optimizer: Optimizer

        if optimizer_config is not None:
            optimizer = optimizer_config.make_optimizer(model)

        if optimizer_state is not None and optimizer is not None:
            # NOTE: because https://github.com/pytorch/pytorch/issues/80809
            # TODO any good fix  for this yet?
            for k in optimizer_state["state"].keys():
                if "step" in optimizer_state["state"][k] and isinstance(
                    optimizer_state["state"][k]["step"], torch.Tensor
                ):
                    optimizer_state["state"][k]["step"] = optimizer_state["state"][k][
                        "step"
                    ].cpu()

            optimizer.load_state_dict(optimizer_state)
        elif optimizer_state is not None:
            self.logger.warn(
                "Supplied `optimizer_state` without `optimizer_config`. Ignoring optimizer."
            )

        return optimizer

    def create_scaler(self, scaler_state: ty.Optional[dict] = None) -> GradScaler:
        scaler = GradScaler(enabled=self.run_config.amp)
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        return scaler

    def reset_optimizer_scheduler(self):
        optimizer_config = self.train_config.optimizer_config
        scheduler_config = self.train_config.scheduler_config
        optimizer = self.create_optimizer(
            model=self.model,
            optimizer_config=optimizer_config,
        )
        scheduler = self.create_scheduler(
            model=self.model,
            optimizer=optimizer,
            scheduler_config=scheduler_config,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

    def load_checkpoint(
        self, save_dict: dict[str, ty.Any], model_only: bool = False
    ) -> None:
        if model_only:
            del save_dict["scheduler"]
            del save_dict["optimizer"]
            del save_dict["scaler"]

        self.create_model(
            save_dict,
            strict_load=True,
        )

    def to_device(self, data: Iterable, device=None) -> Iterable:
        if device is None:
            device = self.device
        return butils.iter_to_device(data, device)

    def model_step(
        self, model: nn.Module, batch: Iterable
    ) -> tuple[dict[str, torch.Tensor] | None, torch.Tensor | None]:
        batch = self.to_device(batch)
        with self.autocast:
            if isinstance(batch, list):
                out = model(*batch)
            elif isinstance(batch, dict):
                out = model(**batch)
            else:
                out = model(batch)

        return out

    @ty.final
    def _update_learning_rate(self):
        self.learning_rate = butils.get_lr(self.optimizer)
        return self.learning_rate

    @ty.final
    def _inc_iter(self):
        self.current_iteration += 1

    def _is_step(self, step_interval):
        return (
            step_interval > 0
            and self.current_iteration > 0
            and self.current_iteration % step_interval == 0
        )

    def _train_evaluation_step(self, smoke_test=False):
        is_best = False
        val_loss = None
        if self.val_dataloader is not None:
            metrics = self._validation_loop(
                model=self.model,
                dataloader=self.val_dataloader,
                tag="val",
                metrics=self.metrics,
                subsample=self.run_config.eval_subsample,
                smoke_test=smoke_test,
            )
            val_loss = metrics["val_loss"] if "val_loss" in metrics else None
        if val_loss is not None:
            # Use val loss for scheduling or finding best checkpoint

            is_best = val_loss < self.best_loss

            if is_best or self.best_loss == 0:
                self.best_iteration = self.current_iteration
                self.best_loss = val_loss

            divergence_step = (
                self.current_iteration > self.epoch_len * self.run_config.warm_up_epochs
            )
            is_diverged = val_loss / self.best_loss > self.run_config.divergence_factor

            if is_diverged and divergence_step:
                raise LossDivergedError(
                    f"Val loss {val_loss:.4e} has diverged by"
                    f"a factor of {self.run_config.divergence_factor} to "
                    f"best loss {self.best_loss:.4e}"
                )

        if (
            self.scheduler is not None
            and hasattr(self.train_config.scheduler_config, "step_when")
            and self.train_config.scheduler_config.step_when == "val"
        ):
            if val_loss is None:
                raise EvaluationError(
                    f"A validation dataset is rquired with {self.scheduler.__name__} scheduler"
                )
            self.scheduler.step(val_loss)

        self._checkpoint()
        if is_best:
            self._checkpoint(is_best=True)

        # Early stopping
        early_stopping_iter = self.run_config.early_stopping_iter

        if (
            early_stopping_iter is not None
            and (self.current_iteration - self.best_iteration) > early_stopping_iter
        ):
            raise TrainPlateauError(
                f"Early stopping, no improvement for {early_stopping_iter} iterations."
            )

    def _model_step(
        self, model: nn.Module, batch: Iterable
    ) -> tuple[dict[str, torch.Tensor] | None, torch.Tensor | None]:
        out = self.model_step(model=model, batch=batch)

        try:
            outputs, loss = out
            assert isinstance(outputs, (dict, type(None))) and isinstance(
                loss, (torch.Tensor, type(None))
            )
            if outputs is not None:
                for k, v in outputs.items():
                    assert isinstance(k, str) and isinstance(v, torch.Tensor)
        except Exception as exc:
            raise RuntimeError(
                "Model should return outputs: dict[str, torch.Tensor] | None, loss: torch.Tensor | None."
            ) from exc
        return outputs, loss

    @ty.final
    def train_step(
        self, batch: Iterable
    ) -> tuple[dict[str, torch.Tensor] | None, dict[str, ty.Any]]:
        model = self.model
        optimizer = self.optimizer
        scaler = self.scaler
        scheduler = self.scheduler
        # Ensure no left-over grads are in the model's parameters from custom evaluation or what-not
        optimizer.zero_grad()
        outputs, loss = self._model_step(model=model, batch=batch)

        loss_value = self.apply_loss(model, loss, optimizer, scaler, scheduler)
        aux_metrics = None
        if outputs is not None:
            aux_metrics = self.aux_metrics(outputs)
        if (
            scheduler is not None
            and getattr(scheduler, "step_when", None) == "epoch"
            and self._is_step(self.epoch_len)
        ):
            scheduler.step()  # type: ignore
        self._inc_iter()
        self._update_learning_rate()

        train_metrics = {}
        if loss is not None:
            train_metrics["loss"] = loss_value
        if aux_metrics is not None:
            assert (
                "loss" not in aux_metrics
            ), "Can not return key `loss` from `aux_metrics`"
            train_metrics.update(aux_metrics)
        return outputs, train_metrics

    def log_step(self):
        self.logger.update(self.metrics)
        msg = self.status_message()
        verbose = self.verbose == "console"
        self.logger.info(msg, verbose=verbose)

    @ty.final
    def mock_train(
        self,
        run_config: ty.Optional[RunConfig] = None,
        run_async=True,
        block: bool = True,
    ) -> mp.Process | TrainMetrics:
        mock_model = copy.deepcopy(self)

        if run_config is None:
            run_config = mock_model.run_config
        if run_async:
            p = mp.Process(target=mock_model.train, args=(run_config, True))
            p.start()
            if block:
                p.join()
            return p
        return mock_model.train(run_config=run_config, smoke_test=True)

    def update_status(self):
        self.metrics.update_static_metrics(self.train_stats)
        if self.verbose != "tqdm":
            return
        rate = self.train_tqdm.format_dict["rate"]
        time_remaining = "??"
        if rate is not None and isinstance(rate, (int, float)):
            time_remaining = self.train_tqdm.format_interval(
                (self.total_steps - self.current_iteration) / rate
            )
        msg = self.status_message()
        self.train_tqdm.set_description(msg)
        self.train_tqdm.set_postfix_str(f"Remaining: {time_remaining}")
        self.train_tqdm.update(1)

    def status_message(self) -> str:
        # must return current epoch, iter, losses and metrics
        return " ".join([f"{k}: {v}" for k, v in self.metrics.to_dict().items()])

    def log(self):
        # Log step
        if self._is_step(self.log_itr):
            self.metrics.evaluate("train", reset=False)
            self.log_step()

    @ty.final
    def eval(self, smoke_test=False):
        # Evaluation step
        if self._is_step(self.eval_itr):
            try:
                self._train_evaluation_step(smoke_test=smoke_test)
            except (LossDivergedError, TrainPlateauError) as e:
                error = traceback.format_exc()
                self.logger.error(error)
                raise e
            finally:
                eval_step = (
                    self.current_iteration
                    if self.eval_itr == 0
                    else self.current_iteration // self.eval_itr
                )
                msg = self.status_message()
                self.logger.info(f"Evaluation Step [{eval_step}] {msg}", verbose=False)

    @property
    def total_steps(self):
        return self.epoch_len * self.epochs

    def train_loop(self, smoke_test=False):
        train_dataloader = self.train_dataloader
        generator = iter(train_dataloader)

        for i in range(self.current_iteration, self.total_steps):
            self.model.train()
            try:
                batch = next(generator)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                generator = iter(train_dataloader)
                batch = next(generator)
                self.metrics.reset("train")
                self.train_tqdm.reset()
            outputs, train_metrics = self.train_step(batch)
            self.metrics.append_batch(**outputs, tag="train")
            self.metrics.update_ma_metrics(train_metrics, tag="train")

            if "loss" in train_metrics and not np.isfinite(train_metrics["loss"]):
                msg = f"Loss Diverged. Terminating. loss: {train_metrics['loss']}"
                self.logger.error(msg)
                raise LossDivergedError(msg)

            if not smoke_test:
                self.update_status()
                self.log()
                self.eval()

            if smoke_test and i > self.epoch_len * 0.01:
                self.eval(smoke_test=True)
                break
        return self.metrics

    @ty.final
    def train(
        self, run_config: RunConfig, smoke_test: bool = False, debug: bool = False
    ) -> TrainMetrics:
        self._init_state(run_config=run_config, smoke_test=smoke_test, debug=debug)

        try:
            return self.train_loop(smoke_test)
        except KeyboardInterrupt:
            self._checkpoint()

        return self.metrics

    @ty.final
    def evaluate(
        self,
        run_config: RunConfig,
    ):
        self._init_state(run_config, resume=True)
        self.logger.info(f"Evaluating {self.current_checkpoint}")

        # TODO make new metrics
        msg = self.metrics.to_dict()
        self.logger.info(f"Current metrics: {msg}")
        metrics = {}
        for loader, tag in zip(
            [self.test_dataloader, self.val_dataloader], ["test", "val"]
        ):
            if loader is not None:
                # NOTE we set max memory limit and let it crash because we do not want
                # inaccurate metrics calculation. Possibly smarter ways to go about it.
                eval_metrics = TrainMetrics(
                    batch_limit=len(loader) + 1,
                    memory_limit=int(1e9),
                    moving_average_limit=len(loader),
                    evaluation_functions=self.evaluation_functions(),
                    tags=[tag],
                    moving_aux_metrics=["loss"] + getattr(self, "aux_metric_names", []),
                )
                self._validation_loop(
                    model=self.model,
                    dataloader=loader,
                    tag=tag,  # type: ignore
                    metrics=eval_metrics,
                    subsample=1,
                )
                metrics[tag] = eval_metrics
                msg = self.metrics.to_dict()
                self.logger.info(f"Evaluation: {msg}")
        return metrics

    def apply_loss(
        self,
        model: nn.Module,
        loss: torch.Tensor | None,
        optimizer: Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        scheduler: ty.Optional[Scheduler],
    ) -> float | None:
        if loss is not None:
            loss = torch.mean(loss)
            if self.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_value = loss.item()
        else:
            loss_value = None

        if self.amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None and getattr(scheduler, "step_when", None) == "train":
            scheduler.step()  # type: ignore
        return loss_value

    @torch.no_grad()
    def _validation_loop(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        metrics: TrainMetrics,
        tag: ty.Literal["train", "test", "val"],
        subsample: float = 1.0,
        smoke_test: bool = False,
    ) -> dict[str, float]:
        was_training = model.training
        model.eval()
        if (batch_lim := metrics.__batch_limit__) < len(dataloader):
            self.logger.warn(
                f"Metrics batch-limit {batch_lim} is smaller than "
                f"the validation dataloader length {len(dataloader)}. "
            )
        metrics_dict = self.validation_loop(
            model, dataloader, metrics, tag, subsample, smoke_test
        )
        if was_training:
            model.train()
        return metrics_dict

    def validation_loop(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        metrics: TrainMetrics,
        tag: ty.Literal["train", "test", "val"],
        subsample: float = 1.0,
        smoke_test: bool = False,
    ) -> dict[str, float]:
        cutoff_itr = len(dataloader) * subsample
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                outputs, loss = self._model_step(model=model, batch=batch)
                if outputs is not None:
                    aux_metrics = self.aux_metrics(outputs)
                    metrics.append_batch(tag=tag, **outputs)

                val_metrics = {}
                if loss is not None:
                    val_metrics["loss"] = torch.mean(loss).item()
                if aux_metrics is not None:
                    assert (
                        "loss" not in aux_metrics
                    ), "Invalid return key `loss` from `aux_metrics`"
                    val_metrics.update(aux_metrics)
                metrics.update_ma_metrics(val_metrics, tag=tag)
                if i > cutoff_itr or smoke_test:
                    break
        metrics.evaluate(tag)
        metrics_dict = {
            k: v for k, v in metrics.to_dict().items() if k.startswith(f"{tag}_")
        }
        return metrics_dict

    @abstractmethod
    def make_dataloader_train(self, run_config: RunConfig) -> DataLoader:
        pass

    def evaluation_functions(self) -> dict[str, Callable] | None:
        return None

    # Functions that can be optionally over-written.
    def make_dataloader_test(self, run_config: RunConfig) -> DataLoader | None:
        pass

    def make_dataloader_val(self, run_config: RunConfig) -> DataLoader | None:
        pass

    def custom_evaluation(
        self, model: nn.Module, dataloader: Iterable
    ) -> ty.Optional[dict[str, ty.Any]]:
        pass

    def aux_metrics(
        self, output_dict: dict[str, torch.Tensor] | None
    ) -> ty.Optional[dict[str, ty.Any]]:
        pass

    def config_parser(self, run_config: RunConfig):
        """
        Used to initialize Derived properties
        """
        return run_config

    def make_dataloaders(self, run_config: RunConfig) -> None:
        """
        This function is done post-initialization because otherwise the
        dataloaders are pickled with the object when running distributed.
        """
        self.train_dataloader = self.make_dataloader_train(run_config)
        self.val_dataloader = self.make_dataloader_val(run_config)
        self.test_dataloader = self.make_dataloader_test(run_config)

    def checkpoint(self, is_best=False):
        self.logger.checkpoint(
            self.current_state,
            self.model.__class__.__name__,
            is_best=is_best,
            itr=self.current_iteration,
        )

    def save_dict(self) -> dict[str, ty.Any]:
        model_state_dict = self.model.state_dict()

        optimizer_state_dict = None
        if self.optimizer is not None:
            optimizer_state_dict = self.optimizer.state_dict()

        scheduler_state_dict = None
        if self.scheduler is not None:
            scheduler_state_dict = self.scheduler.state_dict()

        scaler_state_dict = None
        if self.scaler is not None:
            scaler_state_dict = self.scaler.state_dict()

        return {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "scheduler": scheduler_state_dict,
            "scaler": scaler_state_dict,
        }

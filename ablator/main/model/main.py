import copy
import math
import traceback
import typing as ty
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from functools import cached_property
from pathlib import Path

import setproctitle
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import ablator.utils.base as butils
from ablator.main.configs import RunConfig
from ablator.modules.loggers.main import SummaryLogger
from ablator.modules.metrics.main import TrainMetrics
from ablator.utils.base import Dummy


class EvaluationError(Exception):
    pass


class TrainPlateauError(Exception):
    pass


class LogStepError(Exception):
    pass


class ModelBase(ABC):
    """
    A base class that removes training boiler-plate code with extensible support
    for multiple use-cases. The class follows a stateful initialization paradigm.
    Requires the user to implement specific to their use-case load model and
    creation functionality.
    """

    def __init__(
        self,
        model_class: type[nn.Module],
    ):
        self.model_class = model_class
        self.run_config: RunConfig
        self.train_dataloader: DataLoader
        self.val_dataloader: DataLoader | None = None
        self.test_dataloader: DataLoader | None = None
        self.logger: ty.Union[SummaryLogger, Dummy]
        self.device: str
        self.model_dir: Path | None = None
        self.experiment_dir: Path | None = None
        self.autocast: torch.autocast
        self.verbose: ty.Literal["tqdm", "console", "silent"]
        self.amp: bool
        self.random_seed: ty.Optional[int]
        self.train_tqdm: tqdm = None

        self.current_checkpoint: Path | None = None
        # Runtime metrics
        self.metrics: TrainMetrics
        self.current_state: dict = {}

        # stats
        self.learning_rate = float("inf")
        self.total_steps: int
        self.epochs: int
        self.current_iteration = 0
        self.best_iteration = 0
        self.best_loss = float("inf")

    @property
    def train_stats(self) -> OrderedDict:
        return OrderedDict(
            learning_rate=self.learning_rate,
            total_steps=self.total_steps,
            epochs=self.epochs,
            current_epoch=self.current_epoch,
            current_iteration=self.current_iteration,
            best_iteration=self.best_iteration,
            best_loss=self.best_loss,
        )

    @property
    def current_epoch(self) -> int:
        if self.current_iteration > 0:
            return math.floor(self.current_iteration / self.total_steps * self.epochs)
        return 0

    @cached_property
    def epoch_len(self):
        assert (
            hasattr(self, "train_dataloader") and len(self.train_dataloader) > 0
        ), "Undefined train_dataloader."
        return len(self.train_dataloader)

    @cached_property
    def eval_itr(self):
        return math.ceil(self.run_config.eval_epoch * self.epoch_len)

    @cached_property
    def log_itr(self):
        return math.ceil(self.run_config.log_epoch * self.epoch_len)

    @property
    def uid(self):
        return self.run_config.uid

    def _get_process_name(self) -> str:
        if self.model_dir is not None and self.experiment_dir is not None:
            proc_title = self.model_dir.relative_to(
                self.experiment_dir.parent
            ).as_posix()
        else:
            proc_title = self.uid
        return proc_title

    @abstractmethod
    def create_model(
        self,
        save_dict: dict[str, ty.Any] | None = None,
        strict_load: bool = True,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def checkpoint(self, is_best=False):
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        run_config: RunConfig,
        smoke_test: bool = False,
    ):
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        run_config: RunConfig,
    ):
        raise NotImplementedError

    @abstractmethod
    def make_dataloaders(self, run_config: RunConfig):
        raise NotImplementedError

    @abstractmethod
    def config_parser(self, run_config: RunConfig):
        raise NotImplementedError

    def _init_logger(self, resume=False, debug=False):
        self.logger = SummaryLogger(
            run_config=self.run_config,
            model_dir=self.model_dir,
            resume=resume,
            keep_n_checkpoints=self.run_config.keep_n_checkpoints,
            verbose=self.run_config.verbose == "console",
        )
        if butils.debugger_is_active() and not debug:
            self.logger.warn("Debug flag is False but running in debug mode.")

        self.logger.info(f"Model directory: {self.model_dir}")

    def _make_dataloaders(self, run_config: RunConfig):
        self.make_dataloaders(run_config)
        assert (
            len(self.train_dataloader) > 0
        ), "Must define a train dataloader in `make_dataloader`"
        self.epochs = self.run_config.train_config.epochs

    def _init_class_attributes(self, debug=False):
        """Initializes class attributes based on the configuration"""
        run_config = self.run_config
        self.device = butils.parse_device(run_config.device)

        self.amp = run_config.amp
        if self.device == "cpu" and self.amp:
            raise ValueError(
                "AMP is not supported for CPU. You will need to set `run_config.amp` to False."
            )

        self.autocast = torch.autocast(
            enabled=self.amp,
            device_type="cuda" if "cuda" in self.device else "cpu",
        )

        self.verbose = run_config.verbose

        if self.verbose == "silent":
            warnings.filterwarnings("ignore")

        if (
            run_config.early_stopping_iter is not None
            and run_config.early_stopping_iter > 0
        ):
            assert (
                self.val_dataloader is not None
            ), "dataloader function has to return validation set when setting early stopping to True"

        self.metrics = TrainMetrics(
            batch_limit=run_config.metrics_n_batches,
            memory_limit=int(run_config.metrics_mb_limit * 1e6),
            moving_average_limit=self.epoch_len,
            evaluation_functions=self.evaluation_functions(),
            tags=["train"] + (["val"] if self.val_dataloader is not None else []),
            static_aux_metrics=self.train_stats,
            moving_aux_metrics=["loss"] + getattr(self, "aux_metric_names", []),
        )
        if self.run_config.experiment_dir is not None and not debug:
            self.experiment_dir = Path(self.run_config.experiment_dir)
            self.model_dir = self.experiment_dir.joinpath(self.uid)

        if debug and (self.experiment_dir is not None or self.model_dir is not None):
            self.experiment_dir = self.model_dir = None

        setproctitle.setproctitle(self._get_process_name())

    def _init_model_state(self, resume: bool = False, smoke_test: bool = False):
        if self.run_config.init_chkpt is not None and resume:
            self.current_checkpoint = Path(self.run_config.init_chkpt)
            self._load_model(self.current_checkpoint, model_only=False)

        elif self.run_config.init_chkpt is not None and not resume:
            # Loads only the weights
            self.current_checkpoint = Path(self.run_config.init_chkpt)
            self.logger.info(
                f"Initializing model weights ONLY from checkpoint. {self.current_checkpoint}"
            )

            self._load_model(self.current_checkpoint, model_only=True)

        elif resume and not smoke_test:
            if "recent" not in self.logger.CHKPT_DIRS:
                raise RuntimeError("Checkpoint folder was not found.")
            recent_checkpoint_dir = self.logger.CHKPT_DIRS["recent"]
            # NOTE: current_checkpoint is found in _find_load_valid_checkpoint
            self._find_load_valid_checkpoint(recent_checkpoint_dir)
        else:
            self.current_checkpoint = None
            self.logger.info("Creating new model")
            self.create_model()
            self._update_save_dict()

    def _init_state(
        self,
        run_config: RunConfig,
        smoke_test: bool = False,
        debug: bool = False,
        resume: bool = False,
    ):
        self.run_config = run_config
        self.random_seed = self.run_config.random_seed
        if self.random_seed is not None:
            butils.set_seed(self.random_seed)
        self.run_config = run_config
        _run_config = copy.deepcopy(run_config)
        self._make_dataloaders(self.run_config)

        self.run_config = self.config_parser(run_config)
        self._init_class_attributes(debug=debug)
        # Does not create log artifacts during smoke test
        if not smoke_test:
            self._init_logger(resume=resume, debug=debug)
        else:
            self.logger = butils.Dummy()
        self._init_model_state(resume, smoke_test)
        self.run_config.assert_state(_run_config)

        if self.verbose == "tqdm" and not smoke_test:
            self.train_tqdm = tqdm(
                total=self.epoch_len,
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                position=0,
                leave=True,
                dynamic_ncols=True,
            )
        else:
            self.train_tqdm = butils.Dummy()

    def _find_load_valid_checkpoint(self, chkpt_dir):
        latest_checkpoints = butils.get_latest_chkpts(chkpt_dir)
        current_checkpoint = None
        if len(latest_checkpoints) > 0:
            # Try to load first valid chkpt in case there was a crash and some checkpoint is unrecoverable
            for i, _checkpoint in enumerate(latest_checkpoints):
                try:
                    self.logger.info(f"Loading checkpoint {_checkpoint}")
                    self._load_model(_checkpoint, model_only=False)
                    current_checkpoint = _checkpoint
                    break
                except Exception as e:
                    if i == len(latest_checkpoints) - 1:
                        # if it is the last checkpoint raise exception
                        raise RuntimeError("Checkpoint not found") from e

                    # ignore exception
                    self.logger.error(
                        f"Error loading checkpoint {_checkpoint}. Trying another....\n{traceback.format_exc()}"
                    )
        if current_checkpoint is None:
            raise FileNotFoundError(f"Could not find a valid checkpoint in {chkpt_dir}")
        self.current_checkpoint = current_checkpoint

    def _load_model(self, checkpoint_path: Path, model_only: bool = False) -> None:
        if not hasattr(self, "run_config") or self.run_config is None:
            raise NotImplementedError(
                "Can not load model on an unitialzed model state. Consider run init_experiment_state function first"
            )

        save_dict = torch.load(checkpoint_path, map_location="cpu")

        run_config = type(self.run_config)(**save_dict["run_config"])
        assert run_config.uid == self.run_config.uid

        self._load_stats(save_dict)
        self.load_checkpoint(save_dict, model_only=model_only)
        self.current_state = save_dict

    @abstractmethod
    def load_checkpoint(
        self, save_dict: dict[str, ty.Any], model_only: bool = False
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_dict(self) -> dict[str, ty.Any] | None:
        raise NotImplementedError

    @abstractmethod
    def evaluation_functions(self) -> dict[str, Callable] | None:
        raise NotImplementedError

    def _load_stats(self, save_dict) -> None:
        metrics = copy.deepcopy(save_dict["metrics"])

        for k in self.train_stats:
            if (
                isinstance(getattr(type(self), k, None), property)
                and getattr(type(self), k).fset is None
            ):
                if getattr(self, k, None) != metrics[k]:
                    self.logger.warn(
                        f"Immutable class attribute {k} value {getattr(self, k)}"
                        f"different than loaded value {metrics[k]}"
                    )
                del metrics[k]

        for k in self.train_stats:
            if k in metrics:
                setattr(self, k, metrics[k])
                del metrics[k]

        self.metrics.update_static_metrics(self.train_stats)
        tags = {m.split("_")[0] for m in metrics}
        metric_names = {m.split("_")[1] for m in metrics}

        for tag in tags:
            self.metrics.update_ma_metrics(
                {m: metrics[f"{tag}_{m}"] for m in metric_names}, tag=tag
            )

    def _update_save_dict(self, user_save_dict: dict[str, ty.Any] | None = None):
        self.current_state = {
            "run_config": self.run_config.to_dict(),
            "metrics": self.metrics.to_dict(),
        }
        if user_save_dict is not None:
            self.current_state.update(**user_save_dict)

    def _checkpoint(self, is_best=False):
        user_save_dict = self.save_dict()
        self._update_save_dict(user_save_dict)
        self.checkpoint(is_best=is_best)

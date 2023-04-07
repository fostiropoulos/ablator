import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Type, Union

import torch
import trainer.modules.model.utils as mutils
import trainer.utils.train as tutils
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainer.config.run import (
    DPConfig,
    ModelConfigBase,
    RunConfig,
    TrainConfigBase,
    TrainMode,
)
from trainer.modules.logging.main import SummaryLogger
from trainer.modules.main import Metrics
from trainer.modules.schedulers import Scheduler


class ModelBase(ABC):
    def __init__(
        self,
        model_class: Type[nn.Module],
    ):
        self.model_class = model_class
        self.run_config: RunConfig
        self.dataloader_train_fn: Optional[Callable] = None
        self.dataloader_val_fn: Optional[Callable] = None
        self.dataloader_test_fn: Optional[Callable] = None
        self.train_dataloader: DataLoader
        self.val_dataloader: Optional[DataLoader]
        self.test_dataloader: Optional[DataLoader]
        self.logger: Union[SummaryLogger, tutils.Dummy]
        self.train_tqdm: tqdm = None

        # Setup attributes from config
        # Will be loaded or created from checkpoint
        self.model: nn.Module
        self.optimizer: torch.optim.Optimizer
        self.scaler: torch.cuda.amp.GradScaler
        self.scheduler: Scheduler
        # Used by Distributed Training
        self.is_master: Optional[bool] = None
        self.is_slave: Optional[bool] = None

        self.device: str
        self.model_dir: Path
        self.experiment_dir: Path
        self.autocast: torch.autocast

        self.verbose: bool
        self.verbose_tqdm: bool
        self.random_seed: Optional[int]

        self.current_checkpoint: Optional[Path] = None
        # Runtime metrics
        self.metrics: Metrics
        self.save_dict: dict = {}

        self.smoke_test: bool = False
        self.amp: bool = False
        self.train_mode: TrainMode

    @property
    def total_steps(self):
        if hasattr(self, "metrics"):
            return self.metrics.total_steps
        else:
            return float("inf")


    @property
    def iteration(self):
        if hasattr(self, "metrics"):
            return self.metrics.current_iteration
        else:
            return 0

    @property
    def learning_rate(self):
        if hasattr(self, "metrics"):
            return self.metrics.lr
        else:
            return float("inf")

    @property
    def current_epoch(self):
        if hasattr(self, "metrics"):
            return self.metrics.current_epoch
        else:
            return 0

    @property
    def train_config(self) -> TrainConfigBase:
        return self.run_config.train_config

    @property
    def model_config(self) -> ModelConfigBase:
        return self.run_config.model_config

    def _is_step(self, step_interval):
        return (
            step_interval > 0
            and self.iteration > 0
            and self.iteration % step_interval == 0
            or self.smoke_test
        )

    @property
    def evaluation_functions(self) -> Optional[Dict[str, Callable]]:
        return None

    def init_data_state(self) -> None:
        """
        This function is done post-initialization because otherwise the dataloaders are pickled with the object when running distributed.
        """
        self.train_dataloader = self.make_dataloader_train(self.run_config)
        self.val_dataloader = self.make_dataloader_val(self.run_config)
        self.test_dataloader = self.make_dataloader_test(self.run_config)

    def _update_learning_rate(self):
        self.metrics.lr = tutils.get_lr(self.optimizer)
        return self.learning_rate

    def _inc_iter(self):
        self.metrics.current_iteration += 1

    @property
    def uid(self):
        return self.run_config.uid

    def get_process_name(self) -> str:
        if hasattr(self, "model_dir") and self.model_dir is not None:
            proc_title = self.model_dir.relative_to(self.experiment_dir.parent).as_posix()
        else:
            proc_title = self.uid
        return proc_title

    def apply_loss(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        optimizer: Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        scheduler: Optional[Scheduler],
    ):
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

    def create_model(
        self, device=None, use_amp=None, init_chkpt=None, **kwargs
    ) -> Iterable[Union[nn.Module, nn.Module, nn.Module, nn.Module]]:
        if device is None:
            device = self.device
        if use_amp is None:
            use_amp = self.train_config.amp

        if init_chkpt is not None:
            save_dict = torch.load(init_chkpt, map_location="cpu")
            model_state = save_dict["state_dict"]

        else:
            model_state = None
        optimizer_config = self.train_config.optimizer_config
        scheduler_config = self.train_config.scheduler_config
        if scheduler_config is not None:
            scheduler_config.make_args(self)
        if self.train_mode == TrainMode.data_parallel:
            assert issubclass(type(self.run_config), DPConfig)
            device = self.run_config.device_ids  # type: ignore

        model, optimizer, scaler, scheduler = mutils.create_model(
            model_class=self.model_class,
            model_config=self.model_config,
            model_state=model_state,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            use_amp=use_amp,
            device=device,
            init_mode=self.train_mode,
            **kwargs,
        )
        return model, optimizer, scaler, scheduler

    def reset_optimizer_scheduler(self):
        optimizer_config = self.train_config.optimizer_config
        scheduler_config = self.train_config.scheduler_config
        optimizer, scheduler = mutils.create_optimizer_scheduler(
            self.model,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

    def load_model(self, save_dict, device=None, use_amp=None, ignore_diffs=False):

        if not hasattr(self, "run_config") or self.run_config is None:
            raise NotImplementedError(
                "Can not load model on an unitialzed model state. Consider run init_experiment_state function first"
            )

        run_config = type(self.run_config)(**save_dict["run_config"])
        if not ignore_diffs:
            run_config.assert_state(self.run_config)
            self.run_config.assert_state(run_config)

        self.run_config = run_config.merge(self.run_config, force=ignore_diffs)

        self.metrics.update(save_dict["metrics"])

        if device is None:
            device = self.device

        if use_amp is None:
            use_amp = self.train_config.amp

        if self.train_mode == TrainMode.data_parallel:
            assert issubclass(type(self.run_config), DPConfig)
            device = self.run_config.device_ids  # type: ignore

        (model, optimizer, scaler, scheduler, save_dict,) = mutils.load_model(
            model_class=self.model_class,
            model_config=self.model_config,
            scheduler_config=self.train_config.scheduler_config,
            optimizer_config=self.train_config.optimizer_config,
            save_dict=save_dict,
            device=device,
            use_amp=use_amp,
            init_mode=self.train_mode,
        )
        return (model, optimizer, scaler, scheduler, save_dict)

    def custom_evaluation(
        self, model: nn.Module, dataloader: Iterable
    ) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def make_dataloader_train(self, run_config: RunConfig):
        pass

    def make_dataloader_test(self, run_config: RunConfig):
        pass

    def make_dataloader_val(self, run_config: RunConfig):
        pass

    def config_parser(self, config: RunConfig):
        """
        Used to initialize Derived properties
        """
        return config

    def update_save_dict(self, **kwargs):

        if (
            self.train_mode == TrainMode.dist_data_parallel
            or self.train_mode == TrainMode.data_parallel
        ):
            model_state_dict = self.model.module.state_dict()
        else:
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
        save_dict = {
            "run_config": self.run_config.to_dict(),
            "state_dict": model_state_dict,
            "optimizer": optimizer_state_dict,
            "scheduler": scheduler_state_dict,
            "scaler": scaler_state_dict,
            "metrics": self.metrics.to_dict(),
        }
        self.save_dict.update(**save_dict)
        self.save_dict.update(**kwargs)

    def checkpoint(self, is_best=False):

        self.update_save_dict()
        filename = f"checkpoint_{self.iteration:010}.pt"
        self.logger.checkpoint(
            self.save_dict, filename, "best" if is_best else "recent"
        )

import typing as ty
from abc import abstractmethod

from torch import nn
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, StepLR, _LRScheduler

from ablator.config.main import ConfigBase, Derived, configclass
from ablator.config.types import Literal

Scheduler = ty.Union[_LRScheduler, ReduceLROnPlateau, ty.Any]

StepType = Literal["train", "val", "epoch"]


@configclass
class SchedulerArgs(ConfigBase):
    # step every train step or every validation step
    step_when: StepType

    @abstractmethod
    def init_scheduler(self, model, optimizer):
        pass


@configclass
class SchedulerConfig(ConfigBase):
    name: str
    arguments: SchedulerArgs

    def __init__(self, name, arguments: dict[str, ty.Any]):
        _arguments: None | StepLRConfig | OneCycleConfig | PlateuaConfig
        if (argument_cls := SCHEDULER_CONFIG_MAP[name]) is None:
            _arguments = StepLRConfig(gamma=1)
        else:
            _arguments = argument_cls(**arguments)
        super().__init__(name=name, arguments=_arguments)

    def make_scheduler(self, model, optimizer) -> Scheduler:
        return self.arguments.init_scheduler(model, optimizer)


@configclass
class OneCycleConfig(SchedulerArgs):
    max_lr: float
    total_steps: Derived[int]
    # TODO fix mypy errors for custom types
    # type: ignore
    step_when: StepType = "train"

    def init_scheduler(self, model: nn.Module, optimizer: nn.Module):
        kwargs = self.to_dict()
        del kwargs["step_when"]

        return OneCycleLR(optimizer, **kwargs)


@configclass
class PlateuaConfig(SchedulerArgs):
    patience: int = 10
    min_lr: float = 1e-5
    mode: str = "min"
    factor: float = 0.0
    threshold: float = 1e-4
    verbose: bool = False
    # TODO fix mypy errors for custom types
    # type: ignore
    step_when: StepType = "val"

    def init_scheduler(self, model: nn.Module, optimizer: nn.Module):
        kwargs = self.to_dict()
        del kwargs["step_when"]

        return ReduceLROnPlateau(optimizer, **kwargs)


@configclass
class StepLRConfig(SchedulerArgs):
    step_size: int = 1
    gamma: float = 0.99
    # TODO fix mypy errors for custom types
    # type: ignore
    step_when: StepType = "epoch"

    def init_scheduler(self, model: nn.Module, optimizer: nn.Module):
        kwargs = self.to_dict()
        del kwargs["step_when"]
        return StepLR(optimizer, **kwargs)


SCHEDULER_CONFIG_MAP = {
    "none": None,
    "step": StepLRConfig,
    "cycle": OneCycleConfig,
    "plateau": PlateuaConfig,
}

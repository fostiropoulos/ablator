from dataclasses import dataclass
from typing import Annotated, Any, Callable, Dict, Literal, Optional, Type, Union

from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, _LRScheduler, StepLR
from trainer.config.main import ConfigBase, Derived

Scheduler = Union[_LRScheduler, ReduceLROnPlateau, Any]

StepType = Literal["train", "val", "epoch"]


@dataclass(init=False, repr=False)
class SchedulerArgs(ConfigBase):
    scheduler_class: Type
    scheduler_init_fn: Callable
    # step every train step or every validation step
    step_when: StepType

    def make_args(self, wrapper):
        pass

    def init_scheduler(self, optimizer):
        kwargs = self.make_dict(self.annotations)
        del kwargs["step_when"]
        scheduler = self.scheduler_init_fn(optimizer, **kwargs)
        scheduler.step_when = self.step_when
        return scheduler

    def make_dict(self, *args, **kwargs):
        _dict = super().make_dict(*args, **kwargs)
        # TODO use-case for class-type values that are non-configurable
        del _dict["scheduler_class"]
        del _dict["scheduler_init_fn"]
        # del _dict["step_when"]
        return _dict


@dataclass(init=False, repr=False)
class SchedulerConfig(ConfigBase):
    name: str
    arguments: SchedulerArgs

    def __init__(self, name, arguments: Optional[Dict[str, Any]] = None):
        argument_cls = SCHEDULER_CONFIG_MAP[name]
        self.name = name
        if arguments is None:
            arguments = {}
        self.arguments = argument_cls(**arguments)

    def make_args(self, wrapper):
        return self.arguments.make_args(wrapper)

    def make_scheduler(self, optimizer):
        return self.arguments.init_scheduler(optimizer)

# TODO refactor schedulers as examples
# TODO make a consistent make_args make_scheduler that works for optimizer and config.

@dataclass
class CosSchedulerConfig(SchedulerArgs):
    steps_per_epoch: Annotated[int, Derived] = 0
    total_steps: Annotated[Optional[int], Derived] = None
    mode: str = "fix"
    min_lr: float = 4.0e-05
    buffer_epochs: float = 0
    warmup_epochs: float = 0.5
    multiplier: float = 0
    start_from_zero: bool = True
    # TODO how do i combine?
    # scheduler_class: Type = CosScheduler
    # scheduler_init_fn: Callable = create_cosine_scheduler
    step_when: StepType = "train"

    def make_args(self, wrapper):
        steps_per_epoch = len(wrapper.train_dataloader)
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = wrapper.total_steps

@dataclass
class OneCycleConfig(SchedulerArgs):
    max_lr: Annotated[Optional[float], Derived] = None
    find_lr: bool = True
    total_steps: Annotated[Optional[int], Derived] = None
    scheduler_class: Type = OneCycleLR
    scheduler_init_fn: Callable = OneCycleLR
    step_when: StepType = "train"



    def make_args(self, wrapper):
        from trainer import ModelWrapper

        wrapper: ModelWrapper
        self.total_steps = wrapper.total_steps
        if wrapper.smoke_test and self.find_lr:
            # NOTE: it is an expensive operation to find_lr during a smoke_test
            self.max_lr = 1e-3
        elif self.find_lr and not wrapper.train_config.resume:
            max_lr = find_lr(wrapper, verbose = wrapper.verbose)

            # max_lr = find_lr(wrapper)
            wrapper.logger.info(f"Found max_lr {max_lr:.4f}")
            self.max_lr = max_lr

    def make_dict(self, *args, **kwargs):
        _dict = super().make_dict(*args, **kwargs)
        # TODO add IgnoreDict Annotation
        del _dict["find_lr"]
        return _dict

@dataclass
class PlateuaConfig(SchedulerArgs):
    patience: int = 10
    min_lr: float = 1e-5
    mode: str = "min"
    factor: float = 0.0
    threshold: float = 1e-4
    verbose: bool = False
    scheduler_class: Type = ReduceLROnPlateau
    scheduler_init_fn: Callable = ReduceLROnPlateau
    step_when: StepType = "val"


@dataclass
class StepLRConfig(SchedulerArgs):
    step_size: int = 1
    gamma: float = 0.99
    scheduler_class: Type = StepLR
    scheduler_init_fn: Callable = StepLR
    step_when: StepType = "epoch"


SCHEDULER_CONFIG_MAP = {
    "step": StepLRConfig,
    "super": OneCycleConfig,
    "plateau": PlateuaConfig,
    "cosine": CosSchedulerConfig,
}

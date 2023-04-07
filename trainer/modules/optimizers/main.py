import torch

from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Type
from trainer.config.main import ConfigBase, configclass


from torch.optim import SGD, Adam, AdamW

from torch import nn

from trainer.modules.optimizers.utils import get_optim_parameters



@configclass
class OptimizerArgs(ConfigBase):
    lr: float
    optimizer_class: Type
    optimizer_init_fn: Callable

    def make_args(self, wrapper):
        pass

    def init_optimizer(self, model_parameters: Iterator[nn.Parameter]):
        kwargs = self.make_dict(self.annotations)
        return self.optimizer_init_fn(model_parameters, **kwargs)

    def make_dict(self, *args, **kwargs):
        _dict = super().make_dict(*args, **kwargs)
        del _dict["optimizer_class"]
        del _dict["optimizer_init_fn"]
        return _dict


@configclass
class OptimizerConfig(ConfigBase):
    name: str
    arguments: OptimizerArgs

    def __init__(self, name, arguments: Optional[Dict[str, Any]] = None):
        argument_cls = OPTIMIZER_CONFIG_MAP[name]
        self.name = name
        if arguments is None:
            arguments = {}
        self.arguments = argument_cls(**arguments)

    def make_args(self, wrapper):
        return self.arguments.make_args(wrapper)

    def make_optimizer(self, model: nn.Module):
        weight_decay = getattr(self.arguments, "weight_decay", None)
        # 1e-4
        model_parameters = get_optim_parameters(model, weight_decay)
        return self.arguments.init_optimizer(model_parameters)

    @property
    def lr(self):
        return self.arguments.lr

    @lr.setter
    def lr(self, val):
        self.arguments.lr = val


@configclass
class SGDConfig(OptimizerArgs):
    optimizer_class: Type = SGD
    optimizer_init_fn: Callable = SGD
    weight_decay: float = 0.0
    momentum: float = 0.0


@configclass
class AdamWConfig(OptimizerArgs):
    optimizer_class: Type = AdamW
    optimizer_init_fn: Callable = AdamW
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


@configclass
class AdamConfig(OptimizerArgs):
    optimizer_class: Type = Adam
    optimizer_init_fn: Callable = Adam
    betas: Tuple[float, float] = (0.5, 0.9)
    weight_decay: float = 0.0


OPTIMIZER_CONFIG_MAP = {"adamw": AdamWConfig, "adam": AdamConfig, "sgd": SGDConfig}

import typing as ty
from abc import abstractmethod

import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer

from ablator.config.main import ConfigBase, configclass
from ablator.config.types import Tuple


def get_parameter_names(model: torch.nn.Module, forbidden_layer_types: list[type]):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def get_optim_parameters(
    model: torch.nn.Module,
    weight_decay: float | None = None,
    only_requires_grad: bool = True,
):
    """
    Setup the optimizer.
    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
    """
    # default_val = lambda k, v: kwargs[k] if k in kwargs else v

    params_to_update = {}
    if only_requires_grad:
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update[name] = param
    else:
        params_to_update = model.named_parameters()
    if weight_decay is not None:
        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
        decay_parameters = [
            name
            for name in decay_parameters
            if "bias" not in name and name in params_to_update
        ]
        optimization_params = [
            {
                "params": [
                    p for n, p in params_to_update.items() if n in decay_parameters
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in params_to_update.items() if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimization_params
    return list(params_to_update.values())


@configclass
class OptimizerArgs(ConfigBase):
    lr: float

    @abstractmethod
    def init_optimizer(self, model: nn.Module):
        pass


@configclass
class OptimizerConfig(ConfigBase):
    name: str
    arguments: OptimizerArgs

    def __init__(self, name, arguments: dict[str, ty.Any]):
        argument_cls = OPTIMIZER_CONFIG_MAP[name]
        _arguments = argument_cls(**arguments)
        super().__init__(name=name, arguments=_arguments)

    def make_optimizer(self, model: nn.Module) -> Optimizer:
        return self.arguments.init_optimizer(model)


@configclass
class SGDConfig(OptimizerArgs):
    weight_decay: float = 0.0
    momentum: float = 0.0

    def init_optimizer(self, model: nn.Module):
        kwargs = self.to_dict()
        weight_decay = getattr(self, "weight_decay", None)
        # 1e-4
        model_parameters = get_optim_parameters(model, weight_decay)
        return SGD(model_parameters, **kwargs)


@configclass
class AdamWConfig(OptimizerArgs):
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    def init_optimizer(self, model: nn.Module):
        kwargs = self.to_dict()
        weight_decay = getattr(self, "weight_decay", None)
        # 1e-4
        model_parameters = get_optim_parameters(model, weight_decay)
        return AdamW(model_parameters, **kwargs)


@configclass
class AdamConfig(OptimizerArgs):
    betas: Tuple[float, float] = (0.5, 0.9)
    weight_decay: float = 0.0

    def init_optimizer(self, model: nn.Module):
        kwargs = self.to_dict()
        weight_decay = getattr(self, "weight_decay", None)
        # 1e-4
        model_parameters = get_optim_parameters(model, weight_decay)
        return Adam(model_parameters, **kwargs)


OPTIMIZER_CONFIG_MAP: dict[str, type] = {
    "adamw": AdamWConfig,
    "adam": AdamConfig,
    "sgd": SGDConfig,
}

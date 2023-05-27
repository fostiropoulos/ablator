import typing as ty
from abc import abstractmethod

import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer

from ablator.config.main import ConfigBase, configclass
from ablator.config.types import Tuple


def get_parameter_names(model: torch.nn.Module, forbidden_layer_types: list[type]):
    """
    Recurse into the module and return parameter names of all submodules, excluding
    modules that are of any type defined in ``forbidden_layer_types``.

    Parameters
    ----------
    model : torch.nn.Module
        The model for which to get parameter names.
    forbidden_layer_types : list[type]
        A list of types of modules inside which parameter names should not be included.

    Returns
    -------
    list[str]
        The names of the parameters with the following format: ``<submodule-name>.<parameter-name>``.

    Examples
    --------
    >>> class MyModel(nn.Module):
    >>>     def __init__(self, embedding_dim=10, vocab_size=10, *args, **kwargs) -> None:
    >>>         super().__init__(*args, **kwargs)
    >>>         self.param = nn.Parameter(torch.ones(100))
    >>>         self.embedding = nn.Embedding(num_embeddings=vocab_size,
    >>>                                     embedding_dim=embedding_dim)
    >>>         self.norm_layer = nn.LayerNorm(embedding_dim)
    >>>     def forward(self):
    >>>         x = self.param + torch.rand_like(self.param) * 0.01
    >>>         return x.sum().abs()
    >>> mM = MyModel()
    >>> get_parameter_names(mM,[])
    ['embedding.weight', 'norm_layer.weight', 'norm_layer.bias', 'param']
    >>> get_parameter_names(mM, [torch.nn.LayerNorm])
    ['embedding.weight', 'param']
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
    Setup the optimizer. Get model parameters to be optimized. If ``weight_decay`` is a ``float``,
    apply weight decaying to the parameters too (except for bias and parameters from layer
    normalization module).

    Parameters
    ----------
    model : torch.nn.Module
        The model for which to get parameters that will be optimized.
    weight_decay : float | None
        The amount of weight decay to use, by default ``None``.
    only_requires_grad : bool
        Whether to only use parameters that require gradient or all parameters, by default ``True``.

    Returns
    -------
    dict | list
        - If weight_decay is ``None``, return all model parameters.

        - If weight_decay is not ``None``, return a dictionary of parameter groups of different weight decay. In specific, bias parameters and parameters from layer normalization module will have weight decay of ``0.0``, while any other parameters will have weight decay of ``weight_decay``.

    Notes
    -----
    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.

    Examples
    --------
    >>> class MyModel(nn.Module):
    >>>     def __init__(self, embedding_dim=10, vocab_size=10, *args, **kwargs) -> None:
    >>>         super().__init__(*args, **kwargs)
    >>>         self.param = nn.Parameter(torch.ones(100))
    >>>         self.embedding = nn.Embedding(num_embeddings=vocab_size,
    >>>                                     embedding_dim=embedding_dim)
    >>>         self.norm_layer = nn.LayerNorm(embedding_dim)
    >>>     def forward(self):
    >>>         x = self.param + torch.rand_like(self.param) * 0.01
    >>>         return x.sum().abs()
    >>> mM = MyModel()
    >>> get_optim_parameters(mM, 0.2)
    [
        {'params': ['param', 'embedding.weight'], 'weight_decay': 0.2},
        {'params': ['norm_layer.weight', 'norm_layer.bias'], 'weight_decay': 0.0}
    ]
    """
    # default_val = lambda k, v: kwargs[k] if k in kwargs else v

    params_to_update = {}
    if only_requires_grad:
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update[name] = param
    else:
        params_to_update = dict(model.named_parameters())
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
    """
    A base class for optimizer arguments, here we define learning rate lr.

    Attributes
    ----------
    lr : float
        Learning rate of the optimizer
    """
    lr: float

    @abstractmethod
    def init_optimizer(self, model: nn.Module):
        """
        Abstract method to be implemented by derived classes, which initializes the optimizer.
        """
        pass


@configclass
class OptimizerConfig(ConfigBase):
    """
    Configuration for an optimizer, including optimizer name and arguments (these arguments
    are specific to a certain type of optimizer like SGD, Adam, AdamW).

    Attributes
    ----------
    name : str
        Name of the optimizer.
    arguments : OptimizerArgs
        Arguments for the optimizer, specific to a certain type of optimizer.
    """
    name: str
    arguments: OptimizerArgs

    def __init__(self, name, arguments: dict[str, ty.Any]):
        """
        Initializes the optimizer configuration. Add any provided settings to the optimizer.

        Parameters
        ----------
        name : str
            Name of the optimizer, this can be any in ``['adamw', 'adam', 'sgd']``.
        arguments : dict[str, ty.Any]
            Arguments for the optimizer, specific to a certain type of optimizer. A common argument
            can be learning rate, e.g ``{'lr': 0.5}``. If ``name`` is ``"adamw"``, can add ``eps`` to ``arguments``,
            e.g ``{'lr': 0.5, 'eps': 0.001}``.

        Examples
        --------

        In the following example, ``optim_config`` will initialize property ``arguments`` of type ``SGDConfig``,
        setting ``lr=0.5`` as its property. We also have access to ``init_optimizer()`` method of the property,
        which initalizes an SGD optimizer. This method is actually called in ``make_optimizer()``

        >>> optim_config = OptimizerConfig("sgd", {"lr": 0.5})
        """
        argument_cls = OPTIMIZER_CONFIG_MAP[name]
        _arguments = argument_cls(**arguments)
        super().__init__(name=name, arguments=_arguments)

    def make_optimizer(self, model: nn.Module) -> Optimizer:
        """
        Creates and returns an optimizer for the given model.

        Parameters
        ----------
        model : torch.nn.Module
            The model to optimize.

        Returns
        -------
        optimizer : torch.optim.Optimizer
            The created optimizer.

        Examples
        --------
        >>> optim_config = OptimizerConfig("sgd", {"lr": 0.5, "weight_decay": 0.5})
        >>> optim_config.make_optimizer(my_module)
        SGD (
        Parameter Group 0
            dampening: 0
            differentiable: False
            foreach: None
            lr: 0.5
            maximize: False
            momentum: 0.0
            nesterov: False
            weight_decay: 0.5
        Parameter Group 1
            dampening: 0
            differentiable: False
            foreach: None
            lr: 0.5
            maximize: False
            momentum: 0.0
            nesterov: False
            weight_decay: 0.0
        )
        """
        return self.arguments.init_optimizer(model)


@configclass
class SGDConfig(OptimizerArgs):
    """
    Configuration for an SGD optimizer. This class has ``init_optimizer()`` method,
    which is used to initialize and return an SGD optimizer.

    Attributes
    ----------
    weight_decay : float
        Weight decay rate.
    momentum : float
        Momentum factor.

    Examples
    --------
    >>> config = SGDConfig(lr=0.1, momentum=0.9)
    """
    weight_decay: float = 0.0
    momentum: float = 0.0

    def init_optimizer(self, model: nn.Module):
        """
        Creates and returns an SGD optimizer that optimizes the model's parameters. These parameters
        will be processed via ``get_optim_parameters`` before used to initalized the optimizer.

        Parameters
        ----------
        model : torch.nn.Module
            The model that has parameters that the optimizer will optimize.

        Returns
        -------
        optimizer : torch.optim.SGD
            The created SGD optimizer.

        Examples
        --------
        >>> config = SGDConfig(lr=0.1, weight_decay=0.5, momentum=0.9)
        >>> config.init_optimizer(MyModel())
        SGD (
        Parameter Group 0
            dampening: 0
            differentiable: False
            foreach: None
            lr: 0.1
            maximize: False
            momentum: 0.9
            nesterov: False
            weight_decay: 0.5
        Parameter Group 1
            dampening: 0
            differentiable: False
            foreach: None
            lr: 0.1
            maximize: False
            momentum: 0.9
            nesterov: False
            weight_decay: 0.0
        )
        """
        kwargs = self.to_dict()
        weight_decay = getattr(self, "weight_decay", None)
        # 1e-4
        model_parameters = get_optim_parameters(model, weight_decay)
        return SGD(model_parameters, **kwargs)


@configclass
class AdamWConfig(OptimizerArgs):
    """
    Configuration for an AdamW optimizer. This class has ``init_optimizer()`` method
    used to initialize and return an ``AdamW`` optimizer.

    Attributes
    ----------
    betas : Tuple[float, float]
        Coefficients for computing running averages of gradient and its square (default is ``(0.9, 0.999)``).
    eps : float
        Term added to the denominator to improve numerical stability (default is ``1e-8``).
    weight_decay : float
        Weight decay rate (default is ``0.0``).

    Examples
    --------
    >>> config = AdamWConfig(lr=0.1, weight_decay=0.5, betas=(0.9,0.99))
    """
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    def init_optimizer(self, model: nn.Module):
        """
        Creates and returns an ``AdamW`` optimizer that optimizes the model's parameters. These parameters
        will be processed via ``get_optim_parameters`` before used to initalized the optimizer.

        Parameters
        ----------
        model : torch.nn.Module
            The model that has parameters that the optimizer will optimize.

        Returns
        -------
        Optimizer
            An instance of the ``AdamW`` optimizer.

        Examples
        --------
        >>> config = AdamWConfig(lr=0.1, weight_decay=0.5, betas=(0.9,0.99), eps=0.001)
        >>> config.init_optimizer(MyModel())
        AdamW (
        Parameter Group 0
            amsgrad: False
            betas: (0.9, 0.99)
            capturable: False
            eps: 0.001
            foreach: None
            lr: 0.1
            maximize: False
            weight_decay: 0.5
        Parameter Group 1
            amsgrad: False
            betas: (0.9, 0.99)
            capturable: False
            eps: 0.001
            foreach: None
            lr: 0.1
            maximize: False
            weight_decay: 0.0
        )
        """
        kwargs = self.to_dict()
        weight_decay = getattr(self, "weight_decay", None)
        # 1e-4
        model_parameters = get_optim_parameters(model, weight_decay)
        return AdamW(model_parameters, **kwargs)


@configclass
class AdamConfig(OptimizerArgs):
    """
    Configuration for an ``Adam`` optimizer. This class has ``init_optimizer()`` method
    used to initialize and return an ``Adam`` optimizer.

    Attributes
    ----------
    betas : Tuple[float, float]
        Coefficients for computing running averages of gradient and its square (default is ``(0.5, 0.9)``).
    weight_decay : float
        Weight decay rate (default is ``0.0``).

    """
    betas: Tuple[float, float] = (0.5, 0.9)
    weight_decay: float = 0.0

    def init_optimizer(self, model: nn.Module):
        """
        Creates and returns an ``Adam`` optimizer that optimizes the model's parameters. These parameters
        will be processed via ``get_optim_parameters`` before used to initalized the optimizer.

        Parameters
        ----------
        model : torch.nn.Module
            The model that has parameters that the optimizer will optimize.

        Returns
        -------
        Optimizer
            An instance of the ``Adam`` optimizer.

        Examples
        --------
        >>> config = AdamConfig(lr=0.1, weight_decay=0.5, betas=(0.6,0.9))
        >>> config.init_optimizer(MyModel())
        Adam (
        Parameter Group 0
            amsgrad: False
            betas: (0.6, 0.9)
            capturable: False
            differentiable: False
            eps: 1e-08
            foreach: None
            fused: False
            lr: 0.1
            maximize: False
            weight_decay: 0.5
        Parameter Group 1
            amsgrad: False
            betas: (0.6, 0.9)
            capturable: False
            differentiable: False
            eps: 1e-08
            foreach: None
            fused: False
            lr: 0.1
            maximize: False
            weight_decay: 0.0
        )
        """
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

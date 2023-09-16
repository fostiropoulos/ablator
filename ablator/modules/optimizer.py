from collections import abc
import inspect
import typing as ty
from abc import abstractmethod

import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer

from ablator.config.main import ConfigBase, configclass
from ablator.config.types import Tuple


def get_optim_parameters(
    model: torch.nn.Module,
) -> abc.Iterator[nn.Parameter]:
    """
    Get model parameters to be optimized. It first attempts to derive optimization parameters
    via a user-defined `get_optim_param` function which when it fails to find it simply uses the
    default torch `nn.parameters()`

    Parameters
    ----------
    model : torch.nn.Module
        The model for which to get parameters that will be optimized.

    Returns
    -------
    abc.Iterator[nn.Parameter]
        The list of parameters that require to be optimized. It can be a list, tensor or dictionary. Please see
        Pytorch Optimizer documentation on the specific format.

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
    >>>     def get_optim_param(self):
    >>>         return [{"params": [self.param], 'weight_decay':0.2}]
    >>> mM = MyModel()
    >>> get_optim_parameters(mM)
    [{'params': ['param'], 'weight_decay': 0.2}]
    """
    _model = model
    if isinstance(model, nn.DataParallel):
        _model = model.module
    fn = getattr(_model, "get_optim_param", None)
    if fn is not None and inspect.ismethod(fn):
        return fn()
    return model.parameters()


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
        raise NotImplementedError("init_optimizer method not implemented.")


@configclass
class OptimizerConfig(ConfigBase):
    """
    Configuration for an optimizer, including optimizer name and arguments (these arguments
    are specific to a certain type of optimizer like SGD, Adam, AdamW). This optimizer config
    will be provided to ``TrainConfig`` as part of the training setting of the experiment.

    Parameters
    ----------
    name : str
        Name of the optimizer, this can be any in ``['adamw', 'adam', 'sgd']``.
    arguments : dict[str, ty.Any]
        Arguments for the optimizer, specific to a certain type of optimizer. A common argument
        can be learning rate, e.g ``{'lr': 0.5}``. If ``name`` is ``"adamw"``, can add ``eps`` to ``arguments``,
        e.g ``{'lr': 0.5, 'eps': 0.001}``. Refer to  `Configuration Basics
        scheduler <./notebooks/Configuration-Basics.ipynb>`_ tutorial for more details on each optimizer's arguments.

    Attributes
    ----------
    name : str
        Name of the optimizer.
    arguments : OptimizerArgs
        Arguments for the optimizer, specific to a certain type of optimizer.

    Examples
    --------
    The following example shows how to create an optimizer config for SGD optimizer and use it in
    ``TrainConfig`` to define the training setting of the experiment.

    >>> optim_config = OptimizerConfig("sgd", {"lr": 0.5})
    >>> train_config = TrainConfig(
    ...     dataset="[Dataset Name]",
    ...     batch_size=32,
    ...     epochs=20,
    ...     optimizer_config=optim_config,
    ...     scheduler_config=None
    ... )
    >>> # ... create the run config (proto/parallel), model wrapper, trainer and launch the experiment

    .. note::
        Sometimes we want to run ablation studies on different optimizers to learn about their
        effects on the model performance. However, ``OptimizerConfig`` only configures one single
        optimizer for the experiment. But you can run experiments on different optimizers by creating
        a custom config class and add an extra method called ``make_optimizer``. Go to the tutorial on
        `Search space for different types of optimizers and
        scheduler <./notebooks/Searchspace-for-diff-optimizers.ipynb>`_ for more details.

    """

    name: str
    arguments: OptimizerArgs

    def __init__(self, name: str, arguments: dict[str, ty.Any]):
        # Initializes the optimizer configuration. Add any provided settings to the optimizer.
        argument_cls = OPTIMIZER_CONFIG_MAP[name]
        _arguments = argument_cls(**arguments)
        super().__init__(name=name, arguments=_arguments)

    def make_optimizer(self, model: nn.Module) -> Optimizer:
        """
        Creates and returns an optimizer for the given model.

        Parameters
        ----------
        model : nn.Module
            The model to optimize.

        Returns
        -------
        optimizer : Optimizer
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

    def init_optimizer(self, model: nn.Module) -> SGD:
        """
        Creates and returns an SGD optimizer that optimizes the model's parameters. These parameters
        will be processed via ``get_optim_parameters`` before used to initalized the optimizer.

        Parameters
        ----------
        model : nn.Module
            The model that has parameters that the optimizer will optimize.

        Returns
        -------
        optimizer : SGD
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
        model_parameters = get_optim_parameters(model)
        return SGD(model_parameters, **kwargs)


@configclass
class AdamWConfig(OptimizerArgs):
    """
    Configuration for an AdamW optimizer. This class has ``init_optimizer()`` method
    used to initialize and return an ``AdamW`` optimizer.

    Attributes
    ----------
    betas : Tuple[float, float]
        Coefficients for computing running averages of gradient and its square, by default ``(0.9, 0.999)``.
    eps : float
        Term added to the denominator to improve numerical stability, by default ``1e-8``.
    weight_decay : float
        Weight decay rate, by default ``0.01``.

    Examples
    --------
    >>> config = AdamWConfig(lr=0.1, weight_decay=0.5, betas=(0.9,0.99))
    """

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01

    def init_optimizer(self, model: nn.Module) -> AdamW:
        """
        Creates and returns an ``AdamW`` optimizer that optimizes the model's parameters. These parameters
        will be processed via ``get_optim_parameters`` before used to initalized the optimizer.

        Parameters
        ----------
        model : nn.Module
            The model that has parameters that the optimizer will optimize.

        Returns
        -------
        AdamW
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
        # 1e-4
        model_parameters = get_optim_parameters(model)
        return AdamW(model_parameters, **kwargs)


@configclass
class AdamConfig(OptimizerArgs):
    """
    Configuration for an ``Adam`` optimizer. This class has ``init_optimizer()`` method
    used to initialize and return an ``Adam`` optimizer.

    Attributes
    ----------
    betas : Tuple[float, float]
        Coefficients for computing running averages of gradient and its square, by default ``(0.9, 0.999)``.
    weight_decay : float
        Weight decay rate, by default ``0.0``.

    """

    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0

    def init_optimizer(self, model: nn.Module) -> Adam:
        """
        Creates and returns an ``Adam`` optimizer that optimizes the model's parameters. These parameters
        will be processed via ``get_optim_parameters`` before used to initalized the optimizer.

        Parameters
        ----------
        model : nn.Module
            The model that has parameters that the optimizer will optimize.

        Returns
        -------
        Adam
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
        model_parameters = get_optim_parameters(model)
        return Adam(model_parameters, **kwargs)


OPTIMIZER_CONFIG_MAP: dict[str, type] = {
    "adamw": AdamWConfig,
    "adam": AdamConfig,
    "sgd": SGDConfig,
}

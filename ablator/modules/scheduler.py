import typing as ty
from abc import abstractmethod

from torch import nn
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, StepLR, _LRScheduler
from torch.optim import Optimizer

from ablator.config.main import ConfigBase, Derived, configclass


Scheduler = ty.Union[_LRScheduler, ReduceLROnPlateau, ty.Any]

StepType = ty.Literal["train", "val", "epoch"]


@configclass
class SchedulerArgs(ConfigBase):
    """
    Abstract base class for defining arguments to initialize a learning rate scheduler.

    Attributes
    ----------
    step_when : StepType
        The step type at which the scheduler.step() should be invoked: ``'train'``, ``'val'``, or ``'epoch'``.

    """

    # step every train step or every validation step
    step_when: StepType

    @abstractmethod
    def init_scheduler(self, model, optimizer):
        """
        Abstract method to be implemented by derived classes, which creates and returns a scheduler object.
        """
        raise NotImplementedError("init_scheduler method not implemented.")


@configclass
class SchedulerConfig(ConfigBase):
    """
    A class that defines a configuration for a learning rate scheduler. This scheduler config
    will be provided to ``TrainConfig`` (optional) as part of the training setting of the experiment.

    Attributes
    ----------
    name : str
        The name of the scheduler.
    arguments : SchedulerArgs
        The arguments needed to initialize the scheduler.

    Examples
    --------
    The following example shows how to create a scheduler config and use it in
    ``TrainConfig`` to define the training setting of the experiment.
    
    >>> optim_config = OptimizerConfig("sgd", {"lr": 0.5})
    >>> scheduler_config = SchedulerConfig("step", arguments={"step_size": 1, "gamma": 0.99})
    >>> train_config = TrainConfig(
    ...     dataset="[Dataset Name]",
    ...     batch_size=32,
    ...     epochs=20,
    ...     optimizer_config=optim_config,
    ...     scheduler_config=scheduler_config,
    ...     rand_weights_init = True
    ... )
    >>> # ... create running config (proto/parallel), model wrapper, trainer and launch experiment

    .. note::
        A common use case is to run ablation studies on different schedulers to learn about their
        effects on the model performance. However, ``SchedulerConfig`` only configures one single
        scheduler for the experiment. But you can run experiments on different schedulers by creating
        a custom config class and add an extra method called ``make_scheduler``. Go to this tutorial on
        :ref:`Search space for different types of optimizers and scheduler <search_space_optim_schedule>`
        for more details.
    """

    name: str
    arguments: SchedulerArgs

    def __init__(self, name, arguments: dict[str, ty.Any]):
        """
        Initializes the scheduler configuration.

        Parameters
        ----------
        name : str
            The name of the scheduler, this can be any in ``['None', 'step', 'cycle', 'plateau']``.
        arguments : dict[str, ty.Any]
            The arguments for the scheduler, specific to a certain type of scheduler.

        Examples
        --------
        In the following example, ``scheduler_config`` will initialize property ``arguments`` of type ``StepLRConfig``,
        setting ``step_size=1``, ``gamma=0.99`` as its properties. We also have access to ``init_scheduler()`` method
        of the property, which initalizes an StepLR scheduler. This method is actually called in ``make_scheduler()``

        >>> scheduler_config = SchedulerConfig("step", arguments={"step_size": 1, "gamma": 0.99})
        """
        _arguments: None | StepLRConfig | OneCycleConfig | PlateuaConfig
        if (argument_cls := SCHEDULER_CONFIG_MAP[name]) is None:
            _arguments = StepLRConfig(gamma=1)
        else:
            _arguments = argument_cls(**arguments)
        super().__init__(name=name, arguments=_arguments)

    def make_scheduler(self, model: nn.Module, optimizer: Optimizer) -> Scheduler:
        """
        Creates a new scheduler for an optimizer, based on the configuration.

        Parameters
        ----------
        model: nn.Module
            Some schedulers require information from the model. The model is passed as an argument.
        optimizer
            The optimizer used to update the model parameters, whose learning rate we want to monitor.

        Returns
        -------
        Scheduler
            The scheduler.

        Examples
        --------
        >>> scheduler_config = SchedulerConfig("step", arguments={"step_size": 1, "gamma": 0.99})
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.7, momentum=0.9)
        >>> scheduler_config.make_scheduler(model, optimizer)
        """
        return self.arguments.init_scheduler(model, optimizer)


@configclass
class OneCycleConfig(SchedulerArgs):
    """
    Configuration class for the OneCycleLR scheduler.

    Attributes
    ----------
    max_lr : float
        Upper learning rate boundaries in the cycle.
    total_steps : Derived[int]
        The total number of steps to run the scheduler in a cycle.
    step_when : StepType
        The step type at which the scheduler.step() should be invoked: ``'train'``, ``'val'``, or ``'epoch'``.

    """

    max_lr: float
    total_steps: Derived[int]
    step_when: StepType = "train"

    def init_scheduler(self, model: nn.Module, optimizer: Optimizer):
        """
        Initializes the OneCycleLR scheduler.
        Creates and returns a OneCycleLR scheduler that monitors optimizer's learning rate.

        Parameters
        ----------
        model : nn.Module
            The model.
        optimizer : Optimizer
            The optimizer used to update the model parameters, whose learning rate we want to monitor.

        Returns
        -------
        OneCycleLR
            The OneCycleLR scheduler, initialized with arguments defined as attributes of this class.

        Examples
        --------
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.7, momentum=0.9)
        >>> scheduler = OneCycleConfig(max_lr=0.5, total_steps=100)
        >>> scheduler.init_scheduler(model, optimizer)

        """
        kwargs = self.to_dict()
        del kwargs["step_when"]

        return OneCycleLR(optimizer, **kwargs)


@configclass
class PlateuaConfig(SchedulerArgs):
    """Configuration class for ReduceLROnPlateau scheduler.

    Attributes
    ----------
        patience : int
            Number of epochs with no improvement after which learning rate will be reduced.
        min_lr : float
            A lower bound on the learning rate.
        mode : str
            One of ``'min'``, ``'max'``, or ``'auto'``, which defines the direction of optimization, so as
            to adjust the learning rate accordingly, i.e when a certain metric ceases improving.
        factor : float
            Factor by which the learning rate will be reduced. ``new_lr = lr * factor``.
        threshold : float
            Threshold for measuring the new optimum, to only focus on significant changes.
        verbose : bool
            If ``True``, prints a message to ``stdout`` for each update.
        step_when : StepType
            The step type at which the scheduler should be invoked: ``'train'``, ``'val'``, or ``'epoch'``.

    """

    patience: int = 10
    min_lr: float = 1e-5
    mode: str = "min"
    factor: float = 0.0  # TODO {fixme} this is error prone -> new_lr = 0
    threshold: float = 1e-4
    verbose: bool = False
    step_when: StepType = "val"

    def init_scheduler(self, model: nn.Module, optimizer: Optimizer):
        """
        Initialize the ReduceLROnPlateau scheduler.

        Parameters
        ----------
        model : nn.Module
            The model being optimized.
        optimizer : Optimizer
            The optimizer used to update the model parameters, whose learning
            rate we want to monitor.

        Returns
        -------
        ReduceLROnPlateau
            The ReduceLROnPlateau scheduler, initialized with arguments defined as
            attributes of this class.

        Examples
        --------
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.7, momentum=0.9)
        >>> scheduler = PlateuaConfig(min_lr=1e-7, mode='min')
        >>> scheduler.init_scheduler(model, optimizer)

        """
        kwargs = self.to_dict()
        del kwargs["step_when"]

        return ReduceLROnPlateau(optimizer, **kwargs)


@configclass
class StepLRConfig(SchedulerArgs):
    """
    Configuration class for StepLR scheduler.

    Parameters
    ----------
    step_size : int
        Period of learning rate decay, by default 1.
    gamma : float
        Multiplicative factor of learning rate decay, by default 0.99.
    step_when : StepType
        The step type at which the scheduler should be invoked: ``'train'``, ``'val'``, or ``'epoch'``.

    """

    step_size: int = 1
    gamma: float = 0.99
    step_when: StepType = "epoch"

    def init_scheduler(self, model: nn.Module, optimizer: Optimizer):
        """
        Initialize the StepLR scheduler for a given model and optimizer.

        Parameters
        ----------
        model : nn.Module
            The model to apply the scheduler.
        optimizer : Optimizer
            The optimizer used to update the model parameters, whose learning
            rate we want to monitor.

        Returns
        -------
        StepLR
            The StepLR scheduler, initialized with arguments defined as
            attributes of this class.

        Examples
        --------
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.7, momentum=0.9)
        >>> scheduler = StepLRConfig(step_size=20, gamma=0.9)
        >>> scheduler.init_scheduler(model, optimizer)

        """
        kwargs = self.to_dict()
        del kwargs["step_when"]
        return StepLR(optimizer, **kwargs)


SCHEDULER_CONFIG_MAP = {
    "none": None,
    "step": StepLRConfig,
    "cycle": OneCycleConfig,
    "plateau": PlateuaConfig,
}

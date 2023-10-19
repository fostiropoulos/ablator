from ablator.config.remote import RemoteConfig
from ablator.config.main import ConfigBase, configclass
from ablator.config.types import Dict, Optional, Stateless, Literal, Enum
from ablator.modules.optimizer import OptimizerConfig
from ablator.modules.scheduler import SchedulerConfig


class Optim(Enum):
    """
    Type of optimization direction.

    can take values `min` and `max` that indicate whether the HPO
    algorithm should minimize or maximize the corresponding metric.
    """

    min = "min"
    max = "max"


@configclass
class TrainConfig(ConfigBase):
    """
    Training configuration that defines the training setting, e.g., batch size, number of epochs,
    the optimizer to use, etc. This configuration is required when creating the run configurations
    (``RunConfig`` and ``ParallelConfig``, which set up the running environment of the experiment).

    Attributes
    ----------
    dataset: str
        Dataset name. maybe used in custom dataset loader functions.
    batch_size: int
        Batch size.
    epochs: int
        Number of epochs to train.
    optimizer_config: OptimizerConfig
        Optimizer configuration.
    scheduler_config: Optional[SchedulerConfig]
        Scheduler configuration.

    Examples
    --------
    The following example shows all the steps towards configuring an experiment:

    - Define model config: for simplicity, we use the default one with no custom hyperparameters
      (so we're not running an ablation study on the model architecture):

    >>> my_model_config = ModelConfig()

    - Define optimizer and scheduler config, as training config requires an optimizer
      config, and optionally a scheduler config:

    >>> my_optimizer_config = OptimizerConfig("sgd", {"lr": 0.5, "weight_decay": 0.5})
    >>> my_scheduler_config = SchedulerConfig("step", arguments={"step_size": 1, "gamma": 0.99})

    - Define training config:

    >>> my_train_config = TrainConfig(
    ...     dataset="[Your Dataset]",
    ...     batch_size=32,
    ...     epochs=10,
    ...     optimizer_config = my_optimizer_config,
    ...     scheduler_config = my_scheduler_config
    ... )

    - We now define the run config for prototype training, which is the last configuration step.
      Refer to :ref:`Configurations for single model experiments <run_config>` and
      :ref:`Configurations for parallel models experiments <parallel_config>` for more details on running configs.

    >>> run_config = RunConfig(
    ...     train_config=my_train_config,
    ...     model_config=my_model_config,
    ...     metrics_n_batches = 800,
    ...     experiment_dir = "/tmp/experiments",
    ...     device="cpu",
    ...     amp=False,
    ...     random_seed = 42
    ... )

    """

    dataset: str
    batch_size: int
    epochs: int
    optimizer_config: OptimizerConfig
    scheduler_config: Optional[SchedulerConfig]


# TODO decorator @modelconfig as opposed to @configclass ModelConfig
@configclass
class ModelConfig(ConfigBase):
    """
    A base class for model configuration. This is used for defining model hyperparameters,
    so when initializing a model, it is passed to the model module constructor. The attributes
    from the model config object will be used to construct the model.

    Examples
    --------
    Define a custom model configuration class for your model:

    >>> @configclass
    >>> class CustomModelConfig(ModelConfig):
    >>>     input_size :int
    >>>     hidden_size :int
    >>>     num_classes :int

    Define your model class, pass the configuration to the constructor, and build the model:

    >>> class FashionMNISTModel(nn.Module):
    >>>     def __init__(self, config: CustomModelConfig):
    >>>         super(FashionMNISTModel, self).__init__()
    >>>         self.fc1 = nn.Linear(config.input_size, config.hidden_size) # model config attributes are used here
    >>>         self.relu1 = nn.ReLU()
    >>>         self.fc3 = nn.Linear(config.hidden_size, config.num_classes) # model config attributes are used here
    >>>     def forward(self, x):
    >>>         # code for forward pass
    >>>         return x

    ``RunConfig`` later requires a model config object, so we will create one, remember to pass values
    to the hyperparameters as we defined them to be Stateful:

    >>> model_config = CustomModelConfig(input_size=512, hidden_size=100, num_classes=10)
    """


@configclass
class RunConfig(ConfigBase):
    """
    The base run configuration that defines the setting of an experiment (experiment main directory, number of
    checkpoints to maintain, hardware device to use, etc.). You can use this to configure the experiment of a
    single prototype model.

    ``RunConfig`` encapsulates every configuration (model config, optimizer-scheduler config, train config)
    needed for an experiment. This entire umbrella of configurations is then passed to ``ProtoTrainer`` which
    launches the prototype experiment.

    Attributes
    ----------
    experiment_dir: Stateless[Optional[str]]
        Location to store experiment artifacts, by default ``None``.
    random_seed: Optional[int]
        Random seed, by default ``None``.
    train_config: TrainConfig
        Training configuration.
    model_config: ModelConfig
        Model configuration.
    keep_n_checkpoints: Stateless[int]
        Number of latest checkpoints to keep, by default ``3``.
    tensorboard: Stateless[bool]
        Whether to use tensorboardLogger, by default ``True``.
    amp: Stateless[bool]
        Whether to use automatic mixed precision when running on gpu, by default ``True``.
    device: Stateless[str]
        Device to run on, by default ``"cuda"``.
    verbose: Stateless[Literal["console", "progress", "silent"]]
        Verbosity level, by default ``"console"``.
    eval_subsample: Stateless[float]
        Fraction of the dataset to use for evaluation, by default ``1``.
    metrics_n_batches: Stateless[int]
        Max number of batches stored in every tag(train, eval, test) for evaluation, by default ``32``.
    metrics_mb_limit: Stateless[int]
        Max number of megabytes stored in every tag(train, eval, test) for evaluation, by default ``10_000  # 10GB``.
    early_stopping_iter: Stateless[Optional[int]]
        The maximum allowed difference between the current iteration and the last iteration
        with the best metric before applying early stopping.
        Early stopping will be triggered if the difference ``(current_itr - best_itr)`` exceeds ``early_stopping_iter``.
        If set to ``None``, early stopping will not be applied. By default ``None``.
    eval_epoch: Stateless[float]
        The epoch interval between two evaluations, by default ``1``.
    log_epoch: Stateless[float]
        The epoch interval between two logging, by default ``1``.
    init_chkpt: Stateless[Optional[str]]
        Path to a checkpoint to initialize the model with, by default ``None``.
    warm_up_epochs: Stateless[float]
        Number of epochs marked as warm up epochs, by default ``1``.
    divergence_factor: Stateless[Optional[float]]
        If ``cur_loss > best_metric > divergence_factor``, the model is considered to have diverged, by default ``10``.
    optim_metrics: Stateless[Optional[Dict[Optim]]]
        The optimization metric to use for meta-training procedures, such as for model saving and lr scheduling.
    optim_metric_name: Stateless[Optional[str]]
        The name of the metric to be optimized.

    Examples
    --------
    There are several steps before defining a run config, let's go through them one by one:

    - Define training config:

    >>> my_optimizer_config = OptimizerConfig("sgd", {"lr": 0.5, "weight_decay": 0.5})
    >>> my_scheduler_config = SchedulerConfig("step", arguments={"step_size": 1, "gamma": 0.99})
    >>> train_config = TrainConfig(
    ...     dataset="[Dataset Name]",
    ...     batch_size=32,
    ...     epochs=10,
    ...     optimizer_config = my_optimizer_config,
    ...     scheduler_config = my_scheduler_config,
    ...     rand_weights_init = True
    ... )

    - Define model config, here we use default one with no custom hyperparameters (sometimes you would
      want to customize the model config to run HPO on your model's hyperparameters in the parallel experiments
      with ```ParallelTrainer```, which requires ```ParallelConfig``` instead of ```RunConfig```):

    >>> model_config = ModelConfig()

    - Lastly, we will create the run config, which has train config and model config as parameters:

    >>> run_config = RunConfig(
    ...     train_config=train_config,
    ...     model_config=model_config,
    ...     metrics_n_batches = 800,
    ...     experiment_dir = "/tmp/experiments",
    ...     device="cpu",
    ...     amp=False,
    ...     random_seed = 42
    ... )
    """

    # location to store experiment artifacts
    experiment_dir: Stateless[Optional[str]] = None
    experiment_id: Stateless[Optional[str]] = None
    random_seed: Optional[int] = None
    train_config: TrainConfig
    model_config: ModelConfig
    keep_n_checkpoints: Stateless[int] = 3
    tensorboard: Stateless[bool] = True
    amp: Stateless[bool] = True
    device: Stateless[str] = "cuda"
    verbose: Stateless[Literal["console", "progress", "silent"]] = "console"
    eval_subsample: Stateless[float] = 1
    metrics_n_batches: Stateless[int] = 32
    metrics_mb_limit: Stateless[int] = 10_000  # 10GB
    early_stopping_iter: Stateless[Optional[int]] = None
    eval_epoch: Stateless[float] = 1
    log_epoch: Stateless[float] = 1
    init_chkpt: Stateless[Optional[str]] = None
    warm_up_epochs: Stateless[float] = 1
    divergence_factor: Stateless[Optional[float]] = 10
    optim_metrics: Stateless[Optional[Dict[Optim]]]
    optim_metric_name: Stateless[Optional[str]]
    remote_config: Stateless[Optional[RemoteConfig]]

    @property
    def uid(self) -> str:
        train_uid = self.train_config.uid
        model_uid = self.model_config.uid
        uid = f"{train_uid}_{model_uid}"
        return uid

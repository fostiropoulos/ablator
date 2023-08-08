from ablator.config.main import ConfigBase, configclass
from ablator.config.types import (
    Optional,
    Stateless,
    Literal,
)
from ablator.modules.optimizer import OptimizerConfig
from ablator.modules.scheduler import SchedulerConfig


@configclass
class TrainConfig(ConfigBase):
    """
    Training configuration that defines the training setting, e.g., batch size, number of epochs,
    the optimizer to use, etc. This configuration is required when creating the run configuration
    (``RunConfig`` and ``ParallelConfig``), which sets up the running environment of the experiment.

    Attributes
    ----------
    dataset: str
        dataset name. maybe used in custom dataset loader functions.
    batch_size: int
        batch size.
    epochs: int
        number of epochs to train.
    optimizer_config: OptimizerConfig
        optimizer configuration. (check ``OptimizerConfig`` for more details)
    scheduler_config: Optional[SchedulerConfig]
        scheduler configuration. (check ``SchedulerConfig`` for more details)
    rand_weights_init: bool = True
        whether to initialize model weights randomly.
    
    Examples
    --------
    The following example shows all the steps towards configuring an experiment:

    - Define model config, here we use default one with no custom hyperparameters (so we're not
      running ablation study on the model architecture):

    >>> my_model_config = ModelConfig()

    - Define optimizer and scheduler config, as training config requires an optimizer
      config, and optionally a scheduler config:

    >>> my_optimizer_config = OptimizerConfig("sgd", {"lr": 0.5, "weight_decay": 0.5})
    >>> my_scheduler_config = SchedulerConfig("step", arguments={"step_size": 1, "gamma": 0.99})

    - Define training config:

    >>> my_train_config = CustomTrainConfig(
    ...     dataset="[Your Dataset]",
    ...     batch_size=32,
    ...     epochs=10,
    ...     optimizer_config = my_optimizer_config,
    ...     scheduler_config = my_scheduler_config,
    ...     rand_weights_init = True
    ... )

    - We now define the run config for prototype training, which is the last configuration step. Refer to :ref:`Configurations
      for single model experiments <run_config>` and :ref:`Configurations for parallel models experiments <parallel_config>`
      for more details on running configs.

    >>> run_config = CustomRunConfig(
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
    rand_weights_init: bool = True


# TODO decorator @modelconfig as opposed to @configclass ModelConfig
@configclass
class ModelConfig(ConfigBase):
    """
    A base class for model configuration. This is used for defining model hyperparameters,
    so when initializing a model, this config is passed to the model constructor. The attributes
    from the model config object will be used to construct the model.

    Examples
    --------
    Define custom model configuration class for your model:
    
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
    """


@configclass
class RunConfig(ConfigBase):
    """
    The base configuration that defines the setting of an experiment (experiment main directory, number of 
    checkpoints to maintain, hardware device to use, etc.). You can use this to configure the experiment
    of running a single prototype model.
    
    ``RunConfig`` encapsulates every configuration (model config, optimizer-scheduler config,
    train config) needed for a prototype experiment. The entire umbrella of configurations is then passed
    to ``ProtoTrainer`` which launches the prototype experiment.

    Attributes
    ----------
    experiment_dir: Optional[str] = None
        location to store experiment artifacts.
    random_seed: Optional[int] = None
        random seed.
    train_config: TrainConfig
        training configuration. (check ``TrainConfig`` for more details)
    model_config: ModelConfig
        model configuration. (check ``ModelConfig`` for more details)
    keep_n_checkpoints: int = 3
        number of latest checkpoints to keep.
    tensorboard: bool = True
        whether to use tensorboardLogger.
    amp: bool = True
        whether to use automatic mixed precision when running on gpu.
    device: str = "cuda" or "cpu"
        device to run on.
    verbose: Literal["console", "progress", "silent"] = "console"
        verbosity level.
    eval_subsample: float = 1
        fraction of the dataset to use for evaluation.
    metrics_n_batches: int = 32
        max number of batches stored in every tag(train, eval, test) for evaluation.
    metrics_mb_limit: int = 100
        max number of megabytes stored in every tag(train, eval, test) for evaluation.
    early_stopping_iter: Optional[int] = None
        The maximum allowed difference between the current iteration and the last iteration
        with the best metric before applying early stopping.
        Early stopping will be triggered if the difference ``(current_itr - best_itr)`` exceeds ``early_stopping_iter``.
        If set to ``None``, early stopping will not be applied.
    eval_epoch: float = 1
        The epoch interval between two evaluations.
    log_epoch: float = 1
        The epoch interval between two logging.
    init_chkpt: Optional[str] = None
        path to a checkpoint to initialize the model with.
    warm_up_epochs: float = 0
        number of epochs marked as warm up epochs.
    divergence_factor: float = 100
        if ``cur_loss > best_loss > divergence_factor``, the model is considered to have diverged.

    Examples
    --------
    There are several steps before defining a run config, let's go through them one by one: 

    - Define model config, here we use default one with no custom hyperparameters (sometimes you would
      want to define model config when running HPO on your model's hyperparameters in the parallel experiments
      with ```ParallelTrainer```, which requires ```ParallelConfig``` instead of ```RunConfig```):

    >>> model_config = ModelConfig()

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
    metrics_mb_limit: Stateless[int] = 100
    early_stopping_iter: Stateless[Optional[int]] = None
    eval_epoch: Stateless[float] = 1
    log_epoch: Stateless[float] = 1
    init_chkpt: Stateless[Optional[str]] = None
    warm_up_epochs: Stateless[float] = 1
    divergence_factor: Stateless[Optional[float]] = 100

    @property
    def uid(self) -> str:
        train_uid = self.train_config.uid
        model_uid = self.model_config.uid
        uid = f"{train_uid}_{model_uid}"
        return uid

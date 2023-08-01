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
    Training configuration that defines the training setting, e.g., batch size, number of epochs, the optimizer to be used, etc.

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
    A base class for model configuration. This is used for defining model parameters,
    so when initializing a model, this config is passed to the model constructor.

    Examples
    --------
    Define custom model configuration class for your model:
    
    >>> @configclass
    ... class CustomModelConfig(ModelConfig):
    ...     input_size :int
    ...     hidden_size :int
    ...     num_classes :int

    Define your model class, and pass the configuration to the constructor:

    >>> class FashionMNISTModel(nn.Module):
    ...     def __init__(self, config: CustomModelConfig):
    ...         super(FashionMNISTModel, self).__init__()
    ...         self.fc1 = nn.Linear(config.input_size, config.hidden_size)
    ...         self.relu1 = nn.ReLU()
    ...         self.fc3 = nn.Linear(config.hidden_size, config.num_classes)

    ...     def forward(self, x):
    ...         # code for forward pass
    ...         return x
    """


@configclass
class RunConfig(ConfigBase):
    """
    The base configuration that defines the setting of an experiment (experiment main directory, number of 
    checkpoints to maintain, hardware device to use, etc.). You can use this to configure the experiment
    of running a single prototype model.

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

# TODO fix mypy that does not recognize correctly the types i.e. Stateless
from ablator.config.main import ConfigBase, configclass
from ablator.config.types import (
    Optional,
    Stateless,
    Literal,
    Tuple,
    List,
    Enum,
    Dict,
)
from ablator.modules.optimizer import OptimizerConfig
from ablator.modules.scheduler import SchedulerConfig
from ablator.modules.storage.cloud import GcpConfig
from ablator.modules.storage.remote import RemoteConfig


@configclass
class TrainConfig(ConfigBase):
    """
    Training configuration.

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
    Model configuration.
    When initializing a model, the config is passed to the model constructor.
    """
    pass


@configclass
class RunConfig(ConfigBase):
    """
    Base configuration for running an experiment.

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
    verbose: Literal["console", "tqdm", "silent"] = "console"
        verbosity level.
    eval_subsample: float = 1
        fraction of the dataset to use for evaluation.
    metrics_n_batches: int = 32
        max number of batches stored in every tag(train, eval, test) for evaluation.
    metrics_mb_limit: int = 100
        max number of megabytes stored in every tag(train, eval, test) for evaluation.
    early_stopping_iter: Optional[int] = None
        The maximum allowed difference between the current iteration and the last iteration with the best metric before applying early stopping.
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
    verbose: Stateless[Literal["console", "tqdm", "silent"]] = "console"
    eval_subsample: Stateless[float] = 1
    metrics_n_batches: Stateless[int] = 32
    metrics_mb_limit: Stateless[int] = 100
    early_stopping_iter: Stateless[Optional[int]] = None
    eval_epoch: Stateless[float] = 1
    log_epoch: Stateless[float] = 1
    init_chkpt: Stateless[Optional[str]] = None
    warm_up_epochs: Stateless[float] = 1
    divergence_factor: Stateless[float] = 100

    @property
    def uid(self) -> str:
        train_uid = self.train_config.uid
        model_uid = self.model_config.uid
        uid = f"{train_uid}_{model_uid}"
        return uid


class SearchType(Enum):
    """
    Type of search space.
    """
    integer = "int"
    numerical = "float"


@configclass
class SearchSpace(ConfigBase):
    """
    Search space configuration.
    """
    value_range: Optional[Tuple[str, str]]
    categorical_values: Optional[List[str]]
    value_type: SearchType = SearchType.numerical

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert (
            self.value_range is None or self.categorical_values is None
        ), "Can not specify value_range and categorical_values for SearchSpace."


class SearchAlgo(Enum):
    """
    Type of search algorithm.
    """
    random = "random"
    tpe = "tpe"


class Optim(Enum):
    """
    Type of optimization direction.
    """
    min = "min"
    max = "max"


@configclass
class ParallelConfig(RunConfig):
    """
    Parallel training configuration. ``{"val_loss": "min"}``

    Attributes
    ----------
    total_trials: int
        total number of trials.
    concurrent_trials: int
        number of trials to run concurrently.
    search_space: Dict[SearchSpace]
        search space for hyperparameter search,eg. ``{"train_config.optimizer_config.arguments.lr": SearchSpace(value_range=[0, 10], value_type="int"),}``
    optim_metrics: Dict[Optim]
        metrics to optimize, eg. ``{"val_loss": "min"}``
    search_algo: SearchAlgo = SearchAlgo.tpe
        type of search algorithm.
    ignore_invalid_params: bool = False
        whether to ignore invalid parameters when sampling.
    remote_config: Optional[RemoteConfig] = None
        remote storage configuration.
    gcp_config: Optional[GcpConfig] = None
        gcp configuration.

    """
    total_trials: int
    concurrent_trials: Stateless[int]
    search_space: Dict[SearchSpace]
    optim_metrics: Stateless[Dict[Optim]]
    gpu_mb_per_experiment: Stateless[int]
    cpus_per_experiment: Stateless[float]
    search_algo: Stateless[SearchAlgo] = SearchAlgo.tpe
    ignore_invalid_params: Stateless[bool] = False
    remote_config: Stateless[Optional[RemoteConfig]] = None
    gcp_config: Stateless[Optional[GcpConfig]] = None

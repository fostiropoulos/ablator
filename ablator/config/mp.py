from ablator.config.hpo import SearchSpace
from ablator.config.main import configclass
from ablator.config.proto import RunConfig
from ablator.config.types import Dict, Enum, Optional, Stateless
from ablator.modules.storage.cloud import GcpConfig
from ablator.modules.storage.remote import RemoteConfig

CRASH_EXCEPTION_TYPES: list[type] = []


class SearchAlgo(Enum):
    """
    Type of search algorithm.
    TODO explain each type
    """

    random = "random"
    tpe = "tpe"
    grid = "grid"
    discrete = "discrete"


class Optim(Enum):
    """
    Type of optimization direction.

    can take values `min` and `max` that indicate whether the HPO
    algorithm should minimize or maximize the corresponding metric.
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
        search space for hyperparameter search,
        eg. ``{"train_config.optimizer_config.arguments.lr": SearchSpace(value_range=[0, 10], value_type="int"),}``
    optim_metrics: Dict[Optim]
        metrics to optimize, eg. ``{"val_loss": "min"}``
    search_algo: SearchAlgo = SearchAlgo.tpe
        type of search algorithm.
    ignore_invalid_params: bool = False
        whether to ignore invalid parameters when sampling or raise an error.
    sample_duplicate_params: bool = False
        whether to include duplicate parameters when sampling or ingore them.
    remote_config: Optional[RemoteConfig] = None
        remote storage configuration.
    gcp_config: Optional[GcpConfig] = None
        gcp configuration.

    """

    total_trials: Optional[int]
    concurrent_trials: Stateless[Optional[int]]
    search_space: Dict[SearchSpace]
    optim_metrics: Stateless[Optional[Dict[Optim]]]
    gpu_mb_per_experiment: Stateless[int]
    # cpus_per_experiment: Stateless[float]
    search_algo: Stateless[SearchAlgo] = SearchAlgo.tpe
    ignore_invalid_params: Stateless[bool] = False
    sample_duplicate_params: Stateless[bool] = False
    remote_config: Stateless[Optional[RemoteConfig]] = None
    gcp_config: Stateless[Optional[GcpConfig]] = None

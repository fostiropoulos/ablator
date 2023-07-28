from ablator.config.hpo import SearchSpace
from ablator.config.main import configclass
from ablator.config.proto import RunConfig
from ablator.config.rclone import GcsRcloneConfig, RemoteRcloneConfig
from ablator.config.types import Dict, Enum, Optional, Stateless
from ablator.modules.storage.remote import RemoteConfig


class SearchAlgo(Enum):
    """
    Type of search algorithm.

    Grid Sampling: Discretizes the search space into even intervals `n_bins`.
    TPE Sampling: Tree-Structured Parzen Estimator [1] is a hyper-parameter optimization algorithm.
    Random Sampling: Naively samples from the search space with a random probability.

    The behavior of each algorithm depends highly on the budget allocated for each trial. For example,
    Grid Sampling will repeat sampled configurations only after it has exhaustively evaluated the current
    configuration space.

    TPE and Random Sampling can repeat configurations at random.

    References:
    [1] Bergstra, James S., et al. “Algorithms for hyper-parameter optimization.”
    Advances in Neural Information Processing Systems. 2011.
    """

    random = "random"
    tpe = "tpe"
    grid = "grid"


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
    Parallel training configuration.

    Attributes
    ----------
    total_trials: Optional[int]
        total number of trials.
    concurrent_trials: int
        number of trials to run concurrently.
    search_space: Dict[SearchSpace]
        search space for hyperparameter search,
        eg. ``{"train_config.optimizer_config.arguments.lr": SearchSpace(value_range=[0, 10], value_type="int"),}``
    optim_metrics: Optional[Dict[Optim]]
        metrics to optimize, eg. ``{"val_loss": "min"}``
    gpu_mb_per_experiment: int
        CUDA memory requirement per experimental trial in MB. e.g. a value of 100 is equivalent to 100MB
    search_algo: SearchAlgo = SearchAlgo.tpe
        type of search algorithm.
    ignore_invalid_params: bool = False
        whether to ignore invalid parameters when sampling or raise an error.
    remote_config: Optional[RemoteConfig] = None
        remote storage configuration.

    """

    total_trials: Optional[int]
    concurrent_trials: Stateless[Optional[int]]
    search_space: Dict[SearchSpace]
    optim_metrics: Stateless[Optional[Dict[Optim]]]
    gpu_mb_per_experiment: Stateless[int]
    search_algo: Stateless[SearchAlgo] = SearchAlgo.tpe
    ignore_invalid_params: Stateless[bool] = False
    remote_config: Stateless[Optional[RemoteConfig]] = None
    gcs_rclone_config: Stateless[Optional[GcsRcloneConfig]] = None
    remote_rclone_config: Stateless[Optional[RemoteRcloneConfig]] = None

from ablator.config.hpo import SearchSpace
from ablator.config.main import configclass
from ablator.config.proto import RunConfig
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
    Parallel training configuration, extending from ``RunConfig``, defines the settings of a parallel experiment
    (number of trials to run for, number of concurrent trials, search space for hyperparameter search, etc.).
    
    ``ParallelConfig`` encapsulates every configuration (model config, optimizer-scheduler config, train config,
    and the search space) needed to run a parallel experiment. The entire umbrella of configuration is then passed
    to ``ParallelTrainer`` that launches the experiment.

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

    Examples
    --------
    There are several steps before defining a parallel run config, let's go through them one by one: 

    - Define model config, we want to run HPO on activation functions and model hidden size:

    >>> @configclass
    >>> class CustomModelConfig(ModelConfig):
    >>>     hidden_size: int
    >>>     activation: str
    >>> model_config = CustomModelConfig(hidden_size=100, activation="relu")

    - Define training config:

    >>> my_optim_config = OptimizerConfig("sgd", {"lr": 0.5, "weight_decay": 0.5})
    >>> my_scheduler_config = SchedulerConfig("step", arguments={"step_size": 1, "gamma": 0.99})
    >>> train_config = TrainConfig(
    ...     dataset="[Dataset Name]",
    ...     batch_size=32,
    ...     epochs=10,
    ...     optimizer_config = my_optimizer_config,
    ...     scheduler_config = my_scheduler_config,
    ...     rand_weights_init = True
    ... )

    - Define search space:

    >>> search_space = {
    ...     "train_config.optimizer_config.arguments.lr": SearchSpace(value_range = [0.001, 0.01], value_type = 'float'),
    ...     "model_config.hidden_size": SearchSpace(value_range = [32, 64], value_type = 'int'),
    ...     "model_config.activation": SearchSpace(categorical_values = ["relu", "elu", "leakyRelu"]),
    ... }

    - Lastly, we will define the run config from the previous config components (remember to redefine
      the parallel config to update the model config type to be ``CustomModelConfig``):

    >>> @configclass
    >>> class CustomParallelConfig(ParallelConfig):
    ...    model_config: CustomModelConfig
    >>> parallel_config = CustomParallelConfig(
    ...     train_config=train_config,
    ...     model_config=model_config,
    ...     metrics_n_batches = 800,
    ...     experiment_dir = "/tmp/experiments/",
    ...     device="cuda",
    ...     amp=True,
    ...     random_seed = 42,
    ...     total_trials = 20,
    ...     concurrent_trials = 20,
    ...     search_space = search_space,
    ...     optim_metrics = {"val_loss": "min"},
    ...     gpu_mb_per_experiment = 1024,
    ...     cpus_per_experiment = 1,
    ... )
    """

    total_trials: Optional[int]
    concurrent_trials: Stateless[Optional[int]]
    search_space: Dict[SearchSpace]
    optim_metrics: Stateless[Optional[Dict[Optim]]]
    gpu_mb_per_experiment: Stateless[int]
    search_algo: Stateless[SearchAlgo] = SearchAlgo.tpe
    ignore_invalid_params: Stateless[bool] = False
    remote_config: Stateless[Optional[RemoteConfig]] = None

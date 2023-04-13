# TODO fix mypy that does not recognize correctly the types i.e. Stateless
# type: ignore
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
    dataset: str
    batch_size: int
    epochs: int
    optimizer_config: OptimizerConfig
    scheduler_config: Optional[SchedulerConfig]
    rand_weights_init: bool = True


@configclass
class ModelConfig(ConfigBase):
    pass


@configclass
class RunConfig(ConfigBase):
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
    integer = "int"
    numerical = "float"


@configclass
class SearchSpace(ConfigBase):
    value_range: Optional[Tuple[str, str]]
    categorical_values: Optional[List[str]]
    value_type: SearchType = "float"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert (
            self.value_range is None or self.categorical_values is None
        ), "Can not specify value_range and categorical_values for SearchSpace."


class SearchAlgo(Enum):
    random = "random"
    tpe = "tpe"


class Optim(Enum):
    min = "min"
    max = "max"


@configclass
class ParallelConfig(RunConfig):
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

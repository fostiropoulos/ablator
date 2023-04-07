from typing import Annotated, Any, Dict, List, Optional, Tuple


from trainer.config.main import ConfigBase, Derived, Enum, Stateless, configclass
from trainer.modules.logging import LoggerConfig
from trainer.modules.storage.cloud import RsyncConfig, GcpConfig
from trainer.modules.optimizers import OptimizerConfig
from trainer.modules.schedulers import SchedulerConfig


@configclass
class TrainConfigBase(ConfigBase):
    dataset: str
    batch_size: int
    epochs: int
    # Stateful variables
    scheduler_config: Optional[SchedulerConfig] = None
    optimizer_config: Optional[OptimizerConfig] = None
    random_seed: Optional[int] = None
    amp: bool = True
    init_chkpt: Annotated[Optional[str], Stateless] = None
    debug: Annotated[bool, Stateless] = False
    device: Annotated[Optional[str], Stateless] = None
    eval_epoch: Annotated[float, Stateless] = 1
    eval_subsample: Annotated[float, Stateless] = 1
    log_epoch: Annotated[float, Stateless] = 1
    early_stopping_iter: Annotated[Optional[int], Stateless] = None
    resume: Annotated[bool, Stateless] = False
    verbose: Annotated[bool, Stateless] = True
    tqdm: Annotated[bool, Stateless] = True
    # if loss diverges by at least this much, we stop
    stop_div_factor: Annotated[float, Stateless] = 10
    # only check divergence after this number of steps
    stop_div_step_frac: Annotated[float, Stateless] = 0.3
    keep_n_checkpoints: Annotated[int, Stateless] = 10
    metrics_n_batches: Annotated[int, Stateless] = 30  # metrics run on 30 batches
    metrics_byte_limit: Annotated[int, Stateless] = int(1e8)  # 100MB


@configclass
class ModelConfigBase(ConfigBase):
    pass


class TrainMode(Enum):
    # torch distributed
    dist_data_parallel = "dist_data_parallel"
    data_parallel = "data_parallel"
    vanilla = "vanilla"


class ExperimentType(Enum):
    random = "random"
    # TODO sequential
    grid = "grid"
    tpe = "tpe"


class BackendType(Enum):
    nccl = "nccl"
    gloo = "gloo"


class Optim(Enum):
    min = "min"
    max = "max"


@configclass
class RunConfig(ConfigBase):
    # location to store experiment artifacts
    experiment_dir: Annotated[str, Stateless]
    # location to store model artifacts
    train_config: TrainConfigBase
    model_config: ModelConfigBase
    model_dir: Annotated[Optional[str], Derived] = None
    train_mode: Annotated[TrainMode, Stateless] = TrainMode.vanilla
    logger_configs: Annotated[Optional[List[LoggerConfig]], Stateless] = None
    rsync_config: Annotated[Optional[RsyncConfig], Stateless] = None
    gcp_config: Annotated[Optional[GcpConfig], Stateless] = None
    optim_directions: Optional[List[Tuple[str, Optim]]] = None

    @property
    def uid(self) -> str:

        train_uid = self.train_config.uid
        model_uid = self.model_config.uid
        uid = f"{train_uid}_{model_uid}"
        return uid


@configclass
class ParallelConfig(RunConfig):
    """
    MultiProcess Training
    """

    total_trials: Annotated[int, Stateless]
    concurrent_trials: Annotated[int, Stateless]
    tune: Dict[str, List[Any]]
    num_experiment_per_gpu: Annotated[Optional[int], Stateless] = None
    num_experiment_per_cpu: Annotated[Optional[int], Stateless] = None
    experiment_type: Annotated[ExperimentType, Stateless] = ExperimentType.tpe
    derived_mem_multiplier: Annotated[float, Stateless] = 1
    ignore_errored_trials: Annotated[bool, Stateless] = False

    def __init__(self, *args, add_attributes=False, **kwargs):
        super().__init__(*args, add_attributes=add_attributes, **kwargs)
        assert self.train_mode == TrainMode.vanilla, "Invalid configuration " # TODO

@configclass
class DDPConfig(RunConfig):
    rank: Annotated[Optional[int], Stateless, Derived] = None
    device_ids: Annotated[Optional[List[int]], Stateless] = None
    world_size: Annotated[Optional[int], Stateless] = None
    backend: Annotated[Optional[BackendType], Stateless] = None
    ip: Annotated[str, Stateless] = "localhost"
    port: Annotated[str, Stateless] = "12355"


@configclass
class DPConfig(RunConfig):
    device_ids: Annotated[Optional[List[int]], Stateless] = None

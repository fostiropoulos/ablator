# TODO fix mypy that does not recognize correctly the types i.e. Stateless
# type: ignore
from trainer.config.main import ConfigBase, configclass
from trainer.config.types import Optional, Stateless, Literal
from trainer.modules.optimizer import OptimizerConfig
from trainer.modules.scheduler import SchedulerConfig


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

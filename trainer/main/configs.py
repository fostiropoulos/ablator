from trainer.config.main import ConfigBase, configclass
from trainer.config.types import Derived, Optional, Stateless


@configclass
class TrainConfig(ConfigBase):
    dataset: str
    batch_size: int
    epochs: int
    random_seed: Optional[int] = None


@configclass
class ModelConfig(ConfigBase):
    pass


@configclass
class RunConfig(ConfigBase):
    # location to store experiment artifacts
    experiment_dir: Stateless[str]
    train_config: TrainConfig
    model_config: ModelConfig
    keep_n_checkpoints: Stateless[int] = 3

    @property
    def uid(self) -> str:
        train_uid = self.train_config.uid
        model_uid = self.model_config.uid
        uid = f"{train_uid}_{model_uid}"
        return uid

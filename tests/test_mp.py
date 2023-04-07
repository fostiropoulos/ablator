import random
from pathlib import Path

import ray
from resnet import CONFIG, ModelConfig, MyModel, MyModelWrapper

import trainer
from trainer.config.main import configclass
from trainer.config.run import ParallelConfig
from trainer.mp import MPTrainer
import shutil


@configclass
class MyParallelConfig(ParallelConfig):
    model_config: ModelConfig


def my_train(tmp_path: Path):

    model = MyModelWrapper(
        model_class=MyModel,
    )
    tune_args = dict(
        experiment_dir=tmp_path.as_posix(),
        num_experiment_per_gpu=4,
        ignore_errored_trials=True,
        experiment_type="tpe",
        tune={
            "train_config.optimizer_config.arguments.lr": [1e-4, 1e-3],
            "train_config.optimizer_config.arguments.weight_decay": [1.0e-4, 1.0e-3],
            "train_config.optimizer_config.name": ["adam", "adamw"],
        },
        total_trials=4,
        concurrent_trials=2,
        optim_directions=[("val_loss", "min"), ("val_accuracy_score", "max")],
    )
    CONFIG.update(tune_args)
    run_config = MyParallelConfig(**CONFIG)  # type: ignore
    run_config.train_config.tqdm = False

    # NOTE working_dir is required for accessing resnet.py
    ray.init(
        # address="auto",
        runtime_env={
            "working_dir": Path(__file__).parent.resolve(),
            "py_modules": [trainer],
        },
    )
    mp_trainer = MPTrainer(
        model=model,
        run_config=run_config,
        description="test",
    )
    mp_trainer.launch()
    mp_trainer.evaluate()


if __name__ == "__main__":
    save_dir = Path(f"/tmp/test_dir")
    shutil.rmtree(save_dir, ignore_errors=True)
    my_train(save_dir)

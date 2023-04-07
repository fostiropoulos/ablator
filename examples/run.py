import argparse
from pathlib import Path
import time
from typing import Dict, Literal

import torch
from omegaconf import OmegaConf
from trainer.base import BaseTrainer
from trainer.config.main import configclass
from trainer.config.run import DDPConfig, DPConfig, ParallelConfig
from trainer.ddp import DDPTrainer
from resnet import ModelConfig, MyModel, MyModelWrapper, ResRunConfig

FIXED_EXPERIMENT_DIR = False
if FIXED_EXPERIMENT_DIR:
    EXPERIMENT_DIR_NAME = "resnet_experiments"
else:
    EXPERIMENT_DIR_NAME = str(time.time())


@configclass
class MyDPConfig(DPConfig):
    experiment_dir: str = f"/tmp/dp_{EXPERIMENT_DIR_NAME}"
    model_config: ModelConfig


@configclass
class MyDDPConfig(DDPConfig):
    experiment_dir: str = f"/tmp/ddp_{EXPERIMENT_DIR_NAME}"
    model_config: ModelConfig


@configclass
class MyParallelConfig(ParallelConfig):
    experiment_dir: str = f"/tmp/mp_{EXPERIMENT_DIR_NAME}"
    model_config: ModelConfig


def make_trainer(model, **kwargs):
    kwargs["experiment_dir"] = f"/tmp/base_{EXPERIMENT_DIR_NAME}"
    run_config = ResRunConfig(**kwargs)  # type: ignore
    return BaseTrainer(
        model=model,
        run_config=run_config,
        description="ResNet Experiments",
    )


def make_trainer_dp(model, **kwargs):

    run_config = MyDPConfig(**kwargs)  # type: ignore
    run_config.device_ids = list(range(torch.cuda.device_count()))
    return BaseTrainer(
        model=model,
        run_config=run_config,
        description="Data-Parallel ResNet Experiments",
    )


def make_trainer_ddp(model, **kwargs):
    kwargs["backend"] = "gloo"
    run_config = MyDDPConfig(**kwargs)  # type: ignore
    return DDPTrainer(
        model=model,
        run_config=run_config,
        description="DDP ResNet Experiments",
    )


def make_trainer_mp(model, **kwargs):

    try:
        from trainer.mp import MPTrainer
        import ray
    except ImportError:
        raise ImportError("You need to install model-trainer with option [mp]")

    tune_args = dict(
        num_experiment_per_gpu=4,
        ignore_errored_trials=True,
        experiment_type="tpe",
        # NOTE: Equivalent yaml
        # tune:
        #   train_config.optimizer_config.lr: [1.0e-4, 1.0e-3]
        #   train_config.optimizer_config.name: [adam, adamw]
        tune={
            "train_config.optimizer_config.arguments.lr": [1e-4, 1e-3],
            "train_config.optimizer_config.arguments.weight_decay": [1.0e-4, 1.0e-3],
            "train_config.optimizer_config.name": ["adam", "adamw"],
            # NOTE: This nested config only works for control trials because some settings are incompatible.
            # TODO: how to resolve this?
            # "train_config.optimizer_config": [
            #     {"name": ["adam"], "arguments.lr": [1.0e-4, 1.0e-3]},
            #     {
            #         "name": ["adamw"],
            #         "arguments.lr": [1.0e-4],
            #         "arguments.eps": [1.0e-4, 1.0e-3],
            #     },
            # ],
        },
        total_trials=4,
        concurrent_trials=2,
        optim_directions=[("val_loss", "min"), ("val_accuracy_score", "max")],
    )
    kwargs.update(tune_args)
    run_config = MyParallelConfig(**kwargs)  # type: ignore
    run_config.train_config.tqdm = False

    # NOTE working_dir is required for accessing resnet.py
    ray.init(
        address="auto", runtime_env={"working_dir": Path(__file__).parent.resolve()}
    )
    return MPTrainer(
        model=model,
        run_config=run_config,
        description="Parallel training ResNet Experiments",
    )


def my_train(config: str, train_mode: Literal["dp", "ddp", "pp", None]):

    kwargs: Dict = OmegaConf.to_object(OmegaConf.load(config))  # type: ignore

    del kwargs["experiment_dir"]

    model = MyModelWrapper(
        model_class=MyModel,
    )

    if train_mode == "ddp":
        trainer = make_trainer_ddp(model, **kwargs)
    elif train_mode == "dp":
        trainer = make_trainer_dp(model, **kwargs)
    elif train_mode is None:
        trainer = make_trainer(model, **kwargs)
    elif train_mode == "mp":
        trainer = make_trainer_mp(model, **kwargs)
    else:
        raise NotImplementedError

    trainer.launch()
    trainer.evaluate()



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, required=True)
    args.add_argument("--train_mode", choices=["dp", "ddp", "mp", None], default=None)
    kwargs = vars(args.parse_args())
    config = my_train(**kwargs)

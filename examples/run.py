import argparse
from pathlib import Path
import shutil


from resnet import MyModel, MyModelWrapper, ResConfig, ResRunConfig

from ablator import ParallelConfig, ParallelTrainer, ProtoTrainer, configclass

WORKING_DIRECTORY = Path(__file__).parent


@configclass
class MyParallelConfig(ParallelConfig):
    model_config: ResConfig


# mp_train prepares and launches parallel training
def mp_train(mp_config):
    wrapper = MyModelWrapper(
        model_class=MyModel,
    )
    run_config = MyParallelConfig.load(mp_config)
    shutil.rmtree(run_config.experiment_dir)
    ablator = ParallelTrainer(
        wrapper=wrapper,
        run_config=run_config,
    )

    # NOTE to run on a cluster you will need to start ray with `ray start --head` and pass ray_head_address="auto"
    ablator.launch(
        working_directory=WORKING_DIRECTORY,
    )
    ablator.evaluate()


# base_train prepares and launches single machine training
def base_train(config):
    wrapper = MyModelWrapper(
        model_class=MyModel,
    )
    run_config = ResRunConfig.load(config)
    shutil.rmtree(run_config.experiment_dir)
    ablator = ProtoTrainer(
        wrapper=wrapper,
        run_config=run_config,
    )
    ablator.launch()


# Depending on the 'mp' flag, the function 'run' either launches single machine or parallel training
def run(config: str, mp: bool):
    if mp:
        mp_train(config)
    else:
        base_train(config)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, required=True)
    args.add_argument("--mp", action="store_true")
    kwargs = vars(args.parse_args())
    config = run(**kwargs)

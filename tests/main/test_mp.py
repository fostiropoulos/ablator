from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import pandas as pd

import ray
import torch
from torch import nn

from ablator import (
    ModelConfig,
    ModelWrapper,
    OptimizerConfig,
    RunConfig,
    TrainConfig, Enum,
)
from ablator.analysis.results import Results
from ablator.analysis.main import Analysis
from ablator.config.main import configclass
from ablator.main.configs import ParallelConfig, SearchSpace, Optim
from ablator.main.mp import ParallelTrainer
from ablator import Derived


class CustomModelConfig(ModelConfig):
    lr: Derived[int]


@configclass
class MyParallelConfig(ParallelConfig):
    model_config: CustomModelConfig


optimizer_config = OptimizerConfig(name="sgd", arguments={"lr": 0.1})
train_config = TrainConfig(
    dataset="test",
    batch_size=128,
    epochs=2,
    optimizer_config=optimizer_config,
    scheduler_config=None,
)
search_space = {
    "train_config.optimizer_config.arguments.lr": SearchSpace(
        value_range=[0, 10], value_type="int"
    ),
}


config = MyParallelConfig(
    train_config=train_config,
    model_config=CustomModelConfig(),
    verbose="silent",
    device="cpu",
    amp=False,
    search_space=search_space,
    optim_metrics={"val_loss": "min"},
    total_trials=10,
    concurrent_trials=10,
    gpu_mb_per_experiment=100,
    cpus_per_experiment=0.5,
)


class TestWrapper(ModelWrapper):
    def make_dataloader_train(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl

    def make_dataloader_val(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl

    def config_parser(self, run_config: MyParallelConfig):
        run_config.model_config.lr = (
            run_config.train_config.optimizer_config.arguments.lr
        )
        return super().config_parser(run_config)


class MyCustomModel(nn.Module):
    def __init__(self, config: CustomModelConfig) -> None:
        super().__init__()
        self.lr = config.lr
        self.param = nn.Parameter(torch.ones(100))
        self.itr = 0

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        self.itr += 1
        if self.itr > 10 and self.lr > 5:
            raise Exception("large lr.")
        return {"preds": x}, x.sum().abs()


def assert_error_msg(fn, error_msg):
    try:
        fn()
        assert False
    except Exception as excp:
        if not error_msg == str(excp):
            raise excp


def capture_output(fn):
    out = io.StringIO()

    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        fn()

    return out.getvalue(), err.getvalue()


def test_mp(tmp_path: Path):
    wrapper = TestWrapper(MyCustomModel)

    with tempfile.TemporaryDirectory() as fp:
        config.experiment_dir = fp
        assert_error_msg(
            lambda: ParallelTrainer(wrapper=wrapper, run_config=config),
            "Device must be set to 'cuda' if `gpu_mb_per_experiment` > 0",
        )
        config.device = "cuda"
        config.gpu_mb_per_experiment = 1e12
        config.cpus_per_experiment = 1e12
        out, err = capture_output(
            lambda: ParallelTrainer(wrapper=wrapper, run_config=config)
        )
        assert (
            "Expected GPU memory utilization" in out
            and "Expected CPU core util. exceed system capacity" in out
        )

    config.experiment_dir = tmp_path
    config.device = "cuda"
    config.gpu_mb_per_experiment = 0.001
    config.cpus_per_experiment = 0.001
    ablator = ParallelTrainer(wrapper=wrapper, run_config=config)
    ablator.gpu = 1 / config.concurrent_trials
    ablator.launch(Path(__file__).parent.as_posix(), ray_head_address=None)
    res = Results(MyParallelConfig, ablator.experiment_dir)
    assert res.data.shape[0] // 2 == len(ablator.experiment_state.complete_trials)

if __name__ == "__main__":
    import shutil

    tmp_path = Path("/tmp/experiment_dir")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir()
    test_mp(tmp_path)

    pass

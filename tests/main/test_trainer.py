from pathlib import Path
import torch
from torch import nn
from ablator import (
    ModelConfig,
    OptimizerConfig,
    RunConfig,
    TrainConfig,
    ProtoTrainer,
    ModelWrapper,
    SchedulerConfig,
)
import pytest
from ablator.modules.metrics.main import Metrics
import random


@pytest.fixture
def optimizer_config():
    return OptimizerConfig(name="sgd", arguments={"lr": 0.1})


@pytest.fixture
def train_config(optimizer_config):
    return TrainConfig(
        dataset="test",
        batch_size=128,
        epochs=2,
        optimizer_config=optimizer_config,
        scheduler_config=None,
    )


@pytest.fixture
def config(train_config):
    return RunConfig(
        train_config=train_config,
        model_config=ModelConfig(),
        verbose="silent",
        device="cpu",
        amp=False,
    )


class MyCustomModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        return {"preds": x}, x.sum().abs()


class TestWrapper(ModelWrapper):
    def make_dataloader_train(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl

    def make_dataloader_val(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl


class MyCustomModel2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        return {"preds": x}, x.sum().abs()

class MyCustomModel3(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        return {"preds": x}, None


class TestWrapper2(ModelWrapper):
    def make_dataloader_train(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl

def test_proto(tmp_path: Path, assert_error_msg, config):
    wrapper = TestWrapper(MyCustomModel)
    assert_error_msg(
        lambda: ProtoTrainer(wrapper=wrapper, run_config=config),
        "Must specify an experiment directory.",
    )
    config.experiment_dir = tmp_path.joinpath(f"{random.random()}")
    ablator = ProtoTrainer(wrapper=wrapper, run_config=config)
    metrics = ablator.launch()
    val_metrics = ablator.evaluate()
    assert abs((metrics["val_loss"] - val_metrics["val"]["loss"])) < 0.01


def test_proto_with_scheduler(tmp_path: Path, config):
    wrapper = TestWrapper(MyCustomModel2)
    config.experiment_dir = tmp_path.joinpath(f"{random.random()}")
    ablator = ProtoTrainer(wrapper=wrapper, run_config=config)
    metrics = ablator.launch()
    val_metrics = ablator.evaluate()
    assert abs((metrics["val_loss"] - val_metrics["val"]["loss"])) < 0.01




def test_val_loss_is_none(tmp_path: Path, config, assert_error_msg):
    wrapper = TestWrapper2(MyCustomModel3)
    config.experiment_dir = tmp_path.joinpath(f"{random.random()}")
    config.train_config.scheduler_config = SchedulerConfig("step", arguments={"step_when": "val"})

    ablator = ProtoTrainer(wrapper=wrapper, run_config=config)
    assert_error_msg(
        lambda: ablator.launch(),
        "A validation dataset is required with StepLR scheduler",
    )


if __name__ == "__main__":
    from tests.conftest import _assert_error_msg

    optimizer_config = OptimizerConfig(name="sgd", arguments={"lr": 0.1})
    train_config = TrainConfig(
        dataset="test",
        batch_size=128,
        epochs=2,
        optimizer_config=optimizer_config,
        scheduler_config=None,
    )
    config = RunConfig(
        train_config=train_config,
        model_config=ModelConfig(),
        verbose="silent",
        device="cpu",
        amp=False,
    )

    test_proto(Path("/tmp/"), _assert_error_msg, config)
    test_proto_with_scheduler(Path("/tmp/"), config=config)
    test_val_loss_is_none(Path("/tmp/"), config, _assert_error_msg)

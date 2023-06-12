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

from ablator.modules.metrics.main import TrainMetrics
import random

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


class MyCustomModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100))

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


def assert_error_msg(fn, error_msg):
    try:
        fn()
        assert False
    except Exception as excp:
        if not error_msg == str(excp):
            raise excp


def test_proto(tmp_path: Path):
    wrapper = TestWrapper(MyCustomModel)
    assert_error_msg(
        lambda: ProtoTrainer(wrapper=wrapper, run_config=config),
        "Must specify an experiment directory.",
    )
    config.experiment_dir = tmp_path.joinpath(f"{random.random()}")
    ablator = ProtoTrainer(wrapper=wrapper, run_config=config)
    metrics = ablator.launch()
    val_metrics = ablator.evaluate()
    assert (
        abs((metrics.to_dict()["val_loss"] - val_metrics["val"].to_dict()["val_loss"]))
        < 0.01
    )


class MyCustomModel2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100))

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        return {"preds": x}, x.sum().abs()

optimizer_config = OptimizerConfig(name="sgd", arguments={"lr": 0.1})
train_config = TrainConfig(
    dataset="test",
    batch_size=128,
    epochs=2,
    optimizer_config=optimizer_config,
    scheduler_config=SchedulerConfig("step", arguments={"step_when": "val"}),
)

config = RunConfig(
    train_config=train_config,
    model_config=ModelConfig(),
    verbose="silent",
    device="cpu",
    amp=False,
)

def test_proto_with_scheduler(tmp_path: Path):
    wrapper = TestWrapper(MyCustomModel2)
    config.experiment_dir = tmp_path.joinpath(f"{random.random()}")
    ablator = ProtoTrainer(wrapper=wrapper, run_config=config)
    metrics = ablator.launch()
    val_metrics = ablator.evaluate()
    assert (
        abs((metrics.to_dict()["val_loss"] - val_metrics["val"].to_dict()["val_loss"]))
        < 0.01
    )


class MyCustomModel3(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100))

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        return {"preds": x}, None


class TestWrapper2(ModelWrapper):
    def train(
        self,
        run_config: RunConfig,
        smoke_test: bool = False,
        debug: bool = False,
        resume: bool = False,
    ) -> TrainMetrics:
        self._init_state(
            run_config=run_config, smoke_test=smoke_test, debug=debug, resume=resume
        )

        self.metrics = TrainMetrics(
            batch_limit=run_config.metrics_n_batches,
            memory_limit=int(run_config.metrics_mb_limit * 1e6),
            moving_average_limit=self.epoch_len,
            evaluation_functions=self.evaluation_functions(),
            tags=["train"] + (["val"] if self.val_dataloader is not None else []),
            static_aux_metrics=self.train_stats,
            moving_aux_metrics=getattr(self, "aux_metric_names", []),
        )

        try:
            return self.train_loop(smoke_test)
        except KeyboardInterrupt:
            self._checkpoint()

        return self.metrics

    def make_dataloader_train(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl

    def make_dataloader_val(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl


def test_val_loss_is_none(tmp_path: Path):
    wrapper = TestWrapper2(MyCustomModel3)
    config.experiment_dir = tmp_path.joinpath(f"{random.random()}")
    ablator = ProtoTrainer(wrapper=wrapper, run_config=config)
    assert_error_msg(
        lambda: ablator.launch(),
        "A validation dataset is rquired with StepLR scheduler",
    )


if __name__ == "__main__":
    test_proto(Path("/tmp/"))
    test_proto_with_scheduler(Path("/tmp/"))
    test_val_loss_is_none(Path("/tmp/"))

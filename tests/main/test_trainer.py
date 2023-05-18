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
)
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

def test_parse_metrics():
    optim_dir1 = ["m1","m2","m4"]
    metrics1 = {"m1":1,"m2":2,"m3":3,"m4":4,"m5":5}
    result_correct = {"m1":1,"m2":2,"m4":4}
    result1 = parse_metrics(optim_dir1,metrics1)

    assert result_correct.equals(result1)

    result2 = parse_metrics(optim_dir1,None)
    assert result2 == None


if __name__ == "__main__":
    test_proto(Path("/tmp/"))

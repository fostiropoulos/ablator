from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from ablator import (
    ModelConfig,
    ModelWrapper,
    OptimizerConfig,
    PlotAnalysis,
    RunConfig,
    TrainConfig,
)
from ablator.analysis.results import Results
from ablator.config.mp import ParallelConfig, SearchSpace
from ablator.main.mp import ParallelTrainer

WORKING_DIR = Path(__file__).parent.as_posix()


N_BATCHES = 100


class MockModelConfig(ModelConfig):
    param: int = 0


class MockConfig(ParallelConfig):
    model_config: MockModelConfig = MockModelConfig()
    train_config: TrainConfig


class MyCustomModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        return {"preds": x}, x.sum().abs()


class TestWrapper(ModelWrapper):
    def make_dataloader_train(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(N_BATCHES)]
        return dl

    def make_dataloader_val(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(N_BATCHES)]
        return dl


def get_best(x: pd.DataFrame, task_type: str):
    if task_type == "regression":
        return x.sort_values("val_rmse", na_position="last").iloc[0]
    else:
        return x.sort_values("val_acc", na_position="first").iloc[-1]


@pytest.fixture()
def results(tmp_path: Path):
    return _results(tmp_path)


def _results(tmp_path: Path) -> Results:
    search_space = {
        "train_config.optimizer_config.arguments.lr": SearchSpace(
            value_range=[0, 19],
            value_type="float",
            n_bins=10,
        ),
        "model_config.param": SearchSpace(
            categorical_values=list(range(10)),
        ),
    }
    optimizer_config = OptimizerConfig(name="sgd", arguments={"lr": 0.1})
    train_config = TrainConfig(
        dataset="test",
        batch_size=128,
        epochs=2,
        optimizer_config=optimizer_config,
        scheduler_config=None,
    )
    model_config = MockModelConfig()

    config = MockConfig(
        experiment_dir=tmp_path.joinpath("test_exp"),
        train_config=train_config,
        model_config=model_config,
        verbose="silent",
        device="cpu",
        amp=False,
        search_space=search_space,
        optim_metrics=None,
        total_trials=10,
        search_algo="grid",
        concurrent_trials=10,
        gpu_mb_per_experiment=100,
    )
    wrapper = TestWrapper(MyCustomModel)
    ablator = ParallelTrainer(wrapper=wrapper, run_config=config)
    ablator.launch(WORKING_DIR)

    res = Results(config, ablator.experiment_dir)
    return res


def test_analysis(tmp_path: Path, results: Results):
    PlotAnalysis(results, optim_metrics={"val_loss": "min"})
    categorical_name_remap = {
        "model_config.param": "Some Parameter",
    }
    numerical_name_remap = {
        "train_config.optimizer_config.arguments.lr": "Learning Rate",
    }
    analysis = PlotAnalysis(
        results,
        save_dir=tmp_path.as_posix(),
        cache=True,
        optim_metrics={"val_loss": "min"},
    )
    attribute_name_remap = {**categorical_name_remap, **numerical_name_remap}
    analysis.make_figures(
        metric_name_remap={
            "val_loss": "Val. Loss",
        },
        attribute_name_remap=attribute_name_remap,
    )
    assert all(
        tmp_path.joinpath("violinplot", "val_loss", f"{file_name}.png").exists()
        for file_name in categorical_name_remap
    )

    assert all(
        tmp_path.joinpath("linearplot", "val_loss", f"{file_name}.png").exists()
        for file_name in numerical_name_remap
    )
    pass


if __name__ == "__main__":
    import shutil

    tmp_path = Path("/tmp/save_dir")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(exist_ok=True)
    res = _results(tmp_path)
    test_analysis(tmp_path, res)

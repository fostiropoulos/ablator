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


def get_best(x: pd.DataFrame, task_type: str):
    if task_type == "regression":
        return x.sort_values("val_rmse", na_position="last").iloc[0]
    else:
        return x.sort_values("val_acc", na_position="first").iloc[-1]


@pytest.fixture()
def results(tmp_path: Path, wrapper, make_config, working_dir):
    return _results(tmp_path, wrapper, make_config, working_dir)


def _results(tmp_path: Path, wrapper, make_config, working_dir) -> Results:
    search_space = {
        "train_config.optimizer_config.arguments.lr": SearchSpace(
            value_range=[0, 19],
            value_type="float",
            n_bins=10,
        ),
        "model_config.mock_param": SearchSpace(
            categorical_values=list(range(10)),
        ),
    }
    config = make_config(tmp_path.joinpath("test_exp"), search_space=search_space)
    ablator = ParallelTrainer(wrapper=wrapper, run_config=config)
    ablator.launch(working_dir)
    res = Results(config, ablator.experiment_dir)
    return res


def test_analysis(tmp_path: Path, results: Results):
    PlotAnalysis(results, optim_metrics={"val_loss": "min"})
    categorical_name_remap = {
        "model_config.mock_param": "Some Parameter",
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
    from tests.ray_models.model import (
        WORKING_DIR,
        TestWrapper,
        MyCustomModel,
        _make_config,
    )

    tmp_path = Path("/tmp/save_dir")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(exist_ok=True)
    res = _results(tmp_path, TestWrapper(MyCustomModel), _make_config, WORKING_DIR)
    test_analysis(tmp_path, res)

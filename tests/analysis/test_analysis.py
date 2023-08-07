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
    Optim
)
from ablator.analysis.results import Results
from ablator.analysis.plot.utils import parse_name_remap
from ablator.analysis.main import Analysis


def get_best(x: pd.DataFrame, task_type: str):
    if task_type == "regression":
        return x.sort_values("val_rmse", na_position="last").iloc[0]
    else:
        return x.sort_values("val_acc", na_position="first").iloc[-1]


@pytest.mark.order(1)
def test_analysis(tmp_path: Path, ablator_results):
    results: Results = ablator_results
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

def test_name_remap(tmp_path: Path):
    name_map = {'a': 'A', 'b': 'B'}
    defaults = None
    assert parse_name_remap(defaults, name_map) == name_map

    name_map = {'a': 'A', 'b': 'B'}
    defaults = ['a', 'b', 'c', 'd']
    expected_output = {'a': 'A', 'b': 'B', 'c': 'c', 'd': 'd'}
    assert parse_name_remap(defaults, name_map) == expected_output

    name_map = None
    defaults = ['a', 'b', 'c']
    assert parse_name_remap(defaults, name_map) == {'a': 'a', 'b': 'b', 'c': 'c'}

    try:
        parse_name_remap(None, None)
    except NotImplementedError as err:
        if str(err) != "`defaults` or `name_map` argument required.":
            raise err

def get_best(x: pd.DataFrame, task_type: str):
    if task_type == "regression":
        return x.sort_values("val_rmse", na_position="last").iloc[0]
    else:
        return x.sort_values("val_acc", na_position="first").iloc[-1]
    
def test_analysis_func(tmp_path: Path):
    results_csv = Path(__file__).parent.parent.joinpath("assets", "results.csv")
    df = pd.read_csv(results_csv)
    df = (
        df.groupby("path")
        .apply(lambda x: get_best(x, "classification"))
        .reset_index(drop=True)
    )
    categorical_name_remap = {
        "model_config.activation": "Activation",
        "model_config.initialization": "Weight Init.",
        "train_config.optimizer_config.name": "Optimizer",
        "model_config.mask_type": "Mask Type",
        "train_config.cat_nan_policy": "Policy for Cat. Missing",
        "train_config.normalization": "Dataset Normalization",
    }
    numerical_name_remap = {
        "model_config.n_heads": "N. Heads",
        "model_config.n_layers": "N. Layers",
        "model_config.d_token": "Token Hidden Dim.",
        "model_config.d_ffn_factor": "Hidden Dim. Factor",
    }

    analysis = PlotAnalysis(
        df,
        save_dir=tmp_path.as_posix(),
        cache=True,
        optim_metrics={"val_acc": Optim.max},
        numerical_attributes=list(numerical_name_remap.keys()),
        categorical_attributes=list(categorical_name_remap.keys()),
    )

    attribute_name_remap = {**categorical_name_remap, **numerical_name_remap}
    analysis.make_figures(
        metric_name_remap={
            "val_acc": "Accuracy",
            "val_rmse": "RMSE",
        },
        attribute_name_remap=attribute_name_remap,
    )
    assert all(
        tmp_path.joinpath("violinplot", "val_acc", f"{file_name}.png").exists()
        for file_name in categorical_name_remap
    )
    assert all(
        tmp_path.joinpath("linearplot", "val_acc", f"{file_name}.png").exists()
        for file_name in numerical_name_remap
    )

    try:
        no_cache = PlotAnalysis(
            df,
            save_dir=tmp_path.as_posix(),
            cache=False,
            optim_metrics={"val_acc": Optim.min},
            numerical_attributes=list(numerical_name_remap.keys()),
            categorical_attributes=list(categorical_name_remap.keys()),
        )

        # checking whether if cache = False, clears memory.
        assert no_cache.cache == None

        # Ensuring that the incorrect path provided must not exists.
        assert not Path("/random_folder/nonexistent_folder/").parent.exists()

        incorrect_path = PlotAnalysis(
            df,
            save_dir="/nonexistent_folder/file.txt",
            cache=False,
            optim_metrics={"val_acc": Optim.min},
            numerical_attributes=list(numerical_name_remap.keys()),
            categorical_attributes=list(categorical_name_remap.keys()),
        )
    except FileNotFoundError as f:
        # Expecting the constructor to fail because of incorrect_path. 
        assert str(f).startswith("Save directory does not exist")



if __name__ == "__main__":
    # import shutil
    # from tests.ray_models.model import (
    #     WORKING_DIR,
    #     TestWrapper,
    #     MyErrorCustomModel,
    #     _make_config,
    # )
    # from tests.conftest import DockerRayCluster, _assert_error_msg
    # from tests.mp.test_main import test_mp_run

    # ray_cluster = DockerRayCluster()
    # ray_cluster.setUp()
    # tmp_path = Path("/tmp/save_dir")
    # shutil.rmtree(tmp_path, ignore_errors=True)
    # tmp_path.mkdir(exist_ok=True)
    # test_mp_run = test_mp_run(
    #     tmp_path,
    #     _assert_error_msg,
    #     ray_cluster,
    #     TestWrapper(MyErrorCustomModel),
    #     _make_config,
    #     WORKING_DIR,
    # )
    # config = _make_config(tmp_path, search_space_limit=10)
    # test_mp_run = Results(config, f"/tmp/save_dir/experiment_{config.uid}")
    # test_analysis(tmp_path)

    test_analysis_func(Path("/tmp/analysis_func"))

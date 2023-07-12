from pathlib import Path

import pandas as pd

from ablator import Optim, PlotAnalysis
from ablator.analysis.plot.utils import parse_name_remap
from ablator.analysis.main import Analysis


def get_best(x: pd.DataFrame, task_type: str):
    if task_type == "regression":
        return x.sort_values("val_rmse", na_position="last").iloc[0]
    else:
        return x.sort_values("val_acc", na_position="first").iloc[-1]


def test_analysis(tmp_path: Path):
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
        correct = PlotAnalysis(
            df,
            save_dir=tmp_path.as_posix(),
            cache=False,
            optim_metrics={"val_acc": Optim.min},
            numerical_attributes=list(numerical_name_remap.keys()),
            categorical_attributes=list(categorical_name_remap.keys()),
        )
        incorrect_path = PlotAnalysis(
            df,
            save_dir="/nonexistent_folder/file.txt",
            cache=False,
            optim_metrics={"val_acc": Optim.min},
            numerical_attributes=list(numerical_name_remap.keys()),
            categorical_attributes=list(categorical_name_remap.keys()),
        )
    except FileNotFoundError as f:
        if not str(f).startswith("Save directory does not exist"):
            raise f
    except Exception as e:
        raise e
    pass

        
def test_name_remap():
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
        
def test_get_best_results_by_metrics():
    results_csv = Path(__file__).parent.parent.joinpath("assets", "results.csv")
    df = pd.read_csv(results_csv)
    
    metric_map = {"val_acc": Optim.max, "val_loss" :Optim.min}
    report_results = Analysis._get_best_results_by_metric(df, metric_map)

    # check whether each "metric" from metric_map has all the results.
    assert (df['path'].unique().count() == v for _ , v in dict(report_results["best"].value_counts()))

    # check for Max values
    data1 = df.groupby("path")["val_acc"].agg(lambda x: x.max(skipna=True)).reset_index(name="val_acc")    
    data2 = report_results[report_results["best"] == "val_acc"][["path", "val_acc"]].reset_index(drop=True)
    merged_df = pd.merge(data1, data2, on="path", suffixes=("_df1", "_df2"))

    assert not (merged_df["val_acc_df1"] != merged_df["val_acc_df2"]).any()
    
    # check for Min values
    data1 = df.groupby("path")["val_loss"].agg(lambda x: x.min(skipna=True)).reset_index(name="val_loss")    
    data2 = report_results[report_results["best"] == "val_loss"][["path", "val_loss"]].reset_index(drop=True)
    merged_df = pd.merge(data1, data2, on="path", suffixes=("_df1", "_df2"))
    
    assert not (merged_df["val_loss_df1"] != merged_df["val_loss_df2"]).any()


if __name__ == "__main__":
    import shutil
    tmp_path = Path("/tmp/save_dir")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(exist_ok=True)
    test_analysis(tmp_path)

    test_name_remap()
    test_get_best_results_by_metrics()




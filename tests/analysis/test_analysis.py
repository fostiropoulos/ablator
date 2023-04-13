from pathlib import Path

import pandas as pd

from ablator import Optim, PlotAnalysis


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
    pass


if __name__ == "__main__":
    import shutil
    tmp_path = Path("/tmp/save_dir")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(exist_ok=True)
    test_analysis(tmp_path)

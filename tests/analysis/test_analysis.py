from pathlib import Path

import pandas as pd

from ablator import Optim, PlotAnalysis
from ablator.analysis.main import Analysis
from matplotlib.testing.compare import compare_images

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

#
def test_generated_plot_similarity(tmp_path: Path):
    #compare image results in violinplot
    model_config_activation = tmp_path.joinpath("violinplot", "val_acc", "model_config.activation.png")
    model_config_initialization = tmp_path.joinpath("violinplot", "val_acc", "model_config.initialization.png")
    model_config_mask_type = tmp_path.joinpath("violinplot", "val_acc", "model_config.mask_type.png")

    result_model_config_activation = compare_images(model_config_activation, Path(__file__).parent.parent.joinpath("assets", "violinplot", "model_config.activation.png"), 0.001, in_decorator=True)
    assert result_model_config_activation is None

    result_model_config_intialization = compare_images(model_config_initialization, Path(__file__).parent.parent.joinpath("assets", "violinplot", "model_config.initialization.png"), 0.001, in_decorator=True)
    assert result_model_config_intialization is None

    result_model_config_mask_type = compare_images(model_config_mask_type, Path(__file__).parent.parent.joinpath("assets", "violinplot", "model_config.mask_type.png"), 0.001, in_decorator=True)
    assert result_model_config_mask_type is None

    train_config_cat_nan_policy = tmp_path.joinpath("violinplot", "val_acc", "train_config.cat_nan_policy.png")
    train_config_normalization = tmp_path.joinpath("violinplot", "val_acc", "train_config.normalization.png")
    train_config_optimizer_config_name = tmp_path.joinpath("violinplot", "val_acc", "train_config.optimizer_config.name.png")

    result_train_config_cat_nan_policy = compare_images(train_config_cat_nan_policy, Path(__file__).parent.parent.joinpath("assets", "violinplot", "train_config.cat_nan_policy.png"), 0.001, in_decorator=True)
    assert result_train_config_cat_nan_policy is None

    result_train_config_normalization = compare_images(train_config_normalization, Path(__file__).parent.parent.joinpath("assets", "violinplot", "train_config.normalization.png"), 0.001, in_decorator=True)
    assert result_train_config_normalization is None

    result_train_config_optimizer_config_name = compare_images(train_config_optimizer_config_name, Path(__file__).parent.parent.joinpath("assets", "violinplot", "train_config.optimizer_config.name.png"), 0.001, in_decorator=True)
    assert result_train_config_optimizer_config_name is None

    #compare image results in linearplot
    model_config_d_ffn_factor = tmp_path.joinpath("linearplot", "val_acc", "model_config.d_ffn_factor.png")
    model_config_d_token = tmp_path.joinpath("linearplot", "val_acc", "model_config.d_token.png")
    model_config_n_heads = tmp_path.joinpath("linearplot", "val_acc", "model_config.n_heads.png")
    model_config_n_layers = tmp_path.joinpath("linearplot", "val_acc", "model_config.n_layers.png")

    result_model_config_d_ffn_factor = compare_images(model_config_d_ffn_factor, Path(__file__).parent.parent.joinpath("assets", "linearplot", "model_config.d_ffn_factor.png"), 0.001, in_decorator=True)
    assert result_model_config_d_ffn_factor is None

    result_model_config_d_token = compare_images(model_config_d_token, Path(__file__).parent.parent.joinpath("assets", "linearplot", "model_config.d_token.png"), 0.001, in_decorator=True)
    assert result_model_config_d_token is None

    result_model_config_n_heads = compare_images(model_config_n_heads, Path(__file__).parent.parent.joinpath("assets", "linearplot", "model_config.n_heads.png"), 0.001, in_decorator=True)
    assert result_model_config_n_heads is None

    result_config_n_layers = compare_images(model_config_n_layers, Path(__file__).parent.parent.joinpath("assets", "linearplot", "model_config.n_layers.png"), 0.001, in_decorator=True)
    assert result_config_n_layers is None

# https://github.com/fostiropoulos/ablator/pull/35
def test_get_best_results_by_metric():

    data = {'path': ['path1', 'path1', 'path2', 'path2', 'path3'],
            'metric1': [0.1, 0.2, 0.3, None, 0.5],
            'metric2': [None, 0.6, 0.7, 0.8, 0.9]}
    input = pd.DataFrame(data)

    result = pd.DataFrame({
        'path': ['path1', 'path2', 'path3', 'path1', 'path2', 'path3'],
        'metric1': [0.1, 0.3, 0.5, 0.2, None, 0.5],
        'metric2': [None, 0.7, 0.9, 0.6, 0.8, 0.9],
        'best': ['metric1', 'metric1', 'metric1', 'metric2', 'metric2', 'metric2']
    })

    metricMap = {'metric1': Optim.min, 'metric2': Optim.max}
    # call the function to be tested
    actual_output = Analysis._get_best_results_by_metric(input, metricMap)

    assert actual_output.equals(result)


if __name__ == "__main__":
    import shutil
    tmp_path = Path("/tmp/save_dir")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(exist_ok=True)
    test_analysis(tmp_path)
    test_generated_plot_similarity(tmp_path)

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from PIL import Image

from ablator.analysis.main import Analysis, _parse_results
from ablator.analysis.plot import Plot
from ablator.analysis.plot.cat_plot import Categorical, ViolinPlot
from ablator.analysis.plot.main import PlotAnalysis
from ablator.analysis.plot.num_plot import LinearPlot
from ablator.analysis.plot.utils import parse_name_remap
from ablator.config.proto import Optim


def test_name_remap(tmp_path: Path):
    name_map = {"a": "A", "b": "B"}
    defaults = None
    assert parse_name_remap(defaults, name_map) == name_map

    name_map = {"a": "A", "b": "B"}
    defaults = ["a", "b", "c", "d"]
    expected_output = {"a": "A", "b": "B", "c": "c", "d": "d"}
    assert parse_name_remap(defaults, name_map) == expected_output

    name_map = None
    defaults = ["a", "b", "c"]
    assert parse_name_remap(defaults, name_map) == {"a": "a", "b": "b", "c": "c"}

    try:
        parse_name_remap(None, None)
    except NotImplementedError as err:
        if str(err) != "`defaults` or `name_map` argument required.":
            raise err


def test_analysis_func(tmp_path: Path):
    results_csv = Path(__file__).parent.parent.joinpath("assets", "results.csv")
    df = pd.read_csv(results_csv)

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
        optim_metrics={"val_acc": "max"},
        numerical_attributes=list(numerical_name_remap),
        categorical_attributes=list(categorical_name_remap),
    )

    attribute_name_remap = {**categorical_name_remap, **numerical_name_remap}
    analysis.make_figures(
        metric_name_remap={
            "val_acc": "Accuracy",
            "val_rmse": "RMSE",
        },
        attribute_name_remap=attribute_name_remap,
    )

    def _exist(p: Path, file_names):
        return (p.joinpath(f"{file_name}.png").exists() for file_name in file_names)

    assert all(
        _exist(tmp_path.joinpath("violinplot", "val_acc"), categorical_name_remap)
    )
    assert all(_exist(tmp_path.joinpath("linearplot", "val_acc"), numerical_name_remap))
    assert not tmp_path.joinpath("linearplot", "val_rmse").exists()
    assert not tmp_path.joinpath("violinplot", "val_rmse").exists()
    no_cache = PlotAnalysis(
        df,
        save_dir=tmp_path.as_posix(),
        cache=False,
        optim_metrics={"val_rmse": Optim.min, "val_acc": "max"},
        numerical_attributes=list(numerical_name_remap.keys()),
        categorical_attributes=list(categorical_name_remap.keys()),
    )
    no_cache.make_figures(
        metric_name_remap={
            "val_acc": "Accuracy",
            "val_rmse": "RMSE",
        },
        attribute_name_remap=attribute_name_remap,
    )

    assert all(
        _exist(tmp_path.joinpath("violinplot", "val_rmse"), categorical_name_remap)
    )
    assert all(
        _exist(tmp_path.joinpath("linearplot", "val_rmse"), numerical_name_remap)
    )
    # checking whether if cache = False, clears memory.
    assert no_cache.cache is None

    with pytest.raises(FileNotFoundError, match="Save directory does not exist"):
        incorrect_path = PlotAnalysis(
            df,
            save_dir=tmp_path.joinpath("nonexistent_folder", "a"),
            cache=False,
            optim_metrics={"val_acc": Optim.min},
            numerical_attributes=list(numerical_name_remap.keys()),
            categorical_attributes=list(categorical_name_remap.keys()),
        )
    with pytest.raises(
        ValueError, match="Must provide a `save_dir` when specifying `cache=True`."
    ):
        PlotAnalysis(
            df,
            cache=True,
            optim_metrics={"val_acc": Optim.min},
            numerical_attributes=list(numerical_name_remap.keys()),
            categorical_attributes=list(categorical_name_remap.keys()),
        )
    with pytest.raises(
        ValueError,
        match=(
            "Must specify a `save_dir` either as an argument to `make_figures` or"
            " during class instantiation"
        ),
    ):
        PlotAnalysis(
            df,
            optim_metrics={"val_acc": Optim.min},
            numerical_attributes=list(numerical_name_remap.keys()),
            categorical_attributes=list(categorical_name_remap.keys()),
        ).make_figures()
    with pytest.raises(
        ValueError, match="No valid value was found for metric `val_acc`."
    ):
        PlotAnalysis(
            df[df["val_acc"].isna()],
            optim_metrics={"val_acc": Optim.min},
            numerical_attributes=list(numerical_name_remap.keys()),
            categorical_attributes=list(categorical_name_remap.keys()),
        ).make_figures(save_dir=tmp_path.joinpath("file"))
    PlotAnalysis(
        df[df["val_acc"].isna()],
        optim_metrics={"val_rmse": Optim.min},
        numerical_attributes=list(numerical_name_remap.keys()),
        categorical_attributes=list(categorical_name_remap.keys()),
    ).make_figures(save_dir=tmp_path.joinpath("file"))
    assert all(
        _exist(
            tmp_path.joinpath("file", "violinplot", "val_rmse"), categorical_name_remap
        )
    )
    assert all(
        _exist(
            tmp_path.joinpath("file", "linearplot", "val_rmse"), numerical_name_remap
        )
    )
    assert not tmp_path.joinpath("file", "linearplot", "val_acc").exists()
    assert not tmp_path.joinpath("file", "violinplot", "val_acc").exists()
    assert len(
        list(tmp_path.joinpath("file", "violinplot", "val_rmse").glob("*.png"))
    ) == len(categorical_name_remap)
    assert len(
        list(tmp_path.joinpath("file", "linearplot", "val_rmse").glob("*.png"))
    ) == len(numerical_name_remap)


class MockPlot(Plot):
    def _make(self, **kwargs):
        attributes = self.attributes.values
        metric = self.metric.values
        return attributes, metric


def test_plot(tmp_path: Path):
    metric = pd.Series(np.random.randn(100), name="val_acc")
    cat_attributes = pd.DataFrame(
        np.random.randint(100, size=(100, 10)), columns=[np.arange(10)]
    )
    with pytest.raises(ValueError, match="'x' is not a valid Optim"):
        MockPlot(metric=metric, attributes=cat_attributes, metric_obj_fn="x")
    p = MockPlot(metric=metric, attributes=cat_attributes, metric_obj_fn="min")
    _attr, _metric = p._make()
    assert (_attr == cat_attributes).all().all()
    assert (_metric == metric).all()
    metric[:50] = np.nan
    p = MockPlot(metric=metric, attributes=cat_attributes, metric_obj_fn="min")
    _attr, _metric = p._make()
    assert (_attr == cat_attributes[~metric.isna()]).all().all()
    assert (_metric == metric[~metric.isna()]).all()

    fig, ax = p._make_figure()
    assert (ax.get_xticks() == np.linspace(0, 1, len(ax.get_xticklabels()))).all()
    assert ax.get_legend() is None
    p._parse_figure_axis(ax)
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    assert ax.get_legend() is not None
    ax.get_xticklabels()
    p._parse_figure_axis(ax, x_axis="x", y_axis="y", labels=["a", "b", "c"])
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert (ax.get_xticks() == np.arange(3) + 1).all()
    assert (
        np.array([b.get_text() for b in ax.get_xticklabels()])
        == np.array(["a", "b", "c"])
    ).all()


def test_linear_plot():
    metric = pd.Series(np.random.randn(100), name="val_acc")
    num_attributes = pd.DataFrame(np.random.randn(100, 10), columns=[np.arange(10)])
    with pytest.raises(
        ValueError, match="LinearPlot attributes must be single dimensional."
    ):
        p = LinearPlot(metric=metric, attributes=num_attributes, metric_obj_fn="max")
        fig, ax = p._make()

    with pytest.raises(pd.errors.IndexingError, match="Unalignable boolean Series"):
        num_attributes = pd.DataFrame(np.random.randn(110, 1), columns=[np.arange(1)])
        p = LinearPlot(metric=metric, attributes=num_attributes, metric_obj_fn="max")
    num_attributes = pd.DataFrame(np.random.randn(100, 1), columns=[np.arange(1)])
    p = LinearPlot(metric=metric, attributes=num_attributes, metric_obj_fn="max")
    fig, ax = p._make()
    assert len(ax.lines) == 1
    x, y = ax.lines[0].get_data()
    _x = np.array(sorted(num_attributes.values.flatten()))
    _y = np.array(sorted(metric.values.flatten()))
    x = np.array(sorted(x))
    y = np.array(sorted(metric))
    assert np.isclose(y, _y).all()
    # TODO for some reason this is not true, but it is inherit to the seaborn library
    # assert np.isclose(x, _x).all()

    assert max(x) == max(_x) and min(x) == min(_x)

    num_attributes = pd.DataFrame(np.random.randn(100), columns=[np.arange(1)])
    p = LinearPlot(metric=metric, attributes=num_attributes, metric_obj_fn="max")
    fig, ax = p._make()
    assert len(ax.lines) == 1
    fig, ax = p._make()
    assert len(ax.lines) == 2
    fig, ax = p.make()
    assert ax.get_legend() is None


def test_categorical_plot(capture_output, capture_logger):
    out: io.StringIO = capture_logger()
    metric = pd.Series(np.random.randn(100), name="val_acc")
    cat_attributes = pd.Series(np.random.randint(10, size=(100)), name="attr")
    attribute_metric_map = Categorical._make_attribute_metric_map(
        metric, cat_attributes
    )

    assert set(attribute_metric_map.keys()) == set(range(10))

    assert np.isclose(
        sorted(pd.concat(attribute_metric_map.values())), sorted(metric)
    ).all()
    _, counts = np.unique(cat_attributes, return_counts=True)
    _counts = np.array([len(attribute_metric_map[i]) for i in range(10)])
    assert (counts == _counts).all()

    # sort_vals tests
    metric_min = Categorical._sort_vals_obj(None, metric, "min")
    metric_max = Categorical._sort_vals_obj(None, metric, "max")
    assert np.isclose(metric_min, np.sort(metric)).all()
    assert np.isclose(metric_max, np.sort(metric)[::-1]).all()
    metric[:10] = np.nan
    metric_min = Categorical._sort_vals_obj(None, metric, "min")
    metric_max = Categorical._sort_vals_obj(None, metric, "max")
    assert np.isnan(metric_min[-10:]).all()
    assert np.isnan(metric_max[-10:]).all()

    # missing values ignored
    cat_attributes = cat_attributes.astype(object)
    cat_attributes.iloc[:10] = None
    cat_attributes.iloc[10:20] = np.nan
    cat_attributes.iloc[20:30] = "None"
    cat_attributes.iloc[30:40] = "Type(None)"
    cat_attributes.iloc[40:50] = "nan"
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Type(None), and `None` are both present as categorical values. Unable to"
            " rename None value."
        ),
    ):
        attribute_metric_map = Categorical._make_attribute_metric_map(
            metric, cat_attributes
        )

    cat_attributes.iloc[30:40] = "Type(None)s"

    metric = pd.Series(np.random.randn(100), name="val_acc")

    stdout, stderr = capture_output(
        lambda: Categorical._make_attribute_metric_map(metric, cat_attributes)
    )
    assert (
        "`None` is present as a categorical string value as well as None. Will rename"
        " None to Type(None)."
        in out.getvalue()
    )

    attribute_metric_map = Categorical._make_attribute_metric_map(
        metric, cat_attributes
    )

    for i, v in enumerate(["Type(None)", np.nan, "None", "Type(None)s", "nan"]):
        upper = (i + 1) * 10
        lower = i * 10
        assert np.isclose(
            attribute_metric_map[v].values, metric[lower:upper].values
        ).all()


def test_violin_plot():
    metric = pd.Series(np.random.randn(100), name="val_acc")
    cat_attributes = pd.DataFrame(np.random.randint(10, size=(100, 10)))
    with pytest.raises(
        ValueError, match="ViolinPlot attributes must be single dimensional."
    ):
        p = ViolinPlot(metric=metric, attributes=cat_attributes, metric_obj_fn="max")
        fig, ax = p._make()
    with pytest.raises(pd.errors.IndexingError, match="Unalignable boolean Series"):
        cat_attributes = pd.Series(np.random.randn(110), name="attr")
        p = ViolinPlot(metric=metric, attributes=cat_attributes, metric_obj_fn="max")

    n_categories = 10
    cat_attributes = pd.Series(np.random.randint(n_categories, size=(100)), name="attr")
    p = ViolinPlot(metric=metric, attributes=cat_attributes, metric_obj_fn="max")
    fig, ax = p._make()
    assert len(ax.lines) == n_categories * 3
    x, (y_min, y_max) = ax.lines[0].get_data()
    assert y_min == min(p.attribute_metric_map[0])
    assert y_max == max(p.attribute_metric_map[0])
    assert all(
        l.get_text()
        == f"Mean: {p.attribute_metric_map[i].mean():.2e}\nBest:"
        f" {p.attribute_metric_map[i].max():.2e}\n{i}"
        for i, l in enumerate(ax.get_xticklabels())
    )


def test_get_best_results_by_metric():
    # Create a sample dataframe
    raw_results = pd.DataFrame(
        {
            "trial_uid": [1, 1, 2, 2, 3],
            "metric_1": [0.5, 0.6, 0.7, 0.8, 0.9],
            "metric_2": [0.3, 0.4, 0.2, 0.1, 0.5],
        }
    )

    metric_map = {"metric_1": Optim.min, "metric_2": Optim.max}

    # Call the _get_best_results_by_metric method
    result_df = Analysis._get_best_results_by_metric(raw_results, metric_map)

    # Define expected results
    expected_columns = ["trial_uid", "metric_1", "metric_2", "best"]

    # Check if the result DataFrame has the expected columns
    assert list(result_df.columns) == expected_columns

    # Check if the best results are selected correctly for each metric
    assert result_df.loc[result_df["best"] == "metric_1", "metric_1"].min() == 0.5
    assert result_df.loc[result_df["best"] == "metric_2", "metric_2"].max() == 0.5


def test_remap_results():
    # Create sample dataframes
    attributes = pd.DataFrame({"color": ["red", "blue"], "size": [10, 20]})
    metrics = pd.DataFrame({"loss": [0.5, 0.4], "accuracy": [0.8, 0.9]})
    metric_map = {"loss": Optim.min, "accuracy": Optim.max}
    metric_name_remap = {"loss": "error", "accuracy": "acc"}
    attribute_name_remap = {"color": "c", "size": "s"}

    # Call the _remap_results method
    remapped_attrs, remapped_metrics, updated_map = Analysis._remap_results(
        attributes,
        metrics,
        metric_map,
        metric_name_remap=metric_name_remap,
        attribute_name_remap=attribute_name_remap,
    )

    # Check if the remapping worked correctly
    assert list(remapped_attrs.columns) == ["c", "s"]
    assert list(remapped_metrics.columns) == ["error", "acc"]
    assert updated_map == {"error": Optim.min, "acc": Optim.max}


def test_parse_results_dataframe():
    # Load the CSV data
    results_csv_path = Path(__file__).parent.parent.joinpath("assets", "results.csv")
    data = pd.read_csv(results_csv_path)

    # Expected attributes
    cat_attrs_list = [
        "model_config.activation",
        "model_config.initialization",
        "train_config.optimizer_config.name",
        "model_config.mask_type",
        "train_config.cat_nan_policy",
        "train_config.normalization",
    ]

    num_attrs_list = [
        "model_config.n_heads",
        "model_config.n_layers",
        "model_config.d_token",
        "model_config.d_ffn_factor",
        "val_acc",
        "val_rmse",
    ]

    # For the sake of the example, I'm assuming these optimization metrics:
    metrics_map = {"val_acc": Optim.max, "val_rmse": Optim.min}

    # Test scenario: Provide a DataFrame and necessary attributes
    parsed_df, cat_attrs, num_attrs, metrics = _parse_results(
        data, cat_attrs_list, num_attrs_list, metrics_map
    )

    assert parsed_df.equals(data)
    assert set(cat_attrs) == set(cat_attrs_list)
    assert set(num_attrs) == set(num_attrs_list)
    assert metrics == metrics_map


def test_parse_invalid_results():
    with pytest.raises(ValueError, match=r"Invalid value .*"):
        _parse_results("invalid_value")


def test_analysis_numerical_attributes_none():
    # Create a sample DataFrame (for testing purposes)
    sample_data = {"color": ["red", "blue"], "size": [10, 20]}
    sample_df = pd.DataFrame(sample_data)

    # Define the categorical attributes
    categorical_attributes = ["color"]

    # Define the optimization metrics
    optim_metrics = {"loss": "min", "accuracy": "max"}

    # This test expects a ValueError to be raised when numerical_attributes is None
    with pytest.raises(ValueError, match="Must provide `_numerical_attributes`"):
        Analysis(
            results=sample_df,
            categorical_attributes=categorical_attributes,
            numerical_attributes=None,
            optim_metrics=optim_metrics,
        )


def test_analysis_categorical_attributes_none():
    # Create a sample DataFrame (for testing purposes)
    sample_data = {"color": ["red", "blue"], "size": [10, 20]}
    sample_df = pd.DataFrame(sample_data)

    # Define the numerical attributes
    numerical_attributes = ["size"]

    # Define the optimization metrics
    optim_metrics = {"loss": "min", "accuracy": "max"}

    # This test expects a ValueError to be raised when categorical_attributes is None
    with pytest.raises(ValueError, match="Must provide `categorical_attributes`"):
        Analysis(
            results=sample_df,
            categorical_attributes=None,
            numerical_attributes=numerical_attributes,
            optim_metrics=optim_metrics,
        )


def test_analysis_optim_metrics_none():
    # Create a sample DataFrame (for testing purposes)
    sample_data = {"color": ["red", "blue"], "size": [10, 20]}
    sample_df = pd.DataFrame(sample_data)

    # Define the categorical attributes
    categorical_attributes = ["color"]

    # Define the numerical attributes
    numerical_attributes = ["size"]

    # This test expects a ValueError to be raised when optim_metrics is None
    with pytest.raises(
        ValueError,
        match="Missing `optim_metrics` or unable to derive from supplied results.",
    ):
        Analysis(
            results=sample_df,
            categorical_attributes=categorical_attributes,
            numerical_attributes=numerical_attributes,
            optim_metrics=None,
        )


# def test_generated_plot_similarity(tmp_path: Path):
#     # 1. Load data
#     results_csv = Path(__file__).parent.parent.joinpath("assets", "results.csv")
#     df = pd.read_csv(results_csv)

#     # 2. Set remapping variables
#     categorical_name_remap = {
#         "model_config.activation": "Activation",
#     }
#     numerical_name_remap = {
#         "model_config.n_heads": "N. Heads",
#     }
#     attribute_name_remap = {**categorical_name_remap, **numerical_name_remap}

#     # 3. Create analysis object
#     analysis = PlotAnalysis(
#         df,
#         save_dir=tmp_path.as_posix(),
#         cache=True,
#         optim_metrics={"val_acc": "max"},
#         numerical_attributes=list(numerical_name_remap),
#         categorical_attributes=list(categorical_name_remap),
#     )

#     # 4. Generate plots
#     analysis.make_figures(
#         metric_name_remap={
#             "val_acc": "Accuracy",
#             "val_rmse": "RMSE",
#         },
#         attribute_name_remap=attribute_name_remap,
#     )
#     # Violin Plot Comparison
#     model_config_activation = (
#         tmp_path / "violinplot" / "val_acc" / "model_config.activation.png"
#     )
#     result_model_config_activation = compare_images(
#         model_config_activation,
#         Path(__file__).parent.parent
#         / "assets"
#         / "violinplot"
#         / "model_config.activation.png",
#         0.20,
#         in_decorator=True,
#     )
#     assert result_model_config_activation is None

#     # Linear Plot Comparison

#     model_config_n_heads = (
#         tmp_path / "linearplot" / "val_acc" / "model_config.n_heads.png"
#     )
#     result_model_config_n_heads = compare_images(
#         model_config_n_heads,
#         Path(__file__).parent.parent
#         / "assets"
#         / "linearplot"
#         / "model_config.n_heads.png",
#         0.20,
#         in_decorator=True,
#     )
#     assert result_model_config_n_heads is None


def test_linear_plot_non_numerical_attributes():
    metric = pd.Series(np.random.randn(100), name="val_acc")
    str_attributes = pd.Series(["attr_{}".format(i % 10) for i in range(100)])

    with pytest.raises(TypeError):
        p = LinearPlot(metric=metric, attributes=str_attributes, metric_obj_fn="max")
        p._make()


def test_write_images_for_figure_and_images(tmp_path):
    fig = plt.figure()
    fig_map = {"figure_example": fig}
    img = Image.new("RGB", (60, 30), color=(73, 109, 137))
    fig_map_img = {"image_example": img}
    PlotAnalysis._write_images(fig_map, tmp_path, "png")
    PlotAnalysis._write_images(fig_map_img, tmp_path, "png")

    # Check if the file has been saved
    assert (tmp_path / "figure_example.png").exists()
    assert (tmp_path / "image_example.png").exists()


if __name__ == "__main__":
    from tests.conftest import run_tests_local

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]
    run_tests_local(test_fns)

from pathlib import Path
import mock

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
import pytest
from ablator.analysis.plot.cat_plot import ViolinPlot
from ablator.analysis.plot.num_plot import LinearPlot
from ablator.analysis.plot.utils import get_axes_fig


n_categories = 10


@pytest.fixture()
def metric():
    return pd.Series(np.random.randn(100), name="val_acc")


@pytest.fixture()
def cat_attributes():
    return pd.Series(np.random.randint(n_categories, size=(100)), name="attr")


def fig2img(fig: Figure):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def parse_image(img: Image.Image | Figure) -> np.ndarray:
    if isinstance(img, Figure):
        return np.array(fig2img(img))
    return np.array(img)


def rmse(expected_image: Image.Image, actual_image: Image.Image):
    expected_image = parse_image(expected_image)
    actual_image = parse_image(actual_image)
    return np.sqrt(((expected_image - actual_image).astype(float) ** 2).mean())


@pytest.mark.parametrize("plot_class", [ViolinPlot, LinearPlot])
def test_passing_ax(plot_class: ViolinPlot | LinearPlot, metric, cat_attributes):
    plt_fig = plt.figure(figsize=(4, 4))
    plt_ax = plt_fig.add_subplot(1, 1, 1)
    blank_tmp_img = fig2img(plt_fig)
    plot = plot_class(
        metric=metric, attributes=cat_attributes, ax=plt_ax, metric_obj_fn="max"
    )
    plt_fig_new, plt_ax_new = plot._make_figure()
    assert plt_fig_new != plot.figure
    fig, ax = plot._make()
    tmp_img = fig2img(fig)
    assert rmse(fig, plt_fig) == 0
    assert rmse(fig, blank_tmp_img) > 30
    assert id(get_axes_fig(plt_ax)) == id(fig) and id(get_axes_fig(plt_ax)) == id(
        plt_fig
    )
    plot_class(metric=metric, attributes=cat_attributes, ax=plt_ax, metric_obj_fn="max")
    new_fig, new_ax = plot._make()
    assert id(get_axes_fig(plt_ax)) == id(fig) and id(get_axes_fig(plt_ax)) == id(
        plt_fig
    )
    assert rmse(fig, tmp_img) > 10


@pytest.mark.parametrize("plot_class", [ViolinPlot, LinearPlot])
def test_remake_figure(tmp_path: Path, plot_class, metric, cat_attributes):
    plot = plot_class(metric=metric, attributes=cat_attributes, metric_obj_fn="max")
    fig, ax = plot._make()

    plot = plot_class(metric=metric, attributes=cat_attributes, metric_obj_fn="max")
    fig2, ax2 = plot._make()

    assert rmse(fig, fig2) == 0
    fig.savefig(tmp_path / "fig_1.png")

    fig2.savefig(tmp_path / "fig_4.png")

    ax2.remove()
    assert rmse(fig, get_axes_fig(ax2)) > 30


def test_get_axes_fig():
    plt.close()
    prev_num = len(plt.get_fignums())
    plt_fig = plt.figure(figsize=(4, 4))
    plt_ax = plt_fig.add_subplot(1, 1, 1)
    plt_ax.remove()
    assert len(plt.get_fignums()) == prev_num + 1
    plt.close(plt_fig)
    assert len(plt.get_fignums()) == prev_num
    with (
        pytest.raises(RuntimeError, match="Can not find an active plot."),
        mock.patch("matplotlib.pyplot.get_fignums", returnt=0),
    ):
        get_axes_fig(plt_ax)


if __name__ == "__main__":
    from tests.conftest import run_tests_local

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]
    kwargs = {
        "metric": pd.Series(np.random.randn(100), name="val_acc"),
        "cat_attributes": pd.Series(
            np.random.randint(n_categories, size=(100)), name="attr"
        ),
    }
    run_tests_local(test_fns, kwargs)

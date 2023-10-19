import logging
from abc import ABC, abstractmethod
import typing as ty

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from ablator.analysis.plot.utils import get_axes_fig

from ablator.config.proto import Optim

logger = logging.getLogger(__name__)


class Plot(ABC):
    """
    A base class for parsing experiment results and plotting them with ``pandas`` and ``matplotlib``.

    Parameters
    ----------
    metric : pd.Series
        The ablation study metric values to plot.
    attributes : pd.Series
        The ablation study attributes values to plot.
    metric_obj_fn : Optim | str
        The metric optimization direction.
    y_axis : str | None
        The y-axis label (metric name), by default ``None``.
    x_axis : str | None
        The x-axis label (attribute name), by default ``None``.
    x_ticks : list[str] | None
        The x-axis ticks, by default ``None``.
    ax : Axes | None
        The axes to plot on, by default ``None``.

    Attributes
    ----------
    metric : pd.Series
        The ablation study metric values to plot (with null value removed).
    attributes : pd.Series
        The ablation study attributes values to plot (with null metric value removed).
    metrics_obj_fn : Optim
        The metric optimization direction.
    y_axis : str
        The y-axis label (metric name).
    x_axis : str
        The x-axis label (attribute name).
    x_ticks : list[str]
        The x-axis ticks.
    figure : Figure
        The figure to plot on. If None, new figure of size (4,4) will be created.
    ax : Axes
        The axes to plot on. If None, a new axis will be created as the first subplot
        in the first cell and first column of a 1x1 grid.
    """

    def __init__(
        self,
        metric: pd.Series,
        attributes: pd.Series,
        metric_obj_fn: Optim | str,
        y_axis: str | None = None,
        x_axis: str | None = None,
        x_ticks: list[str] | None = None,
        ax: Axes | None = None,
    ) -> None:
        self.attributes = self._parse_attributes(metric, attributes)
        self.metric = self._parse_metrics(metric)
        self.metrics_obj_fn: Optim = Optim(metric_obj_fn)
        self.y_axis = y_axis
        self.x_axis = x_axis
        self.x_ticks = x_ticks
        self.figure, self.ax = self._make_figure(ax)

    def _make_figure(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        figure = None
        if ax is None:
            figure = plt.figure(figsize=(4, 4))
            ax = figure.add_subplot(1, 1, 1)
        else:
            figure = get_axes_fig(ax)
        return figure, ax

    def _parse_attributes(self, metric: pd.Series, attributes: pd.Series) -> pd.Series:
        attributes = attributes[~metric.isna()]
        return attributes

    def _parse_metrics(self, metric: pd.Series) -> pd.Series:
        if metric.isna().all():
            raise ValueError(f"No valid value was found for metric `{metric.name}`.")
        metric = metric[~metric.isna()]
        return metric

    def _parse_legend(self, ax: Axes):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05))

    def _parse_figure_axis(
        self,
        ax: Axes,
        x_axis: str | None = None,
        y_axis: str | None = None,
        labels: list[str] | None = None,
    ):
        if labels is not None:
            ax.set_xticklabels(labels, size=14)
            ax.set_xticks(np.arange(len(labels)) + 1)
        if x_axis is not None:
            ax.set_xlabel(x_axis, size=18)
        if y_axis is not None:
            ax.set_ylabel(y_axis, size=18)
        self._parse_legend(ax)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

        get_axes_fig(ax).tight_layout()

    def make(self, **kwargs: ty.Any) -> tuple[Figure, Axes]:
        fig, ax = self._make(**kwargs)
        self._parse_figure_axis(ax, self.x_axis, self.y_axis, self.x_ticks)
        return fig, ax

    @abstractmethod
    def _make(self, **kwargs: ty.Any):
        pass

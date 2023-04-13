import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ablator.config.run import Optim

logger = logging.getLogger(__name__)


class Plot(ABC):
    def __init__(
        self,
        metric: pd.Series,
        attributes: pd.Series,
        metric_obj_fn: Optim,
        y_axis: Optional[str] = None,
        x_axis: Optional[str] = None,
        x_ticks: Optional[np.ndarray] = None,
        ax: Optional[Axes] = None,
    ) -> None:

        self.attributes = self._parse_attributes(metric, attributes)
        self.metric = self._parse_metrics(metric)
        self.metrics_obj_fn = metric_obj_fn
        self.y_axis = y_axis
        self.x_axis = x_axis
        self.x_ticks = x_ticks
        self.figure, self.ax = self._make_figure(ax)

    def _make_figure(self,ax: Optional[Axes] = None) -> Tuple[Optional[Figure], Axes]:
        figure = None
        if ax is None:
            figure = plt.figure(figsize=(4,4))
            ax = figure.add_subplot(1, 1, 1)
        return figure, ax

    def _parse_attributes(self, metric: pd.Series, attributes: pd.Series) -> pd.Series:
        attributes = attributes[~metric.isna()]
        return attributes

    def _parse_metrics(self, metric: pd.Series) -> pd.Series:
        metric = metric[~metric.isna()]
        return metric

    def _parse_legend(self, ax):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05))

    def _parse_figure_axis(
        self,
        ax: Axes,
        x_axis: Optional[str] = None,
        y_axis: Optional[str] = None,
        labels: Optional[List[str]] = None,
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

        ax.figure.tight_layout()

    def make(self, **kwargs):
        fig, ax = self._make(**kwargs)
        self._parse_figure_axis(ax, self.x_axis, self.y_axis, self.x_ticks)
        return fig, ax

    @abstractmethod
    def _make(self, **kwargs):
        pass

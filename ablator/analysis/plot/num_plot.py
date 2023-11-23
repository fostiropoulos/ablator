import logging

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ablator.analysis.plot import Plot

logger = logging.getLogger(__name__)


class Numerical(Plot):
    """
    Base class for numerical plots

    Attributes
    ----------
    DATA_TYPE: str
        data_type for numerical plots.

    """

    DATA_TYPE: str = "numerical"


class LinearPlot(Numerical):
    """
    Class for generating linear plots

    Parameters
    ----------
    ax: Axes
        axes object of linear plot.
    figure: Figure
        Its corresponding figure object.
    """

    def _make(
        self,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        attributes = self.attributes.values
        if len(attributes.shape) > 1 and attributes.shape[-1] > 1:
            raise ValueError("LinearPlot attributes must be single dimensional.")
        metric = self.metric.values
        df = pd.concat(
            [
                pd.DataFrame(attributes, columns=["x"]),
                pd.DataFrame(metric, columns=["y"]),
            ],
            axis=1,
        )
        sns.regplot(
            df, x="x", y="y", ax=self.ax, marker=".", scatter_kws={"alpha": 0.3}, seed=0
        )
        return self.figure, self.ax

    def _parse_legend(self, ax: Axes):
        pass

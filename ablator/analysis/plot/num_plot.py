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
        scatter_plot: bool = True,
        polynomial_fit: int | None = None,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        if not scatter_plot and polynomial_fit is None:
            raise ValueError(
                "Must specify `polynomial_fit` when setting `scatter_plot` to False."
            )
        attributes = self.attributes.values
        metric = self.metric.values
        df = pd.concat(
            [
                pd.DataFrame(attributes, columns=["x"]),
                pd.DataFrame(metric, columns=["y"]),
            ],
            axis=1,
        )
        g = sns.regplot(
            df, x="x", y="y", ax=self.ax, marker=".", scatter_kws={"alpha": 0.3}
        )
        self.ax = g
        self.figure = g.figure

        return self.figure, self.ax

    def _parse_legend(self, ax):
        pass

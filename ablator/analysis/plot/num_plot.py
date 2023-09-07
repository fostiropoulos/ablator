import logging

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ablator.analysis.plot import Plot

logger = logging.getLogger(__name__)


class Numerical(Plot):
    DATA_TYPE = "numerical"


class LinearPlot(Numerical):
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
        g = sns.regplot(
            df, x="x", y="y", ax=self.ax, marker=".", scatter_kws={"alpha": 0.3}
        )
        self.ax = g
        self.figure = g.figure
        return self.figure, self.ax

    def _parse_legend(self, ax):
        pass

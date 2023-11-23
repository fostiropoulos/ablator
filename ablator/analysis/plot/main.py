from collections import abc
import logging
import typing as ty
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image

from ablator.analysis.main import Analysis
from ablator.analysis.plot import Plot
from ablator.analysis.plot.cat_plot import ViolinPlot
from ablator.analysis.plot.num_plot import LinearPlot
from ablator.analysis.plot.utils import get_axes_fig
from ablator.config.proto import Optim

logger = logging.getLogger(__name__)


class PlotAnalysis(Analysis):
    """
    Class for plotting experiment results. You can use this class and ``Results`` class to visualize the
    relationship between the result metrics and any hyperparameter you run ablation study on. This
    valuable insight offers an intuitive understanding of how these parameters may influence your
    model's performance.

    Plots supported are linear plots for numerical data and violin plots for categorical data.

    Parameters
    ----------
    results : pd.DataFrame | Results
        The result dataframe.
    categorical_attributes : list[str] | None
        The list of all the categorical hyperparameter names
    numerical_attributes : list[str] | None
        The list of all the numerical hyperparameter names
    optim_metrics : dict[str, Optim] | None
        A dictionary mapping metric names to optimization directions.
    save_dir : str | None
        The directory to save analysis results to.
    cache : bool
        Whether to cache results.

    Examples
    --------
    - Data frame to be used:

    >>> df = pd.DataFrame({'val_accuracy': np.random.uniform(0.8,0.9,10),
    ...       'train_config.optimizer_config.arguments.lr': np.random.uniform(0.001, 0.1,10),
    ...       "index": range(10),
    ...       "path": range(10)})

    - Creating dictionaries that map the configuration parameters [categorical + numerical] to custom labels for plots:

    >>> numerical_name_remap = {
    ...     "train_config.optimizer_config.arguments.lr": "Learning Rate",
    ... }
    ... categorical_name_remap = {}
    ... attribute_name_remap = {**categorical_name_remap, **numerical_name_remap}

    - Initalize the ``PlotAnalysis`` and plot the figures:

    >>> analysis = PlotAnalysis(
    ...     df,
    ...     save_dir="./plots",
    ...     cache=True,
    ...     optim_metrics={"val_accuracy": Optim.max},
    ...     numerical_attributes=list(numerical_name_remap.keys()),
    ...     categorical_attributes=list(categorical_name_remap.keys()),
    ... )
    >>> analysis.make_figures(
    ...    metric_name_remap={
    ...        "val_accuracy": "Validation Accuracy",
    ...    },
    ...    attribute_name_remap= attribute_name_remap
    ... )

    The directory ``"plots"`` contains all the plots of the HPO experiments
    """

    @classmethod
    def _write_images(
        cls,
        fig_map: abc.Mapping[str, ty.Union[Axes, Figure, Image.Image]],
        path: Path,
        file_format: ty.Literal["png", "pdf", "jpg"] = "png",
    ):
        """
        Write images to a directory based on figure types.

        Parameters
        ----------
        fig_map : dict[str, ty.Union[Axes, Figure, Image.Image]]
            A dictionary mapping names to ``matplotlib`` objects.
        path : Path
            Path to save the images to.
        file_format : ty.Literal["png", "pdf", "jpg"]
            the file format to save the images as.

        Examples
        --------
        >>> fig_map = {"figure1": plt.subplots()[0]}
        >>> path = Path("output_dir")
        >>> PlotAnalysis._write_images(fig_map, path, "png")
        """
        path.mkdir(exist_ok=True, parents=True)
        for name, fig in fig_map.items():
            img_path = path.joinpath(f"{name}.{file_format}")
            if isinstance(fig, Axes):
                get_axes_fig(fig).savefig(img_path)
            elif isinstance(fig, Figure):
                fig.savefig(img_path)
            elif isinstance(fig, Image.Image):
                fig.save(img_path)

    @classmethod
    def _make_metric_plots(
        cls,
        path: Path | None,
        plot_cls: type[Plot],
        metrics: pd.DataFrame,
        results: pd.DataFrame,
        metric_map: dict[str, Optim],
        append: bool = False,
        ax: Axes | None = None,
        metric_name_remap: dict[str, str] | None = None,
        attribute_name_remap: dict[str, str] | None = None,
        **kwargs,
    ) -> dict:
        """
        Create the attributes vs. metrics plots.

        Parameters
        ----------
        path : Path | None
            A ``pathlib.Path`` object representing the directory to write images to.
        plot_cls : type[Plot]
            A subclass of ``Plot`` representing the type of plot to make.
        metrics : pd.DataFrame
            A pandas ``DataFrame`` containing metric values.
        results : pd.DataFrame
            A pandas ``DataFrame`` containing attribute values.
        metric_map : dict[str, Optim]
            A dictionary mapping metric names to optimization functions.
        append : bool
            A boolean indicating whether to append plots to an existing axes object.
        ax : Axes | None
            A ``matplotlib.axes.Axes`` object representing the axis to plot on.
        metric_name_remap : dict[str, str] | None
            An optional dictionary mapping metric names to new metric names.
        attribute_name_remap : dict[str, str] | None
            An optional dictionary mapping attribute names to new attribute names.
        **kwargs
            Additional keyword arguments to pass to the plot method.

        Returns
        -------
        dict
            A dictionary of metric name to its axes_map

        Examples
        --------
        >>> metrics = pd.DataFrame({"metric1": [1, 2, 3], "metric2": [4, 5, 6]})
        >>> results = pd.DataFrame({"attr1": [7, 8, 9], "attr2": [10, 11, 12]})
        >>> metric_map = {"metric1": Optim.max, "metric2": Optim.min}
        >>> PlotAnalysis._make_metric_plots(None, LinearPlot, metrics, results, metric_map, False, None, None, None)
        """
        axes = {}

        (results, metrics, metric_map) = cls._remap_results(
            results, metrics, metric_map, metric_name_remap, attribute_name_remap
        )
        inv_metric_name_remap = None
        if metric_name_remap is not None:
            inv_metric_name_remap = {v: k for k, v in metric_name_remap.items()}
        inv_attribute_name_remap = None
        if attribute_name_remap is not None:
            inv_attribute_name_remap = {v: k for k, v in attribute_name_remap.items()}

        for metric_name in metrics.columns:
            metric_values = metrics[metric_name]
            metric_obj_fn = metric_map[metric_name]
            axes_map = cls._make_plot(
                plot_cls=plot_cls,
                metric_values=metric_values,
                metric_obj_fn=metric_obj_fn,
                results=results,
                append=append,
                ax=ax,
                inv_attribute_name_map=inv_attribute_name_remap,
                **kwargs,
            )
            if inv_metric_name_remap is not None:
                original_metric_name = inv_metric_name_remap[metric_name]
            else:
                original_metric_name = metric_name
            if path is not None:
                p = path.joinpath(original_metric_name)
                cls._write_images(axes_map, p)
            for axe in axes_map.values():
                plt.close(get_axes_fig(axe))
            axes[metric_name] = axes_map
        return axes

    @classmethod
    def _make_plot(
        cls,
        metric_values: pd.Series,
        results: pd.DataFrame,
        plot_cls: type[Plot],
        metric_obj_fn: Optim,
        append=False,
        ax: Axes | None = None,
        inv_attribute_name_map: dict[str, str] | None = None,
        **kwargs,
    ) -> dict[str, Axes]:
        axes_map = {}
        for attribute_name in results.columns:
            attribute_values = results[attribute_name]
            _, axes = plot_cls(
                metric=metric_values,
                attributes=attribute_values,
                metric_obj_fn=metric_obj_fn,
                y_axis=metric_values.name,
                x_axis=attribute_name,
                ax=ax,
            ).make(**kwargs)
            if inv_attribute_name_map is not None:
                original_attribute_name = inv_attribute_name_map[attribute_name]
            else:
                original_attribute_name = attribute_name
            if append:
                ax = axes
            else:
                axes_map[original_attribute_name] = axes
                plt.close()
        if append:
            axes_map["combined_attribute"] = axes
            plt.close()
        return axes_map

    def make_violinplot(
        self,
        attribute_names: list[str],
        metrics: list[str],
        save_dir: ty.Union[Path, str],
        **plt_kwargs,
    ):
        """
        Make violin plots for the given attribute names (data type: discrete) v.s. the metrics
        and save the plots to the `save_dir` directory.

        Parameters
        ----------
        attribute_names : list[str]
            list of attributes to plot against the metrics.
        metrics : list[str]
            list of metrics to plot against the given attributes.
        save_dir : ty.Union[Path, str]
            directory to save results.
        **plt_kwargs
            Additional keyword arguments to pass to the plot.
        """
        save_path = Path(save_dir).joinpath("violinplot")
        metric_map = {k: v for k, v in self.optim_metrics.items() if k in metrics}
        self._make_metric_plots(
            path=save_path,
            plot_cls=ViolinPlot,
            metrics=self.results[metrics],
            results=self.results[attribute_names],
            metric_map=metric_map,
            **plt_kwargs,
        )

    def make_linearplot(
        self,
        attribute_names: list[str],
        metrics: list[str],
        save_dir: ty.Union[Path, str],
        **plt_kwargs,
    ) -> dict:
        """
        To make linear plots for the given attribute names (data type: numerical) v.s. the metrics
        and save the plots to the `save_dir` directory.

        Parameters
        ----------
        attribute_names : list[str]
            list of attributes to plot against the metrics.
        metrics : list[str]
            list of metrics to plot against the given attributes.
        save_dir : ty.Union[Path, str]
            directory to save results.
        **plt_kwargs
            Additional keyword arguments to pass to the plot method.

        Returns
        -------
        dict
        """
        save_path = Path(save_dir).joinpath("linearplot")
        metric_map = {k: v for k, v in self.optim_metrics.items() if k in metrics}

        return self._make_metric_plots(
            path=save_path,
            plot_cls=LinearPlot,
            metrics=self.results[metrics],
            results=self.results[attribute_names],
            metric_map=metric_map,
            **plt_kwargs,
        )

    # flake8: noqa: DOC102
    # pylint: disable=differing-type-doc,differing-param-doc
    def make_figures(
        self,
        metric_name_remap: dict[str, str] | None = None,
        attribute_name_remap: dict[str, str] | None = None,
        save_dir: str | Path | None = None,
        **plt_kwargs: ty.Any,
    ):
        """
        Generate violin plots for categorical values and linear plots for numerical values.
        Plots are created as metrics vs. attributes. Additional keyword arguments to pass to
        the plot method are ``ax`` and ``append``:

        Parameters
        ----------
        metric_name_remap : dict[str, str] | None
            mappings for config's metrics keys to user defined names, by default ``None``.
        attribute_name_remap : dict[str, str] | None
            mappings for config's searchspace names to user defined names for attributes, by default ``None``.
        save_dir : str | Path | None
            optional directory of where to save the results, when unspecified, it expects one set during
            class initialization, by default ``None``.
        ax : Axes | None
            A ``matplotlib.axes.Axes`` object representing the axis to plot on),
        append : bool
            A boolean indicating whether to append plots to an existing axes object)
            and extra arguments for creating the plots.
        **plt_kwargs
            Additional keyword arguments to pass to the plot.
        """
        cat_attrs = list(self.categorical_attributes)
        num_attrs = list(self.numerical_attributes)
        if attribute_name_remap is not None:
            cat_attrs = list(set(attribute_name_remap.keys()).intersection(cat_attrs))
            num_attrs = list(set(attribute_name_remap.keys()).intersection(num_attrs))
        if (save_dir := save_dir if save_dir is not None else self.save_dir) is None:
            raise ValueError(
                "Must specify a `save_dir` either as an argument to `make_figures` or"
                " during class instantiation"
            )
        if len(cat_attrs) > 0:
            for plot_fn in ("make_violinplot",):
                getattr(self, plot_fn)(
                    cat_attrs,
                    self.metric_names,
                    metric_name_remap=metric_name_remap,
                    attribute_name_remap=attribute_name_remap,
                    save_dir=save_dir,
                    **plt_kwargs,
                )
        if len(num_attrs) > 0:
            for plot_fn in ("make_linearplot",):
                getattr(self, plot_fn)(
                    num_attrs,
                    self.metric_names,
                    metric_name_remap=metric_name_remap,
                    attribute_name_remap=attribute_name_remap,
                    save_dir=save_dir,
                    **plt_kwargs,
                )

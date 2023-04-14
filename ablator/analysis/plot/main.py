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
from ablator.main.configs import Optim

logger = logging.getLogger(__name__)


class PlotAnalysis(Analysis):
    @classmethod
    def _write_images(
        cls,
        fig_map: dict[str, ty.Union[Axes, Figure, Image.Image]],
        path: Path,
        file_format: ty.Literal["png", "pdf", "jpg"] = "png",
    ):
        path.mkdir(exist_ok=True, parents=True)
        for name, fig in fig_map.items():
            img_path = path.joinpath(f"{name}.{file_format}")
            if isinstance(fig, Axes):
                fig.figure.savefig(img_path)
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
        append=False,
        ax: Axes | None = None,
        metric_name_remap=None,
        attribute_name_remap=None,
        **kwargs,
    ):
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
            for ax in axes_map.values():
                plt.close(ax.figure)
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
    ):
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

    def make_figures(
        self,
        metric_name_remap: dict[str, str] | None = None,
        attribute_name_remap: dict[str, str] | None = None,
        **plt_kwargs,
    ):
        cat_attrs = list(self.categorical_attributes)
        num_attrs = list(self.numerical_attributes)
        if attribute_name_remap is not None:
            cat_attrs = list(set(attribute_name_remap.keys()).intersection(cat_attrs))
            num_attrs = list(set(attribute_name_remap.keys()).intersection(num_attrs))

        if len(cat_attrs) > 0:
            for plot_fn in ["make_violinplot"]:
                getattr(self, plot_fn)(
                    cat_attrs,
                    self.metric_names,
                    metric_name_remap=metric_name_remap,
                    attribute_name_remap=attribute_name_remap,
                    save_dir=self.save_dir,
                    **plt_kwargs,
                )
        if len(num_attrs) > 0:
            for plot_fn in ["make_linearplot"]:
                getattr(self, plot_fn)(
                    num_attrs,
                    self.metric_names,
                    metric_name_remap=metric_name_remap,
                    attribute_name_remap=attribute_name_remap,
                    save_dir=self.save_dir,
                    **plt_kwargs,
                )

import copy
import io
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Type, Union

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image

from trainer.analysis.main import Analysis
from trainer.analysis.plot import Plot
from trainer.analysis.plot.cat_plot import ViolinPlot
from trainer.analysis.plot.num_plot import LinearPlot
from trainer.config.run import Optim

logger = logging.getLogger(__name__)


class PlotAnalysis(Analysis):
    def __init__(self, *args, save_dir: Optional[str] = None, **kwargs) -> None:
        super().__init__(*args, save_dir=save_dir, **kwargs)

    @classmethod
    def _write_images(
        cls,
        fig_map: Dict[str, Union[Axes, Figure, Image.Image]],
        path: Path,
        file_format: Literal["png", "pdf", "jpg"] = "png",
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
        path: Optional[Path],
        plot_cls: Type[Plot],
        metrics: pd.DataFrame,
        results: pd.DataFrame,
        metric_map: Dict[str, Optim],
        append=False,
        ax: Optional[Axes] = None,
        metric_name_remap=None,
        attribute_name_remap=None,
        **kwargs,
    ):
        axes = {}

        (results, metrics, metric_map) = cls._remap_results(
            results, metrics, metric_map, metric_name_remap, attribute_name_remap
        )

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
                **kwargs,
            )
            if path is not None:
                p = path.joinpath(metric_name)
                cls._write_images(axes_map, p)
            [plt.close(ax.figure) for ax in axes_map.values()]
            axes[metric_name] = axes_map
        return axes

    @classmethod
    def _make_plot(
        cls,
        metric_values: pd.Series,
        results: pd.DataFrame,
        plot_cls: Type[Plot],
        metric_obj_fn: Optim,
        append=False,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Dict[str, Axes]:
        axes_map = {}
        for attribute_name in results.columns:
            attribute_values = results[attribute_name]
            figure, axes = plot_cls(
                metric=metric_values,
                attributes=attribute_values,
                metric_obj_fn=metric_obj_fn,
                y_axis=metric_values.name,
                x_axis=attribute_name,
                ax=ax,
            ).make(**kwargs)
            if append:
                ax = axes
            else:
                axes_map[attribute_name] = axes
                plt.close()
        if append:
            axes_map[attribute_name] = axes
            plt.close()
        return axes_map

    def make_violinplot(
        self,
        attribute_names: List[str],
        metrics: List[str],
        save_dir: Union[Path, str],
        **plt_kwargs,
    ):
        save_path = Path(save_dir).joinpath("violinplot")
        metric_map = {k: v for k, v in self.metric_map.items() if k in metrics}
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
        attribute_names: List[str],
        metrics: List[str],
        save_dir: Union[Path, str],
        **plt_kwargs,
    ):
        save_path = Path(save_dir).joinpath("linearplot")
        metric_map = {k: v for k, v in self.metric_map.items() if k in metrics}

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
        save_dir: Optional[str] = None,
        metric_name_remap: Optional[Dict[str, str]] = None,
        attribute_name_remap: Optional[Dict[str, str]] = None,
        **plt_kwargs,
    ):

        # TODO make remapping integral to PlotAnalysis
        report_dir = self._init_directories(save_dir)
        cat_attrs = set(self.categorical_attributes)
        num_attrs = set(self.numerical_attributes)
        if attribute_name_remap is not None:
            cat_attrs = set(attribute_name_remap.keys()).intersection(cat_attrs)
            num_attrs = set(attribute_name_remap.keys()).intersection(num_attrs)

        if len(cat_attrs):
            for plot_fn in ["make_violinplot"]:
                getattr(self, plot_fn)(
                    cat_attrs,
                    self.metric_names,
                    metric_name_remap=metric_name_remap,
                    attribute_name_remap=attribute_name_remap,
                    save_dir=report_dir,
                    **plt_kwargs,
                )
        if len(num_attrs):
            for plot_fn in ["make_linearplot"]:
                getattr(self, plot_fn)(
                    num_attrs,
                    self.metric_names,
                    metric_name_remap=metric_name_remap,
                    attribute_name_remap=attribute_name_remap,
                    save_dir=report_dir,
                    **plt_kwargs,
                )
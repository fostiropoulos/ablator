import logging
from pathlib import Path

import pandas as pd
from joblib import Memory

from ablator.analysis.plot.utils import parse_name_remap
from ablator.main.configs import Optim

logger = logging.getLogger(__name__)


class Analysis:
    def __init__(
        self,
        results: pd.DataFrame,
        categorical_attributes: list[str],
        numerical_attributes: list[str],
        optim_metrics: dict[str, Optim],
        save_dir: str | None = None,
        cache=False,
    ) -> None:
        self.optim_metrics = optim_metrics
        self.save_dir: Path | None = None
        self.cache: Memory | None = None
        if save_dir is not None:
            self.save_dir = Path(save_dir)
            if not self.save_dir.parent.exists():
                raise FileNotFoundError(
                    f"Save directory does not exist. `{self.save_dir.parent}`"
                )
            self.save_dir.mkdir(exist_ok=True)
            self.cache = Memory(Path(save_dir).joinpath(".cache"), verbose=0)
            if not cache:
                self.cache.clear()
                self.cache = None
        self.categorical_attributes: list[str] = categorical_attributes
        self.numerical_attributes: list[str] = numerical_attributes
        self.experiment_attributes: list[str] = (
            self.categorical_attributes + self.numerical_attributes
        )

        self.results: pd.DataFrame = results[
            self.experiment_attributes
            + list(self.optim_metrics.keys())
            + ["path", "index"]
        ]

    @property
    def metric_names(self):
        return list(self.optim_metrics.keys())

    @classmethod
    def _get_best_results_by_metric(
        cls,
        raw_results: pd.DataFrame,
        metric_map: dict[str, Optim],
    ):
        def _best_perf(row: pd.DataFrame, name, obj_fn):
            if Optim(obj_fn) == Optim.min:
                return row.sort_values(name, na_position="last").iloc[0]
            return row.sort_values(name, na_position="first").iloc[-1]

        _ress = []
        for name, obj_fn in metric_map.items():
            res = (
                raw_results.groupby("path")
                .apply(lambda x: _best_perf(x, name, obj_fn))
                .reset_index(drop=True)
            )
            res["best"] = name
            _ress.append(res)
        report_results = pd.concat(_ress).reset_index(drop=True)

        return report_results

    @classmethod
    def _remap_results(
        cls,
        attributes: pd.DataFrame,
        metrics: pd.DataFrame,
        metric_map: dict[str, Optim],
        metric_name_remap: dict[str, str] | None = None,
        attribute_name_remap: dict[str, str] | None = None,
    ):
        metric_name_remap = parse_name_remap(metrics.columns, metric_name_remap)
        attribute_name_remap = parse_name_remap(
            attributes.columns, attribute_name_remap
        )
        metric_map = {
            metric_name_remap[metric_name]: direction
            for metric_name, direction in metric_map.items()
            if metric_name in metric_name_remap
        }

        attributes = attributes[list(attribute_name_remap.keys())]
        metrics = metrics[list(metric_name_remap.keys())]
        attributes.columns = [attribute_name_remap[c] for c in attributes.columns]
        metrics.columns = [metric_name_remap[c] for c in metrics.columns]
        return attributes, metrics, metric_map

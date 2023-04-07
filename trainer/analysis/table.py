import copy
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import pandas as pd

from trainer.analysis.main import Analysis
from trainer.analysis.report_utils import (
    parse_cat_table,
    parse_numerical_table,
    table_to_format,
)
import typing as ty
from trainer.config.run import Optim

logger = logging.getLogger(__name__)


class TableAnalysis(Analysis):
    def make_tables(
        self,
        save_dir: Union[str, Path],
        metric_name_remap: Optional[Dict[str, str]] = None,
        attribute_name_remap: Optional[Dict[str, str]] = None,
        table_format: Literal["md", "latex"] = "latex",
        find_best: bool = False,
    ):
        report_dir = self._init_directories(save_dir)
        report_dir = report_dir.joinpath("tables")
        report_dir.mkdir(exist_ok=True)

        cat_attrs = set(self.categorical_attributes)
        num_attrs = set(self.numerical_attributes)
        metric_names = set(self.metric_names)
        if attribute_name_remap is not None:
            cat_attrs = set(attribute_name_remap.keys()).intersection(cat_attrs)
            num_attrs = set(attribute_name_remap.keys()).intersection(num_attrs)
        if metric_name_remap is not None:
            metric_names = set(metric_name_remap.keys()).intersection(metric_names)

        metric_map = {k: v for k, v in self.metric_map.items() if k in metric_names}

        metric_names = list(metric_names)
        num_attrs = list(num_attrs)
        cat_attrs = list(cat_attrs)
        metrics = self.results[metric_names]
        results = self.results[cat_attrs + num_attrs + ["path"]]
        (results, metrics, metric_map) = self._remap_results(
            results, metrics, metric_map, metric_name_remap, attribute_name_remap
        )

        # res = self._make_results_table(results, metrics, metric_map)

        # fn_name = "to_" + ("markdown" if format == "md" else "latex")
        # getattr(res, fn_name)(
        #     report_dir.joinpath(f"results.{file_extension}"), index=False
        # )
        if len(num_attrs):
            numerical_corr_table = self._make_corr_table(
                results[num_attrs],
                metrics=metrics,
                metric_map=metric_map,
                is_best=find_best,
                is_categorical=False,
                table_format=table_format,
                save_dir=save_dir,
            )

        if len(cat_attrs):
            cat_corr_table = self._make_corr_table(
                results[cat_attrs],
                metrics=metrics,
                metric_map=metric_map,
                is_best=find_best,
                is_categorical=True,
                table_format=table_format,
                save_dir=save_dir,
            )

    @classmethod
    def _make_results_table(
        self,
        results: pd.DataFrame,
        metrics: pd.DataFrame,
        metric_map: Dict[str, Optim],
        sort_key_idx: int = 0,
    ) -> str:
        def _get_best_from_metric(x: pd.DataFrame):
            res = []
            for metric, obj_fn in metric_map.items():
                if Optim(obj_fn) == Optim.min:
                    _row = x.sort_values(metric).iloc[:1]
                    _row["best"] = metric
                    res.append(_row.drop("index", axis=1))
                else:

                    _row = x.sort_values(metric).iloc[-1:]
                    _row["best"] = metric
                    res.append(_row.drop("index", axis=1))
            sort_key = list(metric_map.keys())[sort_key_idx]
            _df = pd.concat(res).sort_values(
                sort_key, ascending=metric_map[sort_key] == Optim.min
            )
            return _df

        return (
            pd.concat([results, metrics], axis=1)
            .reset_index()
            .groupby(["path"])
            .apply(_get_best_from_metric)
            .reset_index(drop=True)
        )

    @classmethod
    def _parse_numerical_corr(
        self,
        results: pd.DataFrame,
        corr_df: pd.DataFrame,
        attr: str,
        metric_name: str,
        metric_obj_fn,
    ):
        coef = corr_df.max() if metric_obj_fn == Optim.max else corr_df.min()
        # corr_df.iloc[corr_df.argmax()]

        return pd.DataFrame(
            [
                {
                    "attr": attr,
                    "metric": metric_name,
                    "coef": coef,
                }
            ]
        )

    @classmethod
    def _parse_categorical_corr(
        self,
        results: pd.DataFrame,
        corr_df: pd.DataFrame,
        attr: str,
        metric_name: str,
        metric_obj_fn: Optim,
    ):

        corr_df.index = list(map(lambda x: x.split(attr + "_")[-1], corr_df.index))

        best_config = corr_df.sort_values().iloc[:1].index[0]
        worst_config = corr_df.sort_values().iloc[-1:].index[0]

        if metric_obj_fn == Optim.max:
            _tmp = copy.deepcopy(best_config)
            best_config = copy.deepcopy(worst_config)
            worst_config = copy.deepcopy(_tmp)

        perf_mean = results.groupby(attr)[metric_name].mean()
        perf_mean.index.name = "attr_value"
        perf_mean.name = f"mean"
        perf = perf_mean.to_frame()
        perf["std"] = results.groupby(attr)[metric_name].std()
        perf.sort_values("mean", inplace=True)
        perf["corr"] = corr_df
        perf["attr"] = attr
        perf["metric"] = metric_name
        perf["best_setting"] = best_config
        perf["worst_setting"] = worst_config

        return perf.reset_index()

    @classmethod
    def _make_corr_table(
        cls,
        results: pd.DataFrame,
        metrics: pd.DataFrame,
        metric_map: Dict[str, Optim],
        is_best: bool,
        is_categorical: bool,
        save_dir: ty.Union[Path, str],
        table_format: Literal["md", "latex"] = "latex",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        """
        df: is a dataframe with columns the metrics we perform analysis on
        and index the control variables. Assumes it was read using `read_results`
        function where the configuration of the experiment is prefixed with `config.`
        and the run configuration is prefixed with `run_config.`

        metrics: increase or decrease each corresponding metric
        """
        # TODO
        # https://www.statsmodels.org/stable/examples/notebooks/generated/quantile_regression.html
        # https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance
        table = pd.concat([results, metrics], axis=1)
        save_dir = Path(save_dir)
        for attr in results.columns:
            rows = []

            for metric_name in metrics.columns:
                obj_fn = metric_map[metric_name]
                if is_best:
                    sub_results = table.loc[
                        table["best"] == metric_name, [attr, metric_name]
                    ]
                else:
                    sub_results = table[[attr, metric_name]]
                if is_categorical:
                    sub_results[attr] = sub_results[attr].astype(str)
                dummies = pd.get_dummies(sub_results)
                # TODO fit https://www.statsmodels.org/stable/examples/notebooks/generated/robust_models_0.html
                corr_map = dummies.corr()[metric_name].drop(metric_name, axis=0)
                res_corr = copy.deepcopy(corr_map)

                # if Optim(obj_fn) == Optim.min:
                #     # this metric needs to be minimized.
                #     res_corr *= -1

                if is_categorical:
                    fn = cls._parse_categorical_corr
                else:
                    assert (
                        pd.api.types.is_numeric_dtype(table[attr])
                        and corr_map.shape[0] == 1
                    )
                    fn = cls._parse_numerical_corr
                rows.append(fn(sub_results, res_corr, attr, metric_name, obj_fn))

            corr_table = pd.concat(rows)

            file_extension = "md" if table_format == "md" else "tex"
            if is_categorical:

                parsed_table = parse_cat_table(corr_table)
                save_path = save_dir.joinpath(f"num", f"{attr}.{file_extension}")
            else:
                parsed_table = parse_numerical_table(corr_table)
                save_path = save_dir.joinpath(f"cat", f"{attr}.{file_extension}")

            formated_table = table_to_format(
                parsed_table,
                is_categorical=is_categorical,
                table_format=table_format,
            )
            save_path.parent.mkdir(exist_ok=True, parents=True)
            save_path.write_text(formated_table)
        return table

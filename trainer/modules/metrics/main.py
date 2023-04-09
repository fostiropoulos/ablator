import math
from functools import cached_property
from typing import Any, Callable, Dict, Optional, Set

import numpy as np

import trainer.utils.base as butils
from trainer.modules.metrics.stores import ArrayStore, MovingAverage, PredictionStore


class LossDivergedError(Exception):
    pass


class TrainMetrics:
    """
    Stores and manages predictions and calculates metrics given some custom evaluation functions.
    Makes batch-updates
    Manages memory limits
    applies evaluation functions.
    provides cached or online updates on the train loss
    """

    def __init__(
        self,
        *args,
        batch_limit=30,
        memory_limit=1e8,
        evaluation_functions: Optional[Dict[str, Callable]] = None,
        moving_average_limit=3000,
        tags=["train"],
        # metrics with their initial value that are updated manually, i.e. learning rate
        static_aux_metrics: Optional[dict[str, Any]] = None,
        # metrics for which we update with their moving average, i.e. loss
        moving_aux_metrics: Optional[set[str]] = None,
    ):
        assert len(args) == 0, f"Metrics takes no positional arguments."

        _static_aux_metrics = {} if static_aux_metrics is None else static_aux_metrics
        _moving_aux_metrics = (
            set({}) if moving_aux_metrics is None else set(moving_aux_metrics)
        )

        _evaluation_functions = (
            {} if evaluation_functions is None else evaluation_functions
        )
        self.__batch_limit__ = batch_limit
        self.__memory_limit__ = memory_limit
        self.__moving_average_limit__ = moving_average_limit
        self.__evaluation_functions__ = _evaluation_functions
        self.__static_aux_attributes__: list[str] = sorted(
            list(set(_static_aux_metrics.keys()))
        )
        self.__moving_aux_attributes__: list[str] = sorted(
            list(set(_moving_aux_metrics))
        )
        self.__tags__: list[str] = sorted(list(set(tags)))
        self.__moving_eval_attributes__: list[str] = sorted(
            list(
                set(
                    {
                        f"{tag}_{eval_fn}"
                        for tag in self.__tags__
                        for eval_fn in self.__evaluation_functions__
                    }
                )
            )
        )
        _intersection = (
            set(self.__moving_aux_attributes__)
            .intersection(self.__moving_eval_attributes__)
            .union(
                set(self.__moving_aux_attributes__).intersection(
                    self.__static_aux_attributes__
                )
            )
        )
        assert _intersection == set(
            {}
        ), f"Overlapping metric names with built-ins {_intersection}"

        for tag, v in _static_aux_metrics.items():
            setattr(self, tag, v)
        for tag in set(self.__moving_aux_attributes__).union(
            self.__moving_eval_attributes__
        ):
            self._init_ma(tag)

        for tag in tags:
            self._init_preds(tag)

    def update_static_metrics(self, metric_dict: Dict[str, Any]):
        diff_metrics = set(metric_dict.keys()).difference(
            self.__static_aux_attributes__
        )
        metric_keys = sorted(list(metric_dict.keys()))
        assert (
            len(diff_metrics) == 0
        ), f"There are difference in the class metrics: {self.__static_aux_attributes__} and updated metrics {metric_keys}"
        metric_dict = butils.iter_to_numpy(metric_dict)
        for k, v in metric_dict.items():
            setattr(self, k, v)

    def update_ma_metrics(self, metric_dict: Dict[str, Any]):
        diff_metrics = set(metric_dict.keys()).difference(
            set(self.__moving_aux_attributes__)
        )
        metric_keys = sorted(list(metric_dict.keys()))
        assert (
            len(diff_metrics) == 0
        ), f"There are difference in the class metrics: {self.__moving_aux_attributes__} and updated metrics {metric_keys}"
        self._update_ma_metrics(metric_dict)

    def _update_ma_metrics(self, metric_dict: Dict[str, Any], tag=None):
        # metric dict should contain scalars
        metric_dict = butils.iter_to_numpy(metric_dict)
        for k, v in metric_dict.items():
            attr_name = f"{tag}_{k}" if tag is not None else k
            self._get_ma(attr_name).append(v)

    def evaluate(self, tag, reset=True, update_ma=True):
        preds = self._get_preds(tag)
        metrics = preds.evaluate()
        if update_ma:
            self._update_ma_metrics(metrics, tag)
        if reset:
            preds.reset()
        return metrics

    def append_batch(self, *args, tag, **kwargs):
        # NOTE this is because it is easy to mix up the order of pred, labels and tags
        assert len(args) == 0, "Metrics.append_batch takes no positional arguments."
        assert (
            tag in self.__tags__
        ), f"Undefined tag '{tag}'. Metric tags {self.__tags__}"
        self._get_preds(tag).append(**kwargs)

    def _init_preds(self, tag) -> PredictionStore:
        attr_name = f"__{tag}_preds__"
        _preds = PredictionStore(
            batch_limit=self.__batch_limit__,
            memory_limit=self.__memory_limit__,
            evaluation_functions=self.__evaluation_functions__,
        )
        setattr(self, attr_name, _preds)
        return getattr(self, attr_name)

    def _get_preds(self, tag) -> PredictionStore:
        attr_name = f"__{tag}_preds__"
        preds = getattr(self, attr_name)
        return preds

    def _init_ma(self, tag) -> MovingAverage:
        attr_name = f"__{tag}_ma__"
        _ma = MovingAverage(
            batch_limit=self.__moving_average_limit__,
            memory_limit=self.__memory_limit__,
        )
        setattr(self, attr_name, _ma)
        return getattr(self, attr_name)

    def _get_ma(self, tag) -> MovingAverage:
        attr_name = f"__{tag}_ma__"
        preds = getattr(self, attr_name)
        return preds

    def to_dict(self):
        attrs = self.__moving_aux_attributes__ + self.__moving_eval_attributes__
        ma_attrs = {k: self._get_ma(k).value for k in attrs}
        static_attrs = {k: getattr(self, k) for k in self.__static_aux_attributes__}
        return {**ma_attrs, **static_attrs}

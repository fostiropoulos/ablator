import math
import sys
from collections import OrderedDict
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch

import trainer.utils.train as tutils
from trainer.config.run import Optim
from trainer.utils.train import iter_to_numpy


class LossDivergedError(Exception):
    pass


class ArrayMetrics(Sequence):
    def __init__(
        self,
        batch_limit: int = 30,
        # 100 MB memory limit
        memory_limit: int = int(1e8),
    ):
        super().__init__()
        self.arr: List[Any] = []
        self.limit = batch_limit
        self.memory_limit = memory_limit

    def append(self, vals: List):
        self.arr.append(iter_to_numpy(vals))
        if len(self.arr) > self.limit:
            self.arr = self.arr[-self.limit :]
        elif sys.getsizeof(self.arr) > self.memory_limit:
            self.limit = len(self.arr) - 1

    def get(self):
        if len(self.arr) > 0:
            return np.concatenate(self.arr)
        return np.array(self.arr)

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        return self.arr[i]

    def reset(self):
        self.vals = []


class PredMetrics:
    def __init__(
        self,
        batch_limit: int = 30,
        # 100 MB memory limit
        memory_limit: int = int(1e8),
        moving_average_limit: int = 3000,
    ):
        super().__init__()
        self.labels = ArrayMetrics(batch_limit=batch_limit, memory_limit=memory_limit)
        self.preds = ArrayMetrics(batch_limit=batch_limit, memory_limit=memory_limit)
        self.limit = batch_limit
        self.memory_limit = memory_limit
        self.loss_avg = MovingAverage(moving_average_limit)

    @property
    def loss(self):
        return self.loss_avg.value

    def append(self, labels: Optional[List], preds: Optional[List], loss=None):
        if loss is not None:
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            self.loss_avg.append(loss)
        if labels is None or preds is None:
            return
        self.labels.append(labels)
        self.preds.append(preds)
        new_limit = min(self.preds.limit, self.labels.limit)
        self.preds.limit = new_limit
        self.labels.limit = new_limit

    def eval_metrics(self, metric_map: Optional[Dict[str, Callable]] = None):
        preds, labels = self.preds.get(), self.labels.get()
        if isinstance(metric_map, dict) and len(preds) > 0 and len(labels) > 0:
            return {k: v(labels, preds) for k, v in metric_map.items()}
        else:
            return {}

    def reset(self):
        self.labels.reset()
        self.preds.reset()
        self.loss_avg.reset()


class MovingAverage(ArrayMetrics):
    """
    Moving average
    """

    @property
    def __mean__(self):
        return np.mean(self.arr)

    @property
    def value(self):
        if len(self.arr) > 0:
            return self.__mean__
        else:
            return None

    def __lt__(self, __o: float) -> bool:
        return float(self.value).__lt__(__o)

    def __eq__(self, __o: object) -> bool:
        return float(self.value).__eq__(__o)

    def __float__(self):
        return float(self.value)

    def __format__(self, format_spec=".2e"):
        return format(self.value, format_spec)

    def __repr__(self) -> str:
        return f"{self.value:.2e}"


class Metrics:
    # TODO evaluate slow-down of metrics during training
    epochs: int = 0
    current_iteration: int = 0
    val_loss: Optional[float] = None
    best_loss: float = float("inf")
    best_iteration: float = 0
    # current_loss: float = float("inf")
    total_steps: float = float("inf")
    lr: float = float("inf")

    def __init__(
        self,
        *args,
        batch_limit=30,
        memory_limit=1e8,
        evaluation_functions: Optional[Dict[str, Callable]] = None,
        moving_average_limit=3000,
        online_update=True,
        **kwargs,
    ):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__batch_limit__ = batch_limit
        self.__memory_limit__ = memory_limit
        self.__moving_average_limit__ = moving_average_limit
        self.__evaluation_functions__ = evaluation_functions
        self.__aux_attributes__: Set[str] = set([])
        self.__online_update__ = online_update
        self.init_preds("train", batch_limit, memory_limit, moving_average_limit)

    @cached_property
    def current_loss(self):
        return self.get_preds("train").loss_avg.value

    @property
    def current_epoch(self):
        if self.current_iteration > 0:
            return math.floor(self.current_iteration / self.total_steps * self.epochs)
        else:
            return 0

    def update(self, metric_dict: Dict[str, Any], prefix: Optional[str] = None):
        for k, v in metric_dict.items():
            attr_name = f"{prefix}_{k}" if prefix is not None else k
            self.__aux_attributes__.add(attr_name)
            setattr(self, attr_name, v)

    def reset_train_preds(self):
        self.get_preds("train").reset()

    def eval_train(self):
        if "current_loss" in self.__dict__:
            del self.__dict__["current_loss"]
        self.eval_metrics("train")

    def eval_best(self, div_factor, div_warm_up_steps):

        is_best = False
        # Use val loss for scheduling or finding best checkpoint
        val_loss = self.val_loss

        is_best = val_loss < self.best_loss

        if is_best or self.best_loss == 0:
            self.best_iteration = self.current_iteration
            self.best_loss = val_loss

        divergence_step = self.current_iteration > self.total_steps * div_warm_up_steps
        is_diverged = val_loss / self.best_loss > div_factor

        if is_diverged and divergence_step:
            raise LossDivergedError(
                f"Val loss {val_loss:.4e} has diverged by a factor of {div_factor} to best loss {self.best_loss:.4e}"
            )
        return is_best

    def eval_metrics(self, tag):
        preds = self.get_preds(tag)
        metrics = preds.eval_metrics(self.__evaluation_functions__)
        metrics[f"loss"] = preds.loss_avg.value
        self.update(metrics, prefix=tag)
        # Update best_loss etc
        if tag != "train":
            preds.reset()

    def append_train(self, pred, labels, loss=None, aux_metrics=None):

        self.append(pred, labels, tag="train", loss=loss, aux_metrics=aux_metrics)
        if self.__online_update__ and "current_loss" in self.__dict__:
            del self.__dict__["current_loss"]

    def append_val(self, pred, labels, loss=None, aux_metrics=None):
        self.append(pred, labels, loss=loss, aux_metrics=aux_metrics, tag="val")

    def append(self, pred, labels, tag, loss=None, aux_metrics=None):
        self.get_preds(tag).append(labels=labels, preds=pred, loss=loss)
        if aux_metrics is not None:
            aux_metrics = tutils.iter_to_numpy(aux_metrics)
            self.update(aux_metrics, prefix=f"{tag}_aux")

    def init_preds(
        self, tag, batch_limit=None, memory_limit=None, moving_average_limit=None
    ):
        attr_name = f"__{tag}_preds__"
        _preds = PredMetrics(
            batch_limit=self.__batch_limit__ if batch_limit is None else batch_limit,
            memory_limit=self.__memory_limit__
            if memory_limit is None
            else memory_limit,
            moving_average_limit=self.__moving_average_limit__
            if moving_average_limit is None
            else moving_average_limit,
        )

        setattr(self, attr_name, _preds)
        return getattr(self, attr_name)

    def init_arr(self, tag, batch_limit=None, memory_limit=None):
        attr_name = f"__{tag}_arr__"
        _arr = ArrayMetrics(
            batch_limit=self.__batch_limit__ if batch_limit is None else batch_limit,
            memory_limit=self.__memory_limit__
            if memory_limit is None
            else memory_limit,
        )

        setattr(self, attr_name, _arr)
        return getattr(self, attr_name)

    def get_arr(self, tag) -> ArrayMetrics:
        attr_name = f"__{tag}_arr__"
        arr = getattr(self, attr_name, None)
        if arr is None:
            return self.init_arr(tag)
        return arr
    # TODO get all arrs and preds tags
    def get_preds(self, tag) -> PredMetrics:
        attr_name = f"__{tag}_preds__"
        preds = getattr(self, attr_name, None)
        if preds is None:
            return self.init_preds(tag)
        return preds

    def to_dict(self):
        return {k: getattr(self, k) for k in self.attributes()}

    def _base_attributes(self):
        return list(self.__annotations__.keys())

    def _all_attributes(self):
        return [
            attr
            for attr in list(self.__dict__.keys())
            if not (attr.startswith("__") and attr.endswith("__"))
        ]

    def _added_attributes(self):
        return list(set(self._all_attributes()).difference(self._base_attributes()))

    def get_added_metrics(self, tag=None):
        if tag is not None:
            attrs = [attr for attr in self.__aux_attributes__ if attr.startswith(tag)]
        else:
            attrs = self.__aux_attributes__
        return {k: getattr(self, k) for k in attrs}

    def get_base_metrics(self):
        return_dict = {}
        metric_fn_pairs = [
            ("current_loss", lambda x: x is not None),
            ("lr", np.isfinite),
            ("best_loss", np.isfinite),
            ("best_iteration", lambda x: x > 0),
        ]
        for k, mask_fn in metric_fn_pairs:
            v = getattr(self, k)
            if mask_fn(v):
                return_dict[k] = v
        return return_dict

    def attributes(self):
        model_class_attr = self._all_attributes() + self._base_attributes()
        return list(np.unique(model_class_attr))

    def get_msg_preds(self, tag: str):
        aux_metrics = self.get_added_metrics(tag)
        return self._make_msg(aux_metrics)

    @classmethod
    def _make_msg(cls, metrics: Dict[str, Any]):
        return " / ".join(
            [
                f"{k}: {v:.2e}"
                # else f"{k}: {v}"
                for k, v in metrics.items()
                if isinstance(v, (MovingAverage, np.number, int, float))
            ]
        )

    def get_msg(self):

        base_metrics = self.get_base_metrics()
        aux_metrics = self.get_added_metrics()
        base_msg = self._make_msg(base_metrics)
        aux_msg = self._make_msg(aux_metrics)

        msg = (
            f"[{self.current_iteration}/{self.total_steps} - ep: {self.current_epoch}/{self.epochs}] "
            + f"{base_msg} {aux_msg} "
        )
        return msg

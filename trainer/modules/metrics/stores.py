import sys
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, TypeVar
import numpy as np
import torch
import typing as ty
import trainer.utils.base as butils
import inspect


class ArrayStore(Sequence):
    def __init__(
        self,
        batch_limit: int = 30,
        # 100 MB memory limit
        memory_limit: Optional[int] = int(1e8),
    ):
        super().__init__()
        self.arr: List[ty.Union[np.ndarray, int, float]] = []
        self.limit = batch_limit
        self.memory_limit = memory_limit

    def append(self, val: ty.Union[np.ndarray, float, int]):
        """Appends a batch of values"""
        # Appending by batch is faster than converting numpy to list
        assert isinstance(
            val, (np.ndarray, int, float)
        ), f"Invalid ArrayStore value type {type(val)}"
        self.arr.append(val)
        if len(self.arr) > self.limit:
            self.arr = self.arr[-self.limit :]
        elif (
            self.memory_limit is not None
            and sys.getsizeof(self.arr) > self.memory_limit
        ):
            self.limit = len(self.arr) - 1

    def get(self) -> np.ndarray:
        """Returns a flatten array of values"""
        if len(self.arr) > 0:
            self.arr = [np.concatenate(self.arr)]
        return np.array(self.arr)

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        return self.arr[i]

    def reset(self):
        self.vals = []


class PredictionStore:
    def __init__(
        self,
        batch_limit: int = 30,
        # 100 MB memory limit
        memory_limit: int = int(1e8),
        moving_average_limit: int = 3000,
        evaluation_functions: Optional[Dict[str, Callable]] = None,
    ):
        super().__init__()
        # self.labels = ArrayStore(batch_limit=batch_limit, memory_limit=memory_limit)
        # self.preds = ArrayStore(batch_limit=batch_limit, memory_limit=memory_limit)
        self.limit = batch_limit
        self.memory_limit = memory_limit
        self.metrics: Dict[str, MovingAverage] = (
            {k: MovingAverage(moving_average_limit) for k in evaluation_functions}
            if evaluation_functions is not None
            else {}
        )
        self.__evaluation_functions__ = evaluation_functions
        self._keys: Optional[list[str]] = None

    def _init_arr(self, tag):
        attr_name = f"__{tag}_arr__"
        _arr = ArrayStore(batch_limit=self.limit, memory_limit=self.memory_limit)
        setattr(self, attr_name, _arr)
        return getattr(self, attr_name)

    def _get_arr(self, tag) -> ArrayStore:
        attr_name = f"__{tag}_arr__"
        arr = getattr(self, attr_name)
        return arr

    def append(self, **batches: dict[str, np.ndarray]):
        if self._keys is None:
            for k in batches:
                self._init_arr(k)
            self._keys = sorted(list(batches.keys()))
        sizes = {}
        limits = []
        assert self._keys == sorted(
            list(batches.keys())
        ), f"Missing keys from the prediction store update. Expected: {self._keys}, received {list(batches.keys())}"
        for k, v in batches.items():
            np_arr = butils.iter_to_numpy(v)
            sizes[k] = len(np_arr)
            self._get_arr(k).append(np_arr)
            limits.append(self._get_arr(k).limit)
        assert (
            len(set(sizes.values())) == 1
        ), f"Different number of batches between inputs. Sizes: {sizes}"

        new_limit = min(limits)
        for k in self._keys:
            self._get_arr(k).limit = new_limit

    def evaluate(self) -> Dict[str, float]:
        if self._keys is None:
            raise RuntimeError("PredictionStore has no predictions to evaluate.")
        batches = {k: self._get_arr(k).get() for k in self._keys}

        if self.__evaluation_functions__ is None or len(batches) == 0:
            return {}
        metrics = {}
        for k, v in self.__evaluation_functions__.items():
            fn_args = sorted(list(inspect.getargspec(v)[0]))

            assert self._keys == fn_args, f"Evaluation function arguments {fn_args} different than stored predictions: {self._keys}"
            metric = v(**batches)
            if isinstance(metric, torch.Tensor):
                metric = metric.item()
            metrics[k] = metric
            try:
                self.metrics[k].append(metric)
            except ValueError:
                raise ValueError(
                    f"Invalid value {metric} returned by evaluation function {v.__name__}. Must be numeric scalar."
                )
        return metrics

    def reset(self):
        for k in self._keys:
            self._get_arr(k).reset()


class MovingAverage(ArrayStore):
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

    def append(self, vals: ty.Union[np.ndarray, torch.Tensor, float, int]):
        if not isinstance(vals, (np.ndarray, torch.Tensor, int, float)):
            raise ValueError(f"Invalid MovingAverage value type {type(vals)}")
        if isinstance(vals, (np.ndarray, torch.Tensor)):
            npval = butils.iter_to_numpy(vals)
            try:
                scalar = npval.item()
            except:
                raise ValueError(f"MovingAverage value must be scalar. {vals}")
        else:
            scalar = vals
        super().append(scalar)

import inspect
import sys
import typing as ty
from collections.abc import (
    Callable,
    Sequence,
)

import numpy as np
import torch

import ablator.utils.base as butils


class ArrayStore(Sequence):
    """
    Base class for manipulations (storing, getting, resetting) of batches of values.

    """

    def __init__(
        self,
        batch_limit: int = 30,
        # 100 MB memory limit
        memory_limit: int | None = int(1e8),
    ):
        """
        Initialize the storage settings.

        Parameters
        ----------
        batch_limit : int, optional
            The maximum number of batches of values to store for this single store. Default is 30.
        memory_limit : int or None, optional
            The maximum memory allowed for all values in bytes. Default is 1e8.

        Examples
        --------
        >>> from ablator.modules.metrics.stores import ArrayStore
        >>> train_metrics = ArrayStore(
        ...     batch_limit=50,
        ...     memory_limit=1000
        ... )
        """
        super().__init__()
        self.arr: list[np.ndarray | int | float] = []
        self.limit = batch_limit
        self.memory_limit = memory_limit

    def append(self, val: np.ndarray | float | int):
        """
        Appends a batch of values, or a single value, constrained on the limits.
        If after appending a new batch, ``batch_limit`` is exceeded, only ``batch_limit`` number
        of latest batches is kept. If memory limit is exceeded, ``batch_limit`` will be reduced.

        Parameters
        ----------
        val : np.ndarray or float or int
            The data, can be a batch of data, or a scalar.

        Raises
        ------
        AssertionError:
            If appended value is not numpy array, an integer, or a float number.

        Examples
        --------

        The following example shows a case where batch limit is exceeded
        (100 values/batches to be appended while only 10 is allowed)

        >>> from ablator.modules.metrics.stores import ArrayStore
        >>> array_store = ArrayStore(
        ...     batch_limit=10,
        ...     memory_limit=1000
        ... )
        >>> for i in range(100):
        >>>     array_store.append(int(i))
        >>> array_store.arr
        [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        >>> array_store.limit
        10

        This example shows a case where memory limit is exceeded. As soon as the 5th
        value is appended, memory of the list is 104 > 100), so ``batch_limit`` is set
        to the length of the store so far (which is 5) reduced by 1, which equals to 4.
        Therefore, from then on, only 4 values/batches is allowed.

        >>> array_store = ArrayStore(
        ...     batch_limit=10,
        ...     memory_limit=100
        ... )
        >>> for i in range(100):
        >>>     array_store.append(int(i))
        >>> array_store.arr
        [96, 97, 98, 99]
        >>> array_store.limit
        4
        """
        """Appends a batch of values"""
        # Appending by batch is faster than converting numpy to list
        assert isinstance(
            val, (np.ndarray, int, float)
        ), f"Invalid ArrayStore value type {type(val)}"
        self.arr.append(val)
        if len(self.arr) > self.limit:
            self.arr = self.arr[-self.limit:]
        elif (
            self.memory_limit is not None
            and sys.getsizeof(self.arr) > self.memory_limit
        ):
            self.limit = len(self.arr) - 1

    def get(self) -> np.ndarray:
        """
        Returns a flatten array of values

        Examples
        --------
        >>> from ablator.modules.metrics.stores import ArrayStore
        >>> array_store = ArrayStore(
        ...     batch_limit=10,
        ...     memory_limit=1000
        ... )
        >>> for i in range(100):
        >>>     array_store.append(np.array([int(i)]))
        >>> array_store.get()
        [[90 91 92 93 94 95 96 97 98 99]]
        """
        if len(self.arr) > 0:
            self.arr = [np.concatenate(self.arr)]
        return np.array(self.arr)

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        return self.arr[i]

    def reset(self):
        """
        Reset list of values to empty.

        Examples
        --------
        >>> from ablator.modules.metrics.stores import ArrayStore
        >>> array_store = ArrayStore(
        ...     batch_limit=10,
        ...     memory_limit=1000
        ... )
        >>> for i in range(100):
        >>>     array_store.append(int(i))
        >>> array_store.arr
        [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        >>> array_store.reset()
        >>> array_store.arr
        []
        """
        self.arr = []


class PredictionStore:
    """
    A class for storing prediction scores. This allows for evaluating prediction results using evaluation functions

    """

    def __init__(
        self,
        batch_limit: int = 30,
        # 100 MB memory limit
        memory_limit: int = int(1e8),
        moving_average_limit: int = 3000,
        evaluation_functions: dict[str, Callable] | None = None,
    ):
        """
        Initialize the storage settings.

        Parameters
        ----------
        batch_limit : int, optional
            Maximum number of batches to keep for each array store corresponding to each category of prediction
            outputs (e.g preds, labels), so only ``batch_limit`` number of latest batches is stored per set of
            array store. Default is 30.
        memory_limit : int or None, optional
            Maximum memory (in bytes) of batches to keep for each array store corresponding to each category of
            prediction outputs (e.g preds, labels). Default is 1e8.
        moving_average_limit : int, optional
            The maximum number of values allowed to store moving average metrics. Default is 3000.
        evaluation_functions : dict[str, Callable], optional
            A dictionary of key-value pairs, keys are evaluation function names, values are
            callable evaluation functions, e.g mean, sum. Note that arguments to this Callable
            must match with names of prediction batches that the model returns. So if model prediction over
            a batch looks like this: ``{"preds": <batch of predictions>, "labels": <batch of predicted labels>}``,
            then callable's arguments should be ``preds`` and ``labels``, e.g ``evaluation_functions=
            {"mean": lambda preds, labels: np.mean(preads) + np.mean(labels)}``. Default is None.

        Examples
        --------
        >>> from ablator.modules.metrics.stores import PredictionStore
        >>> pred_store = PredictionStore(
        ...     batch_limit=10,
        ...     memory_limit=1000,
        ...     moving_average_limit=1000,
        ...     evaluation_functions={"mean": lambda x: np.mean(x)}
        ... )
        """
        super().__init__()
        self.limit = batch_limit
        self.memory_limit = memory_limit
        self.metrics: dict[str, MovingAverage] = (
            {k: MovingAverage(moving_average_limit) for k in evaluation_functions}
            if evaluation_functions is not None
            else {}
        )
        self.__evaluation_functions__ = evaluation_functions
        self._keys: list[str] | None = None

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
        """
        Appends batches of values, constrained on the limits.

        Parameters
        ----------
        tag : str
            A tag that specifies which set of predictions to evaluate.
        **batches : dict[str, np.ndarray]
            A dictionary of key-value pairs, where key is type of prediction (e.g predictions, labels),
            and value is a batch of prediction values. Note that the passed keys in ``**batches`` must match arguments
            in evaluation functions arguments in the Callable in `evaluation_functions`
            when we initialize `PredictionStore` object.

        Raises
        ------
        AssertionError
            If passed keys do not match arguments in evaluation functions,
            or when batches among the keys are different in size.

        Examples
        --------
        >>> from ablator.modules.metrics.stores import PredictionStore
        >>> pred_store = PredictionStore(
        ...     batch_limit=10,
        ...     memory_limit=1000,
        ...     moving_average_limit=1000,
        ...     evaluation_functions={"mean": lambda preds, labels: np.mean(preds) + np.mean(labels)}
        ... )
        >>> pred_store.append(preds=np.array([4,3,0]), labels=np.array([5,1,1]))

        """
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

    def evaluate(self) -> dict[str, float]:
        """
        Apply evaluation_functions to predictions sets, e.g preds, labels.

        Returns
        -------
        metrics : dict
            A dictionary of metric values calculated from different sets of predictions.


        Raises
        ------
        AssertionError
            If passed keys do not match arguments in evaluation functions.

        ValueError
            If evaluation result is not a numeric scalar.

        Examples
        --------
        >>> from ablator.modules.metrics.main import PredictionStore
        >>> pred_store = PredictionStore(
        ...     batch_limit=30,
        ...     evaluation_functions={"mean": lambda preds, labels: np.mean(preds) + np.mean(labels)
        ...     moving_average_limit=100
        ... )
        >>> pred_store.append(preds=np.array([4,3,0]), labels=np.array([5,1,3]))
        >>> pred_store.evaluate()
        {'mean': 5.333333333333334}
        """
        if self._keys is None:
            return {}
        batches = {k: self._get_arr(k).get() for k in self._keys}

        if self.__evaluation_functions__ is None or len(batches) == 0:
            return {}
        metrics = {}
        for k, v in self.__evaluation_functions__.items():
            fn_args = sorted(list(inspect.getfullargspec(v)[0]))

            assert (
                self._keys == fn_args
            ), f"Evaluation function arguments {fn_args} different than stored predictions: {self._keys}"
            metric = v(**batches)
            if isinstance(metric, torch.Tensor):
                metric = metric.item()
            metrics[k] = metric
            try:
                self.metrics[k].append(metric)
            except Exception as exc:
                raise ValueError(
                    f"Invalid value {metric} returned by evaluation function {v.__name__}. Must be numeric scalar."
                ) from exc
        return metrics

    def reset(self):
        """
        Reset to empty all prediction sequences (e.g predictions, labels).

        Examples
        --------
        >>> from ablator.modules.metrics.main import PredictionStore
        >>> pred_store = PredictionStore(
        ...     batch_limit=30,
        ...     memory_limit=None,
        ...     evaluation_functions={"sum": lambda pred: np.mean(pred)},
        ...     moving_average_limit=100
        ... )
        >>> pred_store.append(preds=np.array([4,3,0]), labels=np.array([5,1,3]))
        >>> pred_store.reset()
        """
        if self._keys is None:
            return
        for k in self._keys:
            self._get_arr(k).reset()


class MovingAverage(ArrayStore):
    """
    This class is used to store moving average metrics

    """

    @property
    def __mean__(self):
        return np.mean(self.arr)

    @property
    def value(self):
        if len(self.arr) > 0:
            return self.__mean__
        return np.nan

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

    def append(self, val: ty.Union[np.ndarray, torch.Tensor, float, int]):
        """
        Appends a batch of values, or a single value, constrained on the limits.

        Parameters
        ----------
        val : ty.Union[np.ndarray, torch.Tensor, float, int]
            The data to be appended

        Raises
        ------
        ValueError:
            If appended value is of required type, or if val is not a scalar.

        Examples
        --------
        >>> from ablator.modules.metrics.stores import MovingAverage
        >>> ma_store = MovingAverage()
        >>> for i in range(100):
        >>>     ma_store.append(np.array([int(i)]))
        >>> ma_store.arr
        [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
        86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        """
        if not isinstance(val, (np.ndarray, torch.Tensor, int, float)):
            raise ValueError(f"Invalid MovingAverage value type {type(val)}")
        if isinstance(val, (np.ndarray, torch.Tensor)):
            npval = butils.iter_to_numpy(val)
            try:
                scalar = npval.item()
            except Exception as exc:
                raise ValueError(f"MovingAverage value must be scalar. {val}") from exc
        else:
            scalar = val
        super().append(scalar)

import inspect

import typing as ty
from collections.abc import Callable, Sequence
from functools import cached_property
import bisect
import numpy as np
import torch

import ablator.utils.base as butils


def _parse_moving_average_val(
    val: np.ndarray | torch.Tensor | int | float,
) -> int | float:
    if not isinstance(val, (np.ndarray, torch.Tensor, int, float)):
        raise ValueError(f"Invalid MovingAverage value type {type(val)}")
    if isinstance(val, (np.ndarray, torch.Tensor)):
        npval: np.ndarray = butils.iter_to_numpy(val)
        try:
            scalar = npval.item()
        except Exception as exc:
            raise ValueError(f"MovingAverage value must be scalar. Got {val}") from exc
    else:
        scalar = val
    return scalar


def _parse_array_store_val(
    val: np.ndarray | int | float,
    store_type: None | type = None,
    shape: None | tuple[int, ...] = None,
):
    if not isinstance(val, (np.ndarray, int, float)):
        raise RuntimeError(f"Invalid ArrayStore value type {type(val)}")
    np_val: np.ndarray
    if isinstance(val, (int, float)) or len(val.shape) == 0:
        np_val = np.array([[val]])
    else:
        np_val = val
    if len(np_val.shape) < 2:
        raise ValueError(
            (
                "Missing batch dimension. If supplying a single value array, "
                "reshape to [B, 1] or if suppling a single a batch reshape to [1, N]."
            )
        )
    if store_type is not None and np_val.dtype != store_type:
        raise RuntimeError(
            f"Inhomogeneous types between stored values {store_type} and provided value {np_val.dtype}."
        )
    # skipping batch dim
    data_shape = np_val.shape[1:]
    if shape is not None and data_shape != shape:
        raise RuntimeError(
            f"Inhomogeneous shapes between stored values  {shape} and provided value {data_shape}"
        )
    return np_val


class ArrayStore(Sequence):
    """
    Base class for manipulations (storing, getting, resetting) of batches of values.

    """

    def __init__(
        self,
        batch_limit: int | None = None,
        # 100 MB memory limit
        memory_limit: int | None = None,
    ):
        """
        Initialize the storage based on the memory / batch_limits provided.

        Parameters
        ----------
        batch_limit : int, optional
            The maximum number of batches of values to store for this single store.
            If set to None, unlimited number of batches will be stored. Default is None.
        memory_limit : int or None, optional
            The maximum memory allowed for the prediction store.
            If set to None, unlimited number of batches will be stored. Default is None.

        Examples
        --------
        >>> from ablator.modules.metrics.stores import ArrayStore
        >>> train_metrics = ArrayStore(
        ...     batch_limit=50,
        ...     memory_limit=1000
        ... )
        """
        super().__init__()
        self._arr: list[np.ndarray] = []
        self.limit = int(batch_limit) if batch_limit is not None else None
        self._memory_limit = memory_limit
        self._arr_len: list[int] = []
        self._len = 0
        # initialize memory_size as it is a cached property.
        getattr(self, "memory_size")

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
        # Appending by batch is faster than converting numpy to list
        np_val = _parse_array_store_val(
            val, shape=self.shape, store_type=self.store_type
        )
        self._arr.append(np_val)
        if self.store_type is None:
            # we reset the cached properties
            del self.store_type
            del self.shape
            del self.memory_size
        np_val_len = len(np_val)
        self._arr_len.append(np_val_len)
        self._len += np_val_len
        if (
            self._memory_limit is not None
            and self.memory_size * (self._len + 1) > self._memory_limit
        ):
            memory_limit = int(max(self._memory_limit // self.memory_size, 1))
            limit: int = (
                min(memory_limit, self.limit)
                if self.limit is not None
                else memory_limit
            )
            self.limit = limit
            self._prune(limit)
        elif self.limit is not None and self._len > self.limit:
            self._prune(self.limit)

    @cached_property
    def memory_size(self):
        arr_size = 0
        if len(self) > 0:
            arr = self._arr[0]
            arr_size = arr.size * arr.itemsize

        return arr_size

    def _prune(self, limit: int):
        limit = int(limit)
        lens = np.cumsum(self._arr_len[::-1])
        idx = np.argmax(lens > limit)
        underflow = lens[idx - 1] - limit
        if underflow > 0:
            # overflow!
            underflow = -limit
        if underflow < 0:
            idx += 1
            self._arr[-idx] = self._arr[-idx][underflow:]
            self._arr_len[-idx] = -1 * underflow

        self._arr = self._arr[-idx:]
        self._arr_len = self._arr_len[-idx:]
        self._len = limit
        assert sum(self._arr_len) == limit

    def get(self) -> np.ndarray:
        """
        Returns the stored values as a numpy array.

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
        array([[90], [91], [92], [93], [94], [95], [96], [97], [98], [99]])
        """
        if len(self._arr) == 0:
            return np.array([[]])
        return np.concatenate(self._arr)

    @cached_property
    def store_type(self) -> type | None:
        if len(self._arr) > 0:
            return self._arr[-1].dtype
        return None

    @cached_property
    def shape(self) -> tuple[int, ...] | None:
        if len(self._arr) > 0:
            return self._arr[-1].shape[1:]
        return None

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        idx = self._len + idx if idx < 0 else idx

        if idx < 0 or idx >= self._len:
            raise IndexError("list index out of range")

        cum_sum = np.cumsum(self._arr_len)

        if (arr_idx := bisect.bisect_right(cum_sum, idx)) > 0:
            idx -= cum_sum[arr_idx - 1]
        return self._arr[arr_idx][idx]

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
        self._arr = []
        self._arr_len = []
        self._len = 0


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
        evaluation_functions: dict[str, Callable] | list[Callable] | None = None,
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
        # not all arguments from evaluation functions are required.
        # we infer the overlapping arguments at run-time and store
        # as self._keys, a dictionary with keys the function names,
        # values is a list of infered arguments.
        self._fn_keys: dict[str, list[str]] | None = None
        self._batch_keys: set[str] | None = None
        self.metrics: dict[str, MovingAverage] = {}
        self.__evaluation_functions__ = None
        if isinstance(evaluation_functions, list):
            evaluation_functions_dict = {v.__name__: v for v in evaluation_functions}
        elif isinstance(evaluation_functions, dict):
            evaluation_functions_dict = evaluation_functions
        elif evaluation_functions is None:
            return
        else:
            raise NotImplementedError(
                f"Unrecognized evaluation functions {evaluation_functions}"
            )
        self.metrics = {
            k: MovingAverage(moving_average_limit) for k in evaluation_functions_dict
        }
        self.__evaluation_functions__ = evaluation_functions_dict

    def _init_arr(self, tag):
        attr_name = f"__{tag}_arr__"
        _arr = ArrayStore(batch_limit=self.limit, memory_limit=self.memory_limit)
        setattr(self, attr_name, _arr)
        return getattr(self, attr_name)

    def _get_arr(self, tag) -> ArrayStore:
        attr_name = f"__{tag}_arr__"
        arr = getattr(self, attr_name)
        return arr

    def _init_keys(self, batch_keys: set[str]):
        missing_keys: list[str] = []
        self._batch_keys = batch_keys
        self._fn_keys = {}
        for k, v in self.evaluation_function_arguments.items():
            if len(set(batch_keys).intersection(v)) == 0:
                missing_keys.append(f"{k}: {v}")
            else:
                self._fn_keys[k] = list(batch_keys.intersection(v))
        if len(missing_keys) > 0:
            err_msg = "\n".join(missing_keys)
            raise ValueError(
                f"Batch keys do not match any function arguments: {err_msg}"
            )
        for k in batch_keys:
            self._init_arr(k)

    def append(self, **batches: dict[str, np.ndarray]):
        """
        Appends batches of values, constrained on the limits in unison.

        Parameters
        ----------
        **batches : dict[str, np.ndarray]
            A dictionary of key-value pairs, where key is type of prediction (e.g predictions, labels),
            and value is a batch of prediction values. Note that the passed keys in ``**batches`` must match arguments
            in evaluation functions arguments in the Callable in `evaluation_functions`
            when we initialize `PredictionStore` object.

        Raises
        ------
        ValueError
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
        batch_keys = set(batches.keys())
        if len(batch_keys) == 0:
            raise ValueError("Must provide keyed batch arguments.")
        if self._batch_keys is None:
            self._init_keys(batch_keys=batch_keys)
            assert self._batch_keys is not None
        sizes = {}
        assert self._batch_keys == batch_keys, (
            f"Inhomogeneous keys from the prediction store update. "
            f"Expected: {sorted(self._batch_keys)}, received {sorted(batch_keys)}"
        )
        for k, v in batches.items():
            np_arr = butils.iter_to_numpy(v)
            sizes[k] = len(np_arr)
            self._get_arr(k).append(np_arr)
        assert (
            len(set(sizes.values())) == 1
        ), f"Inhomegenous batches between inputs. Sizes: {sizes}"
        limits = []
        for k in self._batch_keys:
            if (_limit := self._get_arr(k).limit) is not None:
                limits.append(_limit)

        if self.limit is not None or len(limits) > 0:
            self.limit = min(limits)
            for k in self._batch_keys:
                self._get_arr(k).limit = self.limit

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

        metrics: dict[str, float] = {}
        if (
            self._batch_keys is None
            or self.__evaluation_functions__ is None
            or len(self._batch_keys) == 0
        ):
            return metrics
        batches = {k: self._get_arr(k).get() for k in self._batch_keys}
        for k, v in self.__evaluation_functions__.items():
            fn_args = set(inspect.signature(v).parameters.keys())
            intersecting_args = set(self._batch_keys).intersection(fn_args)
            assert (
                len(intersecting_args) > 0
            ), f"Evaluation function arguments {fn_args} different than stored predictions: {self._batch_keys}"
            metric = v(**{k: v for k, v in batches.items() if k in intersecting_args})
            metric = _parse_moving_average_val(metric)
            metrics[k] = metric
            try:
                self.metrics[k].append(metric)
            except Exception as exc:
                raise ValueError(
                    f"Invalid value {metric} returned by evaluation function {v.__name__}. Must be numeric scalar."
                ) from exc
        return metrics

    @property
    def evaluation_function_arguments(self):
        if self.__evaluation_functions__ is None:
            return {}
        return {
            k: list(inspect.signature(v).parameters.keys())
            for k, v in self.__evaluation_functions__.items()
        }

    def get(self) -> dict[str, np.ndarray] | None:
        if self._batch_keys is None:
            return None
        return {k: self._get_arr(k).get() for k in self._batch_keys}

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
        if self._batch_keys is None:
            return
        for k in self._batch_keys:
            self._get_arr(k).reset()


class MovingAverage(ArrayStore):
    """
    This class is used to store moving average metrics

    """

    @property
    def __mean__(self):
        return np.mean(self._arr)

    @property
    def value(self):
        if len(self._arr) > 0:
            return self.__mean__
        return np.nan

    def __lt__(self, __o: float) -> bool:
        return float(self.value).__lt__(__o)

    def __gt__(self, __o: float) -> bool:
        return float(self.value).__gt__(__o)

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
        scalar = _parse_moving_average_val(val)
        super().append(scalar)

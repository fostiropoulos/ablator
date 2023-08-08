import typing as ty
from collections.abc import Callable, Iterable

import ablator.utils.base as butils
from ablator.modules.metrics.stores import MovingAverage, PredictionStore


class LossDivergedError(Exception):
    pass


class Metrics:
    """
    Stores and manages predictions and calculates metrics given some custom evaluation functions.
    This class makes batch-updates as metrics are calculated while training/evaluating a model. It takes into
    account the memory limits, applies evaluation functions, and provides cached or online updates on the metrics.

    We can access all the metrics from the ``Metrics`` object using its ``to_dict()`` method. Refer to
    :ref:`Prototyping Models <prototype_models>` tutorial for more details.

    """

    def __init__(
        self,
        *args,
        batch_limit=30,
        memory_limit=1e8,
        evaluation_functions: dict[str, Callable] | None = None,
        moving_average_limit=3000,
        # metrics with their initial value that are updated manually, i.e. learning rate
        static_aux_metrics: dict[str, ty.Any] | None = None,
        # metrics for which we update with their moving average, i.e. loss
        moving_aux_metrics: Iterable[str] | None = None,
    ):
        """
        Initialize the train metrics settings

        Parameters
        ----------
        batch_limit : int, optional
            Maximum number of batches to keep for every category of data (specified by ``tags``), so only `batch_limit`
            number of latest batches is stored for each of the categories. Default is 30.
        memory_limit : int, optional
            Maximum memory (in bytes) of batches to keep for every category of data (specified by ``tags``). Every time
            this limit is exceeded, ``batch_limit`` will be reduced by 1. Default is 1e8.
        evaluation_functions : dict[str, Callable], optional
            A dictionary of key-value pairs, keys are evaluation function names, values are
            callable evaluation functions, e.g mean, sum. Note that arguments to this Callable
            must match with names of prediction batches that the model returns. So if model prediction over
            a batch looks like this: {"preds": <batch of predictions>, "labels": <batch of predicted labels>},
            then callable's arguments should be ``preds`` and ``labels``, e.g ``evaluation_functions=
            {"mean": lambda preds, labels: np.mean(preads) + np.mean(labels)}``. Default is None.
        moving_average_limit : int, optional
            The maximum number of values allowed to store moving average metrics. Default is 3000.
        static_aux_metrics : dict[str, ty.Any], optional
            A dictionary of static metrics, those with their initial value that are updated manually,
            such as learning rate, best loss, total steps, etc. Keys of this dictionary are static metric names,
            while values is a proper initial value. Default is None.
        moving_aux_metrics : Iterable[str], optional
            A list of metrics, those we update with their moving average, such as loss. Default is None.

        Examples
        --------
        Initialize an object of Metrics:

        >>> from ablator.modules.metrics.main import Metrics
        >>> train_metrics = Metrics(
        ...     batch_limit=30,
        ...     memory_limit=None,
        ...     evaluation_functions={"mean": lambda x: np.mean(x)},
        ...     moving_average_limit=100,
        ...     static_aux_metrics={"lr": 1.0},
        ...     moving_aux_metrics={"loss"},
        ... )
        >>> train_metrics.to_dict() # metrics are set to np.nan if it's not updated yet
        {
            "mean": np.nan, "loss": np.nan, "lr": 1.0
        }
        """

        assert len(args) == 0, "Metrics takes no positional arguments."

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
            list(_static_aux_metrics.keys())
        )
        self.__moving_aux_attributes__: list[str] = sorted(
            list(set(_moving_aux_metrics))
        )
        self.__moving_eval_attributes__: list[str] = sorted(
            list(self.__evaluation_functions__)
        )
        _all_attr_names = (
            self.__moving_aux_attributes__
            + self.__moving_eval_attributes__
            + self.__static_aux_attributes__
        )
        duplicates = {x for x in _all_attr_names if _all_attr_names.count(x) > 1}

        assert len(duplicates) == 0, (
            f"Duplicate metric names {duplicates}, for \n"
            f"`evaluation_functions`={self.__moving_eval_attributes__}, \n"
            f"`moving_aux_metrics`={self.__moving_aux_attributes__}, \n"
            f"`static_aux_metrics`={self.__static_aux_attributes__}."
        )

        for name, v in _static_aux_metrics.items():
            setattr(self, name, v)
        for name in set(self.__moving_aux_attributes__).union(
            self.__moving_eval_attributes__
        ):
            self._init_ma(name)

        self._preds = PredictionStore(
            batch_limit=self.__batch_limit__,
            memory_limit=self.__memory_limit__,
            evaluation_functions=self.__evaluation_functions__,
        )

    def update_static_metrics(self, metric_dict: dict[str, ty.Any]):
        """
        Update static metrics with the values in metric_dict.

        Parameters
        ----------
        metric_dict : dict[str, ty.Any]
            A dictionary containing the static metrics values to update.

        Raises
        ------
        AssertionError:
            If metric_dict has metrics that are not in static_aux_attributes.

        Notes
        -----
        Not all metric_dict items must be preset from static_aux_attributes.
        i.e. metric_dict.items - static_aux_attributes =/= static_aux_attributes - metric_dict.items

        Examples
        --------
        >>> from ablator.modules.metrics.main import Metrics
        >>> train_metrics = Metrics(
        ...     batch_limit=30,
        ...     memory_limit=None,
        ...     evaluation_functions={"mean": lambda x: np.mean(x)},
        ...     moving_average_limit=100,
        ...     static_aux_metrics={"lr": 1.0},
        ...     moving_aux_metrics={"loss"},
        ... )
        >>> train_metrics.to_dict()
        {
            "train_mean": np.nan, "train_loss": np.nan,
            "lr": 1.0
        }
        >>> train_metrics.update_static_metrics({"lr": 0.3})
        >>> train_metrics.to_dict()
        {
            "train_mean": np.nan, "train_loss": np.nan,
            "lr": 0.3
        }

        """
        diff_metrics = set(metric_dict.keys()).difference(
            self.__static_aux_attributes__
        )
        metric_keys = sorted(list(metric_dict.keys()))
        assert len(diff_metrics) == 0, (
            "There are difference in the class metrics: "
            f"{self.__static_aux_attributes__} and updated metrics {metric_keys}"
        )
        metric_dict = butils.iter_to_numpy(metric_dict)
        for k, v in metric_dict.items():
            setattr(self, k, v)

    def update_ma_metrics(self, metric_dict: dict[str, ty.Any]):
        """
        Keep the moving average aux metrics updated with new values from metric_dict.
        This method will append the new metric values to its collection of metric results.
        A sample use case for this method is when we finish a training iteration, we
        can add the training loss to ``loss`` moving average metric collection.

        Parameters
        ----------
        metric_dict : dict[str, ty.Any]
            A dictionary containing the moving average metric values to update.

        Raises
        ------
        AssertionError:
            If metric_dict has metrics that are not in moving_aux_metrics.

        Examples
        --------
        >>> from ablator.modules.metrics.main import Metrics
        >>> train_metrics = Metrics(
        ...     batch_limit=30,
        ...     memory_limit=None,
        ...     evaluation_functions={"sum": lambda x: np.mean(x)},
        ...     moving_average_limit=100,
        ...     static_aux_metrics={"lr": 1.0},
        ...     moving_aux_metrics={"loss"},
        ... )
        >>> train_metrics.to_dict()
        {
            "train_sum": np.nan, "train_loss": np.nan,
            "val_sum": np.nan, "val_loss": np.nan,
            "lr": 1.0
        }
        >>> train_metrics.update_ma_metrics({"loss": 0.35})
        >>> train_metrics.to_dict()
        {
            "train_sum": np.nan, "train_loss": np.nan,
            "val_sum": np.nan, "val_loss": 0.35,
            "lr": 1.0
        }
        """
        metric_keys = set(metric_dict)
        diff_metrics = metric_keys.difference(set(self.__moving_aux_attributes__))
        assert len(diff_metrics) == 0, (
            "There are difference in the class metrics: "
            f"{self.__moving_aux_attributes__} and parsed metrics {sorted(list(metric_keys))}"
        )
        self._update_ma_metrics(metric_dict)

    def _update_ma_metrics(self, metric_dict: dict[str, ty.Any]):
        # metric dict should contain scalars
        metric_dict = butils.iter_to_numpy(metric_dict)
        for k, v in metric_dict.items():
            self._get_ma(k).append(v)

    def reset(self):
        """
        Reset to empty all prediction sequences (e.g predictions, labels).

        Examples
        --------
        >>> train_metrics = Metrics(
        ...     batch_limit=30,
        ...     memory_limit=None,
        ...     evaluation_functions={"sum": lambda pred: np.mean(pred)},
        ...     moving_average_limit=100,
        ...     static_aux_metrics={"lr": 1.0},
        ...     moving_aux_metrics={"loss"},
        ... )
        >>> train_metrics.append_batch(pred=np.array([1] * 3))    # e.g add 3 predictions all of class 1
        >>> train_metrics.reset()
        """
        self._preds.reset()

    def evaluate(self, reset=True, update_ma=True):
        """
        Apply evaluation_functions to the set of predictions. Possibly update the
        moving averages (only those associated with evaluation functions, not moving auxiliary metrics) with
        the evaluated results, or reset the predictions.

        Parameters
        ----------
        reset : bool, optional
            A flag that indicates whether to reset the predictions to empty after evaluation. Default is True.
        update_ma : bool, optional
            A flag that indicates whether to update the moving averages after evaluation. Default is True.

        Returns
        -------
        metrics : dict
            A dictionary of metric values calculated from the predictions.

        Examples
        --------
        >>> from ablator.modules.metrics.main import Metrics
        >>> train_metrics = Metrics(
        ...     batch_limit=30,
        ...     memory_limit=None,
        ...     evaluation_functions={"mean": lambda pred: np.mean(pred)},
        ...     moving_average_limit=100,
        ...     static_aux_metrics={"lr": 1.0},
        ...     moving_aux_metrics={"loss"},
        ... )
        >>> train_metrics.append_batch(pred=np.array([100]))
        >>> train_metrics.evaluate("val", reset=False, update=True) # val_mean is updated to
            mean among batch mean values: (100 / 1) / 1 = 100.0
        >>> train_metrics.append_batch(pred=np.array([0] * 3))

        For the following examples, the current evaluation result is: ``(100 + 0 + 0 + 0) / 4 = 25`` (which is returned
        by evaluate() function), and since update=True, val_mean is updated to: ``(100.0 + 25) / 2 = 62.5`` (we can
        see this if we use .to_dict())

        >>> train_metrics.evaluate("val", reset=True, update=True)
        {'mean': 25.0}
        >>> train_metrics.to_dict()
        {'val_mean': 62.5}
        """
        metrics = self._preds.evaluate()
        if update_ma:
            self._update_ma_metrics(metrics)
        if reset:
            self._preds.reset()
        return metrics

    # pylint: disable=missing-param-doc
    def append_batch(self, *args, **kwargs):
        """
        Appends a batch of predictions to a specific set.

        Parameters
        ----------
        **kwargs : dict
            A dictionary of key-value pairs, where key is type of prediction (e.g predictions, labels),
            and value is a batch of prediction values. Note that the passed keys in ``**kwrags`` must match arguments in
            evaluation functions arguments in Callable in evaluation_functions when we initialize Metrics object.

        Raises
        ------
        ValueError
            If any positional arguments are passed.

        Notes
        -----
        this is because it is easy to mix up the order of pred, labels and tags

        Examples
        --------
        >>> from ablator.modules.metrics.main import Metrics
        >>> train_metrics = Metrics(
        ...     batch_limit=30,
        ...     memory_limit=None,
        ...     evaluation_functions={"mean": lambda labels: np.mean(labels)},
        ...     moving_average_limit=100,
        ...     static_aux_metrics={"lr": 1.0},
        ...     moving_aux_metrics={"loss"},
        ... )
        >>> train_metrics.append_batch(labels=np.array([100]))
        >>> train_metrics.append_batch(labels=np.array([0] * 3))
        >>> train_metrics.append_batch(labels=np.array([50]))

        """
        # NOTE this is because it is easy to mix up the order of pred, labels and tags
        if len(args) > 0:
            raise ValueError("Metrics.append_batch takes no positional arguments.")
        self._preds.append(**kwargs)

    def _init_ma(self, name) -> MovingAverage:
        attr_name = f"__{name}_ma__"
        _ma = MovingAverage(
            batch_limit=self.__moving_average_limit__,
            memory_limit=self.__memory_limit__,
        )
        setattr(self, attr_name, _ma)
        return getattr(self, attr_name)

    def _get_ma(self, name) -> MovingAverage:
        attr_name = f"__{name}_ma__"
        preds = getattr(self, attr_name)
        return preds

    def to_dict(self):
        """
        Get all metrics, i.e moving auxiliary metrics, moving evaluation metrics, and static auxiliary metrics.
        Note that moving attributes will be an averaged value of all previous batches. Metrics are
        set to np.nan if it's never updated before

        Examples
        --------
        >>> from ablator.modules.metrics.main import Metrics
        >>> train_metrics = Metrics(
        ...     batch_limit=30,
        ...     memory_limit=None,
        ...     evaluation_functions={"mean": lambda preds: np.mean(preds)},
        ...     moving_average_limit=100,
        ...     static_aux_metrics={"lr": 0.75},
        ...     moving_aux_metrics={"loss"},
        ... )
        >>> train_metrics.append_batch(preds=np.array([100]))
        >>> train_metrics.evaluate(reset=False, update=True)
        >>> train_metrics.to_dict()
        {
            'train_mean': np.nan, 'train_loss': np.nan,
            'val_mean': 100.0, 'val_loss': np.nan,
            'lr': 0.75
        }
        >>> train_metrics.append_batch(preds=np.array([0] * 3))
        >>> train_metrics.evaluate(reset=True, update=True)
        >>> train_metrics.to_dict()
        {
            'train_mean': np.nan, 'train_loss': np.nan,
            'val_mean': 62.5, 'val_loss': np.nan,
            'lr': 0.75
        }
        """
        attrs = self.__moving_aux_attributes__ + self.__moving_eval_attributes__
        ma_attrs = {k: self._get_ma(k).value for k in attrs}
        static_attrs = {k: getattr(self, k) for k in self.__static_aux_attributes__}
        return {**ma_attrs, **static_attrs}

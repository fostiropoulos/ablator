import typing as ty
from collections.abc import Callable, Iterable

import ablator.utils.base as butils
from ablator.modules.metrics.stores import MovingAverage, PredictionStore


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
        evaluation_functions: dict[str, Callable] | None = None,
        moving_average_limit=3000,
        tags: list[str] | None = None,
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
        tags : list[str], optional
            A list of tags to specify predictions results from different categories, a sample use case is to
            categorize different sets of data (train, evaluation, test sets), e.g: ``tags=["train", "val"]``
            This will be combined with evaluation function names and moving auxiliary metrics names to create metrics.
            For example, if ``evaluation_functions.keys() = ["mean"]``, ``moving_aux_metrics = ["loss"]``, then metrics
            that will be tracked are: ``train_mean``, ``train_loss``, ``val_mean``, ``val_loss``.
            Default is ``["train"]``.
        static_aux_metrics : dict[str, ty.Any], optional
            A dictionary of static metrics, those with their initial value that are updated manually,
            such as learning rate, best loss, total steps, etc. Keys of this dictionary are static metric names,
            while values is a proper initial value. Default is None.
        moving_aux_metrics : Iterable[str], optional
            A list of metrics, those we update with their moving average, such as loss. Default is None.

        Examples
        --------
        Initialize an object of TrainMetrics:

        >>> from ablator.modules.metrics.main import TrainMetrics
        >>> train_metrics = TrainMetrics(
        ...     batch_limit=30,
        ...     memory_limit=None,
        ...     evaluation_functions={"mean": lambda x: np.mean(x)},
        ...     moving_average_limit=100,
        ...     tags=["train", "val"],
        ...     static_aux_metrics={"lr": 1.0},
        ...     moving_aux_metrics={"loss"},
        ... )
        >>> train_metrics.to_dict() # metrics are set to np.nan if it's not updated yet
        {
            "train_mean": np.nan, "train_loss": np.nan,
            "val_mean": np.nan, "val_loss": np.nan,
            "lr": 1.0
        }
        """
        if tags is None:
            tags = ["train"]
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
        self.__tags__: list[str] = sorted(list(tags))
        self.__moving_aux_attributes__: list[str] = sorted(
            list(
                f"{tag}_{eval_metric}"
                for tag in self.__tags__
                for eval_metric in list(set(_moving_aux_metrics))
            )
        )
        self.__moving_eval_attributes__: list[str] = sorted(
            list(
                f"{tag}_{eval_fn}"
                for tag in self.__tags__
                for eval_fn in self.__evaluation_functions__
            )
        )
        _all_attr_names = (
            self.__moving_aux_attributes__
            + self.__moving_eval_attributes__
            + self.__static_aux_attributes__
        )
        duplicates = {x for x in _all_attr_names if _all_attr_names.count(x) > 1}

        assert (
            len(duplicates) == 0
        ), f"Duplicate metric names with built-ins {duplicates}"

        for tag, v in _static_aux_metrics.items():
            setattr(self, tag, v)
        for tag in set(self.__moving_aux_attributes__).union(
            self.__moving_eval_attributes__
        ):
            self._init_ma(tag)

        for tag in tags:
            self._init_preds(tag)

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
        >>> from ablator.modules.metrics.main import TrainMetrics
        >>> train_metrics = TrainMetrics(
        ...     batch_limit=30,
        ...     memory_limit=None,
        ...     evaluation_functions={"mean": lambda x: np.mean(x)},
        ...     moving_average_limit=100,
        ...     tags=["train"],
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

    def update_ma_metrics(self, metric_dict: dict[str, ty.Any], tag: str):
        """
        Keep the moving average aux metrics updated with new values from metric_dict.
        This method will append the new metric values to its collection of metric results.
        A sample use case for this method is when we finish a training iteration, we
        can add the training loss to ``loss`` moving average metric collection on tag ``train``,
        aka the train set.

        Parameters
        ----------
        metric_dict : dict[str, ty.Any]
            A dictionary containing the moving average metric values to update.
        tag : str
            A tag that specifies which set of predictions to update metric values.

        Raises
        ------
        AssertionError:
            If metric_dict has metrics that are not in moving_aux_metrics.

        Examples
        --------
        >>> from ablator.modules.metrics.main import TrainMetrics
        >>> train_metrics = TrainMetrics(
        ...     batch_limit=30,
        ...     memory_limit=None,
        ...     evaluation_functions={"sum": lambda x: np.mean(x)},
        ...     moving_average_limit=100,
        ...     tags=["train", "val"],
        ...     static_aux_metrics={"lr": 1.0},
        ...     moving_aux_metrics={"loss"},
        ... )
        >>> train_metrics.to_dict()
        {
            "train_sum": np.nan, "train_loss": np.nan,
            "val_sum": np.nan, "val_loss": np.nan,
            "lr": 1.0
        }
        >>> train_metrics.update_ma_metrics({"loss": 0.35}, tag="val")
        >>> train_metrics.to_dict()
        {
            "train_sum": np.nan, "train_loss": np.nan,
            "val_sum": np.nan, "val_loss": 0.35,
            "lr": 1.0
        }
        """
        metric_keys = {f"{tag}_{k}" for k in metric_dict}
        diff_metrics = metric_keys.difference(set(self.__moving_aux_attributes__))
        assert len(diff_metrics) == 0, (
            "There are difference in the class metrics: "
            f"{self.__moving_aux_attributes__} and parsed metrics {sorted(list(metric_keys))}"
        )
        self._update_ma_metrics(metric_dict, tag)

    def _update_ma_metrics(self, metric_dict: dict[str, ty.Any], tag=None):
        # metric dict should contain scalars
        metric_dict = butils.iter_to_numpy(metric_dict)
        for k, v in metric_dict.items():
            attr_name = f"{tag}_{k}" if tag is not None else k
            self._get_ma(attr_name).append(v)

    def reset(self, tag: str):
        """
        Reset to empty all prediction sequences (e.g predictions, labels)
        in a set of predictions specified by ``tag`` argument.

        Parameters
        ----------
        tag : str
            A tag that specifies which set of predictions to be reset.

        Examples
        --------
        >>> train_metrics = TrainMetrics(
        ...     batch_limit=30,
        ...     memory_limit=None,
        ...     evaluation_functions={"sum": lambda pred: np.mean(pred)},
        ...     moving_average_limit=100,
        ...     tags=["train", "val"],
        ...     static_aux_metrics={"lr": 1.0},
        ...     moving_aux_metrics={"loss"},
        ... )
        >>> train_metrics.append_batch(pred=np.array([1] * 3), tag="train")    # e.g add 3 predictions all of class 1
        >>> train_metrics.reset(tag="train")
        """
        preds = self._get_preds(tag)
        preds.reset()

    def evaluate(self, tag, reset=True, update_ma=True):
        """
        Apply evaluation_functions to a set of predictions specified by ``tag`` argument. Possibly update the
        moving averages (only those associated with evaluation functions, not moving auxiliary metrics) with
        the evaluated results, or reset the predictions.

        Parameters
        ----------
        tag : str
            A tag that specifies which set of predictions to evaluate.
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
        >>> from ablator.modules.metrics.main import TrainMetrics
        >>> train_metrics = TrainMetrics(
        ...     batch_limit=30,
        ...     memory_limit=None,
        ...     evaluation_functions={"mean": lambda pred: np.mean(pred)},
        ...     moving_average_limit=100,
        ...     tags=["train", "val"],
        ...     static_aux_metrics={"lr": 1.0},
        ...     moving_aux_metrics={"loss"},
        ... )
        >>> train_metrics.append_batch(pred=np.array([100]), tag="val")
        >>> train_metrics.evaluate("val", reset=False, update=True) # val_mean is updated to
            mean among batch mean values: (100 / 1) / 1 = 100.0
        >>> train_metrics.append_batch(pred=np.array([0] * 3), tag="val")

        For the following examples, the current evaluation result is: ``(100 + 0 + 0 + 0) / 4 = 25`` (which is returned
        by evaluate() function), and since update=True, val_mean is updated to: ``(100.0 + 25) / 2 = 62.5`` (we can
        see this if we use .to_dict())

        >>> train_metrics.evaluate("val", reset=True, update=True)
        {'mean': 25.0}
        >>> train_metrics.to_dict()
        {'val_mean': 62.5}
        """
        preds = self._get_preds(tag)
        metrics = preds.evaluate()
        if update_ma:
            self._update_ma_metrics(metrics, tag)
        if reset:
            preds.reset()
        return metrics

    def append_batch(self, *args, tag, **kwargs):
        """
        Appends a batch of predictions to a specific set.

        Parameters
        ----------
        tag : str
            A tag that specifies which set of predictions to evaluate.
        **kwargs : dict
            A dictionary of key-value pairs, where key is type of prediction (e.g predictions, labels),
            and value is a batch of prediction values. Note that the passed keys in ``**kwrags`` must match arguments in
            evaluation functions arguments in Callable in evaluation_functions when we initialize TrainMetrics object.

        Raises
        ------
        AssertionError
            If any positional arguments are passed, or if the provided tag is not a defined metric category.

        Notes
        -----
        this is because it is easy to mix up the order of pred, labels and tags

        Examples
        --------
        >>> from ablator.modules.metrics.main import TrainMetrics
        >>> train_metrics = TrainMetrics(
        ...     batch_limit=30,
        ...     memory_limit=None,
        ...     evaluation_functions={"mean": lambda labels: np.mean(labels)},
        ...     moving_average_limit=100,
        ...     tags=["train", "val"],
        ...     static_aux_metrics={"lr": 1.0},
        ...     moving_aux_metrics={"loss"},
        ... )
        >>> train_metrics.append_batch(labels=np.array([100]), tag="train")
        >>> train_metrics.append_batch(labels=np.array([0] * 3), tag="train")
        >>> train_metrics.append_batch(labels=np.array([50]), tag="val")

        """
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
        """
        Get all metrics, i.e moving aux metrics, moving evaluation metrics, and static aux metrics.
        Note that moving attributes will be an averaged value of all previous batches. Metrics are
        set to np.nan if it's never updated before

        Examples
        --------
        >>> from ablator.modules.metrics.main import TrainMetrics
        >>> train_metrics = TrainMetrics(
        ...     batch_limit=30,
        ...     memory_limit=None,
        ...     evaluation_functions={"mean": lambda preds: np.mean(preds)},
        ...     moving_average_limit=100,
        ...     tags=["train", "val"],
        ...     static_aux_metrics={"lr": 0.75},
        ...     moving_aux_metrics={"loss"},
        ... )
        >>> train_metrics.append_batch(preds=np.array([100]), tag="val")
        >>> train_metrics.evaluate("val", reset=False, update=True)
        >>> train_metrics.to_dict()
        {
            'train_mean': np.nan, 'train_loss': np.nan,
            'val_mean': 100.0, 'val_loss': np.nan,
            'lr': 0.75
        }
        >>> train_metrics.append_batch(preds=np.array([0] * 3), tag="val")
        >>> train_metrics.evaluate("val", reset=True, update=True)
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

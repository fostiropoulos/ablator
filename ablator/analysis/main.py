import logging
from pathlib import Path

import pandas as pd
from joblib import Memory

from ablator.analysis.plot.utils import parse_name_remap
from ablator.main.configs import Optim

logger = logging.getLogger(__name__)


class Analysis:
    """
    A class for analyzing experimental results.

    Attributes
    ----------
    optim_metrics : dict[str, Optim]
        A dictionary mapping metric names to optimization directions.
    save_dir : str | None
        The directory to save analysis results to.
    cache : Memory | None
        A joblib memory cache for saving results.
    categorical_attributes : list[str]
        The list of all the categorical hyperparameter names
    numerical_attributes : list[str]
        The list of all the numerical hyperparameter names
    experiment_attributes : list[str]
        The list of all the hyperparameter names
    results : pd.DataFrame
        The dataframe extracted from the results file based on given metrics names and hyperparameter names.

    """

    def __init__(
        self,
        results: pd.DataFrame,
        categorical_attributes: list[str],
        numerical_attributes: list[str],
        optim_metrics: dict[str, Optim],
        save_dir: str | None = None,
        cache=False,
    ) -> None:
        """
        Initialize the Analysis class.

        Parameters
        ----------
        results : pd.DataFrame
            The result dataframe.
        categorical_attributes : list[str]
            The list of all the categorical hyperparameter names
        numerical_attributes : list[str]
            The list of all the numerical hyperparameter names
        optim_metrics : dict[str, Optim]
            A dictionary mapping metric names to optimization directions.
        save_dir : str | None
            The directory to save analysis results to.
        cache : bool
            Whether to cache results.
        """
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
        """
        Remaps attribute and metric names in ``attributes`` and ``metrics`` DataFrames
        based on ``attribute_name_remap`` and ``metric_name_remap``, and updates ``metric_map``
        accordingly.

        Parameters
        ----------
        attributes : pandas.DataFrame
            The DataFrame containing attribute values for each experiment.
        metrics : pandas.DataFrame
            The DataFrame containing metric values for each experiment.
        metric_map : dict of str to Optim
            A dictionary mapping metric names to optimization objectives (minimization or maximization).
        metric_name_remap : dict of str to str or None, optional
            A dictionary mapping input metric names to output metric names.
            If None, the output metric names will be the same as the input metric names.
        attribute_name_remap : dict of str to str or None, optional
            A dictionary mapping input attribute names to output attribute names.
            If None, the output attribute names will be the same as the input attribute names.

        Returns
        -------
        pandas.DataFrame, pandas.DataFrame, dict of str to Optim
            The remapped ``attributes`` DataFrame, the remapped ``metrics`` DataFrame,
            and the updated ``metric_map`` dictionary.

        Examples
        --------
        >>> import pandas as pd
        >>> from enum import Enum
        >>> class Optim(Enum):
        ...     min = "min"
        ...     max = "max"
        ...
        >>> attributes = pd.DataFrame({"color": ["red", "blue"], "size": [10, 20]})
        >>> metrics = pd.DataFrame({"loss": [0.5, 0.4], "accuracy": [0.8, 0.9]})
        >>> metric_map = {"loss": Optim.min, "accuracy": Optim.max}
        >>> metric_name_remap = {"loss": "error", "accuracy": "acc"}
        >>> attribute_name_remap = {"color": "c", "size": "s"}
        >>> remapped_attrs, remapped_metrics, updated_map = Analysis._remap_results(
        ...     attributes, metrics, metric_map,
        ...     metric_name_remap=metric_name_remap,
        ...     attribute_name_remap=attribute_name_remap
        ... )
        >>> assert list(remapped_attrs.columns) == ["c", "s"]
        >>> assert list(remapped_metrics.columns) == ["error", "acc"]
        >>> assert updated_map == {"error": Optim.min, "acc": Optim.max}
        """
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

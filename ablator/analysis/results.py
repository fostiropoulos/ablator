import multiprocessing as mp
import traceback
from logging import warning
from pathlib import Path
import builtins

import numpy as np
import pandas as pd
import ray
from joblib import Memory

from ablator.config.main import ConfigBase
from ablator.config.mp import Optim, ParallelConfig, SearchSpace
from ablator.config.proto import RunConfig


def read_result(config_type: type[ConfigBase], json_path: Path) -> pd.DataFrame | None:
    """
    Read the results of an experiment and return them as a pandas DataFrame.

    The function reads the data from a JSON file, processes each row, and appends
    experiment attributes from a YAML configuration file. The resulting DataFrame
    is indexed and returned.

    Parameters
    ----------
    config_type : type[ConfigBase]
        The type of the configuration class that is used to load the experiment
        configuration from a YAML file.
    json_path : Path
        The path to the JSON file containing the results of the experiment.

    Returns
    -------
    pd.DataFrame | None
        A pandas DataFrame containing the processed experiment results.
        Returns None if there was an error in reading the json_path results.


    Examples
    --------
    >>> result json file:
    {
    "run_id": "run_1",
    "accuracy": 0.85,
    "loss": 0.35
    }
    {
    "run_id": "run_2",
    "accuracy": 0.87,
    "loss": 0.32
    }
    >>> config file
    experiment_name: "My Experiment"
    batch_size: 64
    >>> return value
           run_id  accuracy loss experiment_name batch_size     path
    0       run_1      0.85  0.35    My Experiment    64  path/to/experiment
    1        run_2      0.87  0.32    My Experiment    64  path/to/experiment
    """

    try:
        experiment_config = config_type.load(json_path.parent.joinpath("config.yaml"))
        experiment_attributes = experiment_config.make_dict(
            experiment_config.annotations, flatten=True
        )
        df = pd.read_json(json_path)
        df = pd.concat([pd.DataFrame([experiment_attributes] * len(df)), df], axis=1)
        df.index.name = "step"
        df.reset_index(inplace=True)
        df["trial_uid"] = json_path.parent.name
        return df.set_index(["trial_uid", "step"])

    # pylint: disable=broad-exception-caught
    except builtins.Exception:
        traceback.print_exc()
        return None


class Results:
    """
    Class for processing experiment results. You can use this class to read the results in an
    experiment output directory. This can be used in combination with ``PlotAnalysis`` to show the
    correlation between hyperparameters and metrics. Refer to :ref:`Interpreting Results
    <interpret_results>` tutorial for more details on plotting and interpreting experiment results.

    Examples
    --------

    >>> directory_path = Path('<path to experiment output defined in experiment_dir>')
    >>> results = Results(config = ParallelConfig, experiment_dir=directory_path, use_ray=True)
    >>> df = results.read_results(config_type=ParallelConfig, experiment_dir=directory_path)
    
    Pass ``df`` to ``PlotAnalysis`` to create an analysis object that's able to plot the correlation between
    the hyperparameters and metrics and save the plots to an output directory. For example, the following
    code snippet generates plots for each of the numerical and categorical hyperparameters and saves them to
    ``./plots`` directory. Here "Validation Accuracy" is the name of the main metric.

    >>> analysis = PlotAnalysis(
    ...         df,
    ...         save_dir="./plots",
    ...         cache=True,
    ...         optim_metrics={"val_accuracy": Optim.max},
    ...         numerical_attributes=<numerical name remap keys names>,
    ...         categorical_attributes=<categorical name remap keys names>,
    ...     )
    >>> analysis.make_figures(
    ...     metric_name_remap={
    ...         "val_accuracy": "Validation Accuracy",
    ...     },
    ...     attribute_name_remap= attribute_name_remap
    ... )
     

    Parameters
    ----------
    config : type[ParallelConfig]
        The configuration class used
    experiment_dir : str | Path
        The path to the experiment directory.
    cache : bool, optional
        Whether to cache the results, by default ``False``
    use_ray : bool, optional
        Whether to use ray for parallel processing, by default ``False``

    Attributes
    ----------
    experiment_dir : Path
        The path to the experiment directory.
    config : type[ParallelConfig]
        The configuration class used
    metric_map : dict[str, Optim]
        A dictionary mapping optimize metric names to their optimization direction.
    data: pd.DataFrame
        The processed results of the experiment. Refer ``read_results`` for more details.
    config_attrs: list[str]
        The list of all the optimizable hyperparameter names
    search_space: dict[str, ty.Any]
        All the search space of the experiment.
    numerical_attributes: list[str]
        The list of all the numerical hyperparameter names
    categorical_attributes: list[str]
        The list of all the categorical hyperparameter names
    """

    def __init__(
        self,
        config: type[ParallelConfig] | ParallelConfig,
        experiment_dir: str | Path,
        cache: bool = False,
        use_ray: bool = False,
    ) -> None:
        if not isinstance(config, type):
            config_type = type(config)
        else:
            config_type = config
        if issubclass(config_type, RunConfig) and not issubclass(
            config_type, ParallelConfig
        ):
            raise ValueError(
                "Provided a ``RunConfig`` used for a single-trial. Analysis "
                "is not meaningful for a single trial. Please provide a ``ParallelConfig``."
            )
        if not issubclass(config_type, ParallelConfig):
            raise ValueError(
                "Configuration must be subclassed from ``ParallelConfig``. "
            )
        # TODO parse results from experiment directory as opposed to configuration.
        # Need a way to derived MPConfig implementation from a pickled file.
        # We need the types of the configuration, metric map.
        self.experiment_dir = Path(experiment_dir)
        run_config_path = self.experiment_dir.joinpath("default_config.yaml")
        if not run_config_path.exists():
            raise FileNotFoundError(f"{run_config_path}")
        self.config = config_type.load(run_config_path)
        self.metric_map: dict[str, Optim] = self.config.optim_metrics
        self._make_cache(clean=not cache)
        self.data: pd.DataFrame = self._parse_results(
            self.experiment_dir, init_ray=use_ray
        )

        self.config_attrs: list[str] = list(self.config.search_space.keys())
        self.search_space: dict[str, SearchSpace] = self.config.search_space
        # NOTE possibly a good idea to set small range integer attributes to categorical
        self.numerical_attributes = [
            k for k, v in self.search_space.items() if v.value_range is not None
        ]
        self.categorical_attributes = [
            k for k, v in self.search_space.items() if v.categorical_values is not None
        ]
        self._assert_cat_attributes(self.categorical_attributes)

    def _make_cache(self, clean=False):
        memory = Memory(self.experiment_dir.joinpath(".cache"), verbose=0)
        self._parse_results = memory.cache(
            self._parse_results, ignore=["self", "init_ray"]
        )
        if clean:
            self._parse_results.clear()

    def _assert_cat_attributes(self, categorical_attributes: list[str]):
        """
        Check if the categorical attributes are imbalanced
        default ratio is 0.8, which means if the most frequent value
        is more than 80% every other kind of values, then it is imbalanced


        Parameters
        ----------
        categorical_attributes : list[str]
            list of categorical attributes

        Raises
        ------
        AssertionError
            if the categorical attributes are imbalanced

        Examples
        --------
        >>> [X,X,Y,Z]imbalanced
        >>> [X,Y,Z]balanced
        >>> [X,X Y,Y,Z]balanced
        """
        for attr in categorical_attributes:
            value_counts = self.data[attr].value_counts()
            unique_values, counts = np.array(value_counts.index), value_counts.values
            imbalance_ratio_cut_off = 0.8
            imbalanced_values = unique_values[
                counts / counts.max() > (1 - imbalance_ratio_cut_off)
            ]
            if len(imbalanced_values) == 1:
                warning(
                    f"Imbalanced trials for attr {attr} and values: {unique_values} with counts {counts}."
                )

    @property
    def metric_names(self) -> list[str]:
        """
        Get the list of all optimize directions

        Returns
        -------
        list[str]
            list of optimize metric names
        """
        return [str(v) for v in self.metric_map.values()]

    # method-hidden because we over-write it with the cached version.
    # pylint: disable=method-hidden
    def _parse_results(
        self,
        experiment_dir: Path,
        init_ray: bool,
    ):
        """
        Read multiple results from experiment directory with ray to enable parallel processing.

        Parameters
        ----------
        experiment_dir : Path
            The experiment directory
        init_ray : bool
            Whether to use ray for parallel processing
        """
        assert (
            experiment_dir.exists()
        ), f"Experiment directory {experiment_dir} does not exist. You can provide one as an argument `experiment_dir`"
        if init_ray and not ray.is_initialized():
            ray.init(address="local")
        return self.read_results(type(self.config), experiment_dir)

    @classmethod
    def read_results(
        cls,
        config_type: type[ConfigBase],
        experiment_dir: Path | str,
        num_cpus=None,
    ) -> pd.DataFrame:
        """
        Read multiple results from experiment directory with ray to enable parallel processing.

        This function calls ``read_result`` many times, refer to ``read_result`` for more details.

        Parameters
        ----------
        config_type : type[ConfigBase]
            The configuration class
        experiment_dir : Path | str
            The experiment directory
        num_cpus : int, optional
            Number of CPUs to use for ray processing, by default ``None``

        Returns
        -------
        pd.DataFrame
            A dataframe of all the results
        """
        results: list[pd.DataFrame] = []
        futures: list[ray.ObjectRef] = []
        json_paths = list(Path(experiment_dir).rglob("results.json"))
        if len(json_paths) == 0:
            raise RuntimeError(f"No results found in {experiment_dir}")
        cpu_count = mp.cpu_count()
        if num_cpus is None:
            num_cpus = len(json_paths) / (cpu_count * 4)
        json_path = None
        for json_path in json_paths:
            if ray.is_initialized():
                futures.append(
                    ray.remote(num_cpus=num_cpus)(read_result).remote(
                        config_type, json_path
                    )
                )
            else:
                results.append(read_result(config_type, json_path))
        if ray.is_initialized() and len(json_paths) > 0:
            # smoke test
            read_result(config_type, json_path)
            results = ray.get(futures)
            results = list(filter(lambda x: x is not None, results))
        return pd.concat(results)

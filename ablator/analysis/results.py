import functools
import json
import multiprocessing as mp
import traceback
import typing as ty
from logging import warning
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from joblib import Memory

from ablator.config.main import ConfigBase
from ablator.main.configs import Optim, ParallelConfig, SearchSpace


def process_row(row: str, **aux_info) -> dict[str, ty.Any] | None:
    """
    Process a given row to make it conform to the JSON format, loading it as a JSON object,
    and updating it with auxiliary information.

    Parameters
    ----------
    row : str
        The input row to be processed, expected to be a JSON-like string.
    **aux_info : dict
        Additional key-value pairs to be added to the resulting dictionary.

    Returns
    -------
    dict[str, ty.Any] | None
        A dictionary resulting from the combination of the processed row and auxiliary information,
        or ``None`` if the input row cannot be parsed as a JSON object.

    Raises
    ------
    AssertionError
        If there are overlapping column names between the auxiliary information and the input row.

    Examples
    --------
    >>> row = '"name": "John Doe", "age": 30'
    >>> aux_info = {"city": "San Francisco"}
    >>> process_row(row, **aux_info)
    {'name': 'John Doe', 'age': 30, 'city': 'San Francisco'}

    >>> row = '{"name": "John Doe", "age": 30}'
    >>> aux_info = {"age": 25, "city": "San Francisco"}
    >>> process_row(row, **aux_info)
    AssertionError: Overlapping column names between auxiliary dictionary and run results.
    aux_info: {'age': 25, 'city': 'San Francisco'}
    row: {"name": "John Doe", "age": 30}
    """
    if not row.startswith("{"):
        row = "{" + row
    if not row.endswith("}") and not row.endswith("}\n"):
        row += "}"
    s: dict[str, ty.Any] = {}
    try:
        s = json.loads(row)
    except json.decoder.JSONDecodeError:
        return None
    assert (
        len(list(filter(lambda k: k in s, aux_info.keys()))) == 0
    ), f"Overlapping column names between auxilary dictionary and run results. aux_info: {aux_info}\n\nrow:{row} "
    s.update(aux_info)
    return s


def read_result(config_type: type[ConfigBase], json_path: Path) -> pd.DataFrame:
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
    pd.DataFrame
        A pandas DataFrame containing the processed experiment results.

    Raises
    ------
    Exception
        If there is an error in processing the JSON file or loading the
        experiment configuration, the exception will be caught and the
        traceback will be printed.

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

        with open(json_path, "r", encoding="utf-8") as f:
            lines = f.read().split("}\n{")

        _process_row = functools.partial(
            process_row,
            **{
                **experiment_attributes,
                **{"path": json_path.parent.as_posix()},
            },
        )
        processed_rows = [_process_row(l) for l in lines]  # noqa
        processed_jsons = list(filter(lambda x: x is not None, processed_rows))
        df = pd.DataFrame(processed_jsons)

        if (malformed_rows := len(processed_rows) - len(processed_jsons)) > 0:
            print(f"Found {malformed_rows} malformed rows in {json_path}")
        return df.reset_index()

    except Exception:
        traceback.print_exc()
        return None


class Results:
    """
    Class for processing experiment results.

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
        config: type[ParallelConfig],
        experiment_dir: str | Path,
        cache: bool = False,
        use_ray: bool = False,
    ) -> None:
        assert issubclass(config, ParallelConfig), "Configuration must be of type. "
        # TODO parse results from experiment directory as opposed to configuration.
        # Need a way to derived MPConfig implementation from a pickled file.
        # We need the types of the configuration, metric map.
        self.experiment_dir = Path(experiment_dir)
        run_config_path = self.experiment_dir.joinpath("default_config.yaml")
        if not run_config_path.exists():
            raise FileNotFoundError(f"{run_config_path}")
        self.config = config.load(run_config_path)
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
        return list(map(str, self.metric_map.values()))

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
        experiment_dir: Path,
        num_cpus=None,
    ) -> pd.DataFrame:
        """
        Read multiple results from experiment directory with ray to enable parallel processing.

        This function calls ``read_result`` many times, refer to ``read_result`` for more details.

        Parameters
        ----------
        config_type : type[ConfigBase]
            The configuration class
        experiment_dir : Path
            The experiment directory
        num_cpus : int, optional
            Number of CPUs to use for ray processing, by default ``None``

        Returns
        -------
        pd.DataFrame
            A dataframe of all the results
        """
        results = []
        json_paths = list(experiment_dir.rglob("results.json"))
        if len(json_paths) == 0:
            raise RuntimeError(f"No results found in {experiment_dir}")
        cpu_count = mp.cpu_count()
        if num_cpus is None:
            num_cpus = len(json_paths) / (cpu_count * 4)
        json_path = None
        for json_path in json_paths:
            if ray.is_initialized():
                results.append(
                    ray.remote(num_cpus=num_cpus)(read_result).remote(
                        config_type, json_path
                    )
                )
            else:
                results.append(read_result(config_type, json_path))
        if ray.is_initialized() and len(json_paths) > 0:
            # smoke test
            read_result(config_type, json_path)
            results = ray.get(results)
            results = list(filter(lambda x: x is not None, results))
        return pd.concat(results)

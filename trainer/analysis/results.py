import functools
import multiprocessing as mp
import traceback
from logging import warning
from pathlib import Path
from typing import Dict, List, Optional, Type, Union, Literal

import numpy as np
import optuna
import pandas as pd
from joblib import Memory

import trainer.analysis.utils as autils
from trainer.config.main import ConfigBase
from trainer.config.run import ParallelConfig, Optim
from trainer.utils.config import flatten_nested_dict

try:
    import ray

except ImportError:
    pass


class Results:
    def __init__(
        self,
        config: type[ParallelConfig],
        experiment_dir: Union[str, Path],
        cache: bool = False,
        use_ray: bool = False,
        return_only: Literal["all", "last", "best"] = "all",
    ) -> None:
        assert issubclass(config, ParallelConfig), "Configuration must be of type. "
        # TODO parse results from experiment directory as opposed to configuration.
        # Need a way to derived MPConfig implementation from a pickled file.
        # We need the types of the configuration, metric map.
        self.experiment_dir = Path(experiment_dir)
        run_config_path = self.experiment_dir.joinpath("run_config.yaml")
        if not run_config_path.exists():
            raise FileNotFoundError(f"{run_config_path}")
        self.config = config.load(run_config_path)
        self.metric_map: Dict[str, Optim] = self._parse_metrics(self.config)
        self.make_cache(clean=not cache)
        self.data: pd.DataFrame = self._parse_results(
            self.experiment_dir, return_only=return_only, init_ray=use_ray
        )

        self.config_attrs: List[str] = list(
            flatten_nested_dict(config.tune, expand_list=False).keys()
        )

        self.numerical_attributes = self._parse_numerical_types(self.config_attrs)
        cat_attrs = set(self.numerical_attributes) - set(self.config_attrs)
        self.categorical_attributes = self._parse_categorical_types(cat_attrs)

    def export(self):
        # TODO export self.data, optuna study and config
        raise NotImplementedError()

    @classmethod
    def load(cls) -> "Results":
        # load from exported file. Results.load(pickle.file)
        raise NotImplementedError()

    def make_cache(self, clean=False):

        memory = Memory(self.experiment_dir.joinpath(".cache"), verbose=0)
        self._parse_results = memory.cache(
            self._parse_results, ignore=["self", "init_ray"]
        )
        if clean:
            self._parse_results.clear()

    def _parse_numerical_types(self, config_attrs, n_cat_box_plot_cutoff=5):
        numerical_attributes = []
        for attr in config_attrs:
            if (
                self.config.get_type_with_dot_path(attr) == int
                or self.config.get_type_with_dot_path(attr) == float
            ):
                # TODO maybe determine if attribute is numerical or categorical
                # based on optimization range.
                numerical_attributes.append(attr)
        return numerical_attributes

    def _parse_categorical_types(self, config_attrs):
        categorical_attributes = []
        for attr in config_attrs:
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
            categorical_attributes.append(attr)
        return categorical_attributes

    def _parse_metrics(self, config: ParallelConfig):
        metrics = config.optim_directions
        if metrics is None:
            raise ValueError("Need to specify `optim_directions` in the config")
        return {name: obj_fn for name, obj_fn in metrics}

    @property
    def metric_names(self) -> List[str]:
        return list(self.metric_map.values())

    def _parse_results(
        self,
        experiment_dir: Path,
        return_only: Literal["all", "best", "last"],
        init_ray: bool,
    ):

        assert (
            experiment_dir.exists()
        ), f"Experiment directory {experiment_dir} does not exist. You can provide one as an argument `experiment_dir`"
        if init_ray and not ray.is_initialized():
            ray.init(address="local")
        return self.read_results(type(self.config), experiment_dir, return_only)

    @classmethod
    def read_result(
        cls, config_type: Type[ConfigBase], json_path: Path
    ) -> pd.DataFrame:
        experiment_config = config_type.load(json_path.parent.joinpath("config.yaml"))
        experiment_attributes = experiment_config.make_dict(
            experiment_config.annotations, flatten=True
        )

        with open(json_path, "r") as f:
            lines = f.read().split("}\n{")

        _process_row = functools.partial(
            autils.process_row,
            **{
                **experiment_attributes,
                **{"path": json_path.parent.as_posix()},
            },
        )
        processed_rows = list(map(_process_row, lines))
        processed_jsons = list(filter(lambda x: x is not None, processed_rows))
        df = pd.DataFrame(processed_jsons)

        malformed_rows = len(processed_rows) - len(processed_jsons)
        if malformed_rows > 0:
            print(f"Found {malformed_rows} malformed rows in {json_path}")
        return df.reset_index()

    @classmethod
    def read_results(
        cls,
        config_type: Type[ConfigBase],
        experiment_dir: Path,
        num_cpus=None,
    ) -> pd.DataFrame:
        results = []
        json_paths = list(experiment_dir.rglob("results.json"))
        if len(json_paths) == 0:
            raise RuntimeError(f"No results found in {experiment_dir}")
        cpu_count = mp.cpu_count()
        if num_cpus is None:
            num_cpus = len(json_paths) / (cpu_count * 4)
        if num_cpus > 1:
            num_cpus = np.ceil(num_cpus)
        for json_path in json_paths:
            if ray.is_initialized():

                @ray.remote(num_cpus=num_cpus)
                def _lambda():
                    try:
                        return cls.read_result(config_type, json_path)
                    except:
                        traceback.print_exc()
                        return None

                results.append(_lambda.remote())
            else:
                results.append(cls.read_result(config_type, json_path))
        if ray.is_initialized():
            # smoke test
            cls.read_result(config_type, json_path)
            results = ray.get(results)
            results = filter(lambda x: x is not None, results)
        return pd.concat(results)

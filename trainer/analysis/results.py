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

from trainer.config.main import ConfigBase
from trainer.main.configs import Optim, ParallelConfig, SearchSpace


def process_row(row: str, **aux_info) -> dict[str, ty.Any] | None:
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


class Results:
    def __init__(
        self,
        config: type[ParallelConfig],
        experiment_dir: str | Path,
        cache: bool = False,
        use_ray: bool = False,
        return_only: ty.Literal["all", "last", "best"] = "all",
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
            self.experiment_dir, return_only=return_only, init_ray=use_ray
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
        categorical_attributes = []
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
            categorical_attributes.append(attr)
        return categorical_attributes

    @property
    def metric_names(self) -> list[str]:
        return list(self.metric_map.values())

    def _parse_results(
        self,
        experiment_dir: Path,
        return_only: ty.Literal["all", "best", "last"],
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
        cls, config_type: type[ConfigBase], json_path: Path
    ) -> pd.DataFrame:
        experiment_config = config_type.load(json_path.parent.joinpath("config.yaml"))
        experiment_attributes = experiment_config.make_dict(
            experiment_config.annotations, flatten=True
        )

        with open(json_path, "r") as f:
            lines = f.read().split("}\n{")

        _process_row = functools.partial(
            process_row,
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
        config_type: type[ConfigBase],
        experiment_dir: Path,
        return_only: ty.Literal["all", "last", "best"],
        num_cpus=None,
    ) -> pd.DataFrame:
        results = []
        json_paths = list(experiment_dir.rglob("results.json"))
        if len(json_paths) == 0:
            raise RuntimeError(f"No results found in {experiment_dir}")
        cpu_count = mp.cpu_count()
        if num_cpus is None:
            num_cpus = len(json_paths) / (cpu_count * 4)
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

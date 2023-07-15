import copy
import logging
import random
import typing as ty

import numpy as np


from ablator.config.mp import SearchAlgo
from ablator.config.hpo import SearchSpace
from ablator.main.hpo.base import BaseSampler


def expand_search_space(
    search_space: dict[str, SearchSpace]
) -> list[dict[str, str | int | float | dict]]:
    configs: list[dict[str, str | int | float | dict]] = [{}]

    def _parse_search_space(space: SearchSpace):
        if space.sub_configuration is not None:
            return expand_search_space(space.sub_configuration.arguments)
        if space.value_range is not None:
            low, high = space.parsed_value_range()
            num = space.n_bins if space.n_bins is not None else 10
            return np.linspace(low, high, num).tolist()
        if space.categorical_values is not None:
            return space.categorical_values
        if space.subspaces is not None:
            return [_ for _v in space.subspaces for _ in _parse_search_space(_v)]
        raise ValueError(f"Invalid SearchSpace: {space}")

    for k, v in search_space.items():
        if isinstance(v, dict):
            _configs = []
            for _config in configs:
                for _v in expand_search_space(v):
                    _config[k] = _v
                    _configs.append(copy.deepcopy(_config))

            configs = _configs
        elif isinstance(v, SearchSpace):
            _configs = []
            for _config in configs:
                for _v in _parse_search_space(v):
                    _config[k] = _v
                    _configs.append(copy.deepcopy(_config))
            configs = _configs
        else:
            for _config in configs:
                _config[k] = v
    return configs


class GridSampler(BaseSampler):
    def __init__(
        self,
        search_algo: SearchAlgo,
        search_space: dict[str, SearchSpace],
        configs: list[dict[str, ty.Any]] | None = None,
    ) -> None:
        self.search_space = search_space
        self.configs = expand_search_space(search_space)
        self.sampled_configs = configs if configs is not None else []
        for c in self.sampled_configs:
            if c in self.configs:
                self.configs.remove(c)
            else:
                self.sampled_configs.remove(c)
                logging.warning(
                    "Invalid sampled configuration provided to GridSampler. %s", c
                )
        if search_algo == SearchAlgo.grid:
            self.reset = False
        if search_algo == SearchAlgo.discrete:
            self.reset = True
        self._lock = False
        random.shuffle(self.configs)

    @property
    def _idx(self):
        return len(self.sampled_configs) - 1

    def _eager_sample(self):
        if len(self.configs) == 0 and self.reset:
            # reset
            self.configs = expand_search_space(self.search_space)
            random.shuffle(self.configs)
        elif len(self.configs) == 0:
            self._lock = False
            raise StopIteration
        cfg = self.configs.pop()
        self.sampled_configs.append(cfg)
        return self._idx, cfg

    def _drop(self):
        self.sampled_configs.pop()

    def update_trial(self, trial_id, metrics: dict[str, float] | None, state):
        """
        This function is a no-op for grid sampling as it is entirely random.
        """

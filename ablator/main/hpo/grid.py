import copy
import logging
import typing as ty

import numpy as np


from ablator.config.hpo import SearchSpace
from ablator.main.hpo.base import BaseSampler


def _parse_search_space(space: SearchSpace):
    if space.sub_configuration is not None:
        return _expand_search_space(space.sub_configuration.arguments)
    if space.value_range is not None:
        low, high = space.parsed_value_range()
        if space.n_bins is None:
            raise ValueError(f"`n_bins` must be specified for {space}.")
        num = space.n_bins
        dtype = int if space.value_type == "int" else float
        return sorted(set(np.linspace(low, high, num, dtype=dtype)))
    if space.categorical_values is not None:
        return space.categorical_values
    if space.subspaces is not None:
        return [_ for _v in space.subspaces for _ in _parse_search_space(_v)]  # type: ignore
    raise ValueError(f"Invalid SearchSpace: {space}")


def _expand_configs(configs, value: dict[str, SearchSpace] | SearchSpace | ty.Any, key):
    _configs = []
    if isinstance(value, dict):
        expanded_space = _expand_search_space(value)
    elif isinstance(value, SearchSpace):
        expanded_space = _parse_search_space(value)
    else:
        expanded_space = [value]
    for _config in configs:
        for _v in expanded_space:
            _config[key] = _v
            _configs.append(copy.deepcopy(_config))
    return _configs


def _expand_search_space(
    search_space: dict[str, SearchSpace]
) -> list[dict[str, str | int | float | dict]]:
    configs: list[dict[str, str | int | float | dict]] = [{}]

    for k, v in search_space.items():
        try:
            configs = _expand_configs(configs, v, k)
        except ValueError as e:
            raise ValueError(f"Invalid search space for {k}. {str(e)}") from e
    return configs


class GridSampler(BaseSampler):
    def __init__(
        self,
        search_space: dict[str, SearchSpace],
        configs: list[dict[str, ty.Any]] | None = None,
        seed: int | None = None,
    ) -> None:
        """
        GridSampler, expands the grid-space into evenly spaced intervals. For example,
        a search space over ``SearchSpace(value_range=[1,10], n_bins=10)`` will be discritized to
        10 intervals [1,..,10]. If the search space is composed of integers, e.g. ``value_type='int'``
        the search space will be rounded down via the default python `int()` implementation and only the unique subset
        will be considered. As a result the discritized search-space can be smaller than n_bins. For example:
        ``SearchSpace(value_range=[1,5], value_type='int', n_bins=1000)`` will produce a SearchSpace of ``{1,2,3,4,5}``.
        In contrast, ``SearchSpace(value_range=[1,5], value_type='float', n_bins=1000)`` will
        produce a SearchSpace of 1000 floats,
        ``[1. , 1.004004  , 1.00800801, ... , 4.98798799, 4.99199199, 4.995996  , 5.]``.


        Previous configurations can be supplied via the `configs` argument. If the configurations are not found in
        the discretized search_space (could be because of numerical stability errors or poor instantiation)
        they will be stored in memory. Any duplicate configurations will be removed from current sampling
        memory.

        Parameters
        ----------
        search_space : dict[str, SearchSpace]
            A dictionary with keys the configuration name and the search space to sample from
        configs : list[dict[str, ty.Any]] | None, optional
            Previous configurations to resume the state from, by default None
        seed : int | None, optional
            A seed to use for the HPO sampler, by default None
        """
        super().__init__()
        self.search_space = search_space
        self.configs = _expand_search_space(search_space)
        self.sampled_configs = configs if configs is not None else []
        for c in self.sampled_configs:
            if c in self.configs:
                self.configs.remove(c)
            else:
                logging.warning(
                    "Invalid sampled configuration provided to GridSampler. %s", c
                )
        self._lock = False
        self._rng = np.random.default_rng(seed)
        # mypy error because of nested dictionary
        self._rng.shuffle(self.configs)  # type: ignore

    @property
    def _idx(self):
        return len(self.sampled_configs) - 1

    def _eager_sample(self):
        if len(self.configs) == 0:
            # reset
            self.configs = _expand_search_space(self.search_space)
            self._rng.shuffle(self.configs)
        cfg = self.configs.pop()
        self.sampled_configs.append(cfg)
        return self._idx, cfg

    def _drop(self):
        self.sampled_configs.pop()

    def update_trial(self, trial_id, metrics: dict[str, float] | None, state):
        """
        This function is a no-op for grid sampling as it is entirely random.
        """

    def internal_repr(self, trial_id):
        """
        This function is a no-op for grid sampling as it does not need a reason
        to maintain an internal representation of trials.
        """
        return None

# type: ignore
# pylint: skip-file
"""
TODO current implementation is meant to be temporary until there is a concrete replacement to
Optuna or better integeration. The problems with using optuna are several:
1. Optuna does not support conditional search spaces
    for example if config-01.a=1 was sampled from [config-01,config-02] it is not taken into account when sampling
    config.a

2. Optuna internal TrialStates are limited, there are few states, and transition
between the states can cause error. For example we can not report metrics None
for TrialState.COMPLETE which is required when there are no optim_metrics
for a given sampling strategy.

3. Resuming the Sampler is problematic. As we have to now match the
internal experiment state to that of Optuna sampler.

4. Obscure implementation details. For example, it is unclear the benefit the distincition between
`indepedent sampling` and `relative sampling`. Additional implementation nuances can be
seen by inspecting the code, like if a parameter is already sampled for a given trial,
return that parameter, which is error prone as we might need to for example re-sample a
parameter in case of an error.

5. Removing trials in case of errors or issues is not possible. For example once a trial is sampled, it is stored
in the internal state. If the sampled configuration is invalid for whatever reason we do not want to store it.

Just to name a few...

"""
import collections
import typing as ty
import warnings
from collections import OrderedDict

import numpy as np
import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.study._study_direction import StudyDirection
from optuna.trial import TrialState

from ablator.config.hpo import FieldType, SearchSpace
from ablator.config.mp import Optim, SearchAlgo
from ablator.main.state import store as _state_store
from ablator.main.hpo.base import BaseSampler


class _Trial:
    """
    Mock `optuna.Trial` object for the sake of using optuna
    """

    def __init__(
        self,
        id_: int,
        study: "_Study",
        sampler: optuna.samplers.BaseSampler,
        optim_metrics: OrderedDict[str, str],
        resume_trial: ty.Optional["_state_store.Trial"] = None,
    ) -> None:
        self.state = TrialState.RUNNING

        self.values: np.ndarray | None = None
        self.id_ = id_
        self.params: dict[ty.Any, ty.Any] = {}
        self.distributions = {}
        self.optim_metrics = optim_metrics
        if resume_trial is not None:
            self.params = resume_trial._opt_params
            self.distributions = {
                k: eval(resume_trial._opt_distributions_types[k])(
                    **resume_trial._opt_distributions_kwargs[k]
                )
                for k in resume_trial._opt_distributions_kwargs
            }
            metrics = None
            if len(resume_trial.metrics) > 0:
                metrics = resume_trial.metrics[-1]
            self.update(metrics, resume_trial.state)
        self.relative_search_space = sampler.infer_relative_search_space(study, self)
        self.relative_params = sampler.sample_relative(
            study, self, self.relative_search_space
        )

    def update(
        self, metrics: OrderedDict[str, float] | None, state: "_state_store.TrialState"
    ):
        if state == _state_store.TrialState.COMPLETE:
            self.state = TrialState.COMPLETE
        elif state == _state_store.TrialState.FAIL:
            self.state = TrialState.FAIL
        elif state == _state_store.TrialState.RUNNING:
            self.state = TrialState.RUNNING
        else:
            return
        if metrics is not None:
            metric_keys = set(metrics.keys())
            optim_keys = set(self.optim_metrics.keys())
            if metric_keys != optim_keys:
                raise ValueError(
                    f"metric keys {metric_keys} do not match optim_keys {optim_keys}"
                )
            values = [
                metrics[k] if metrics[k] is not None else float("inf")
                for k in self.optim_metrics
            ]
            self.values = values
        else:
            self.values = None

    def is_relative_param(self, name: str, distribution: BaseDistribution) -> bool:
        return self._is_relative_param(name, distribution)

    def _is_relative_param(self, name: str, distribution: BaseDistribution) -> bool:
        if name not in self.relative_params:
            return False

        if name not in self.relative_search_space:
            raise ValueError(
                f"The parameter '{name}' was sampled by `sample_relative` method "
                "but it is not contained in the relative search space."
            )

        param_value = self.relative_params[name]
        param_value_in_internal_repr = distribution.to_internal_repr(param_value)
        return distribution._contains(param_value_in_internal_repr)


# type: ignore
class _Study:
    """
    Mock `optuna.Study` object for the sake of using optuna
    """

    def __init__(
        self,
        optim_metrics: collections.OrderedDict,
        sampler: optuna.samplers.BaseSampler,
        trials: list["_state_store.Trial"] | None = None,
    ) -> None:
        trials = [] if trials is None else trials
        self.trials: list[_Trial] = [
            _Trial(
                id_=trial.trial_num,
                study=self,
                sampler=sampler,
                resume_trial=trial,
                optim_metrics=optim_metrics,
            )
            for trial in trials
        ]
        self.sampler = sampler
        self.directions = []
        self.optim_metrics = optim_metrics
        for v in optim_metrics.values():
            if v in {Optim.min, "min"}:
                self.directions.append(StudyDirection.MINIMIZE)
            elif v in {Optim.max, "max"}:
                self.directions.append(StudyDirection.MAXIMIZE)
            else:
                raise ValueError(f"Unrecognized optim. Direction `{v}`")

    def get_trials(self, deepcopy, states):
        assert deepcopy is False
        return [t for t in self.trials if t.state in states]

    def _is_multi_objective(self):
        return len(self.directions) > 1

    def drop(self, trial_id):
        for i, trial in enumerate(self.trials):
            if trial.id_ == trial_id:
                del self.trials[i]
                return
        raise RuntimeError(f"Trial {trial_id} was not found.")

    def make_trial(self):
        id_ = max(t.id_ for t in self.trials) + 1 if len(self.trials) > 0 else 0
        return _Trial(
            id_=id_,
            study=self,
            sampler=self.sampler,
            optim_metrics=self.optim_metrics,
        )

    def get_trial(self, trial_id):
        for trial in self.trials:
            if trial.id_ == trial_id:
                return trial
        raise RuntimeError(f"Trial {trial_id} was not found.")

    def update(self, trial_id, metrics, state):
        self.get_trial(trial_id).update(metrics, state)


class OptunaSampler(BaseSampler):
    """
    OptunaSampler this class serves as an interface for Optuna based samplers.
    WARNING: The class will be refactored in future versions and should not be used
    by library users. The class is meant for internal use only.
    """

    def __init__(
        self,
        search_algo: SearchAlgo,
        search_space: dict[str, SearchSpace],
        optim_metrics: collections.OrderedDict[str, Optim],
        trials: list["_state_store.Trial"] | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.sampler: optuna.samplers.TPESampler | optuna.samplers.RandomSampler
        assert (
            len(optim_metrics) > 0
        ), "Need to specify 'optim_metrics' with `OptunaSampler`"
        self.optim_metrics = OrderedDict(optim_metrics)
        if search_algo == SearchAlgo.tpe:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.sampler = optuna.samplers.TPESampler(constant_liar=True, seed=seed)
        elif search_algo == SearchAlgo.random:
            self.sampler = optuna.samplers.RandomSampler(seed=seed)
        else:
            raise ValueError(f"Unrecognized search algorithm: {search_algo}.")
        self._study = _Study(optim_metrics, sampler=self.sampler, trials=trials)
        self.search_space = search_space

    def _eager_sample(self):
        trial = self._study.make_trial()
        config = self._sample_trial_params(trial, self.search_space)
        trial.state = TrialState.RUNNING
        self._study.trials.append(trial)
        return trial.id_, config

    def _drop(self):
        self._study.trials.pop()

    def _suggest(self, trial: _Trial, name, dist):
        if trial.is_relative_param(name, dist):
            val = trial.relative_params[name]
        else:
            val = self.sampler.sample_independent(self._study, trial, name, dist)
        trial.params[name] = val
        trial.distributions[name] = dist
        return val

    def _suggest_int(
        self,
        trial: _Trial,
        name: str,
        value_range: tuple[int, int] | tuple[float, float],
        log: bool = False,
        n_bins: int | None = None,
    ):
        low, high = value_range
        if n_bins is None:
            step = 1
        else:
            step = max((high - low) // n_bins, 1)
        dist = IntDistribution(low, high, log=log, step=step)
        return self._suggest(trial, name, dist)

    def _suggest_float(
        self,
        trial: _Trial,
        name: str,
        value_range: tuple[int, int] | tuple[float, float],
        log: bool = False,
        n_bins: int | None = None,
    ):
        low, high = value_range
        if n_bins is None:
            step = n_bins
        else:
            step = (high - low) / n_bins
        dist = FloatDistribution(low, high, log=log, step=step)
        return self._suggest(trial, name, dist)

    def _suggest_categorical(self, trial: _Trial, name: str, vals: list[str]):
        dist = CategoricalDistribution(choices=vals)
        return self._suggest(trial, name, dist)

    def _sample_trial_params(
        self,
        trial: _Trial,
        search_space: dict[str, SearchSpace | dict],
    ) -> dict[str, ty.Any]:
        parameter: dict[str, ty.Any] = {}

        def _sample_params(
            v,
            prefix: str = "",
        ):
            if isinstance(v, dict):
                return {
                    _k: _sample_params(_v, prefix=f"{prefix}.{_k}")
                    for _k, _v in v.items()
                }
            if not isinstance(v, SearchSpace):
                return v
            if v.value_range is not None and v.value_type == FieldType.discrete:
                return self._suggest_int(
                    trial, prefix, v.parsed_value_range(), v.log, v.n_bins
                )
            if v.value_range is not None and v.value_type == FieldType.continuous:
                return self._suggest_float(
                    trial, prefix, v.parsed_value_range(), v.log, v.n_bins
                )
            if v.categorical_values is not None:
                return self._suggest_categorical(trial, prefix, v.categorical_values)
            if v.subspaces is not None:
                # TODO make it non-random e.g. pick the best sub-configuration.
                # Can use a dummy categorical variable
                idx = np.random.choice(len(v.subspaces))
                return _sample_params(
                    v.subspaces[idx],
                    prefix=f"{prefix}_{idx}",
                )
            if v.sub_configuration is not None:
                return {
                    _k: _sample_params(_v, prefix=f"{prefix}.{_k}")
                    for _k, _v in v.sub_configuration.arguments.items()
                }
            raise ValueError(f"Invalid SearchSpace {v}.")

        for k, v in search_space.items():
            parameter[k] = _sample_params(v, k)

        return parameter

    def update_trial(
        self, trial_id, metrics: dict[str, float] | None, state: TrialState
    ):
        self._study.update(trial_id, metrics, state)

    def internal_repr(self, trial_id):
        params = self._study.get_trial(trial_id).params
        distributions = self._study.get_trial(trial_id).distributions
        return {
            "_opt_params": params,
            "_opt_distributions_kwargs": {
                k: v.__dict__ for k, v in distributions.items()
            },
            "_opt_distributions_types": {
                k: type(v).__name__ for k, v in distributions.items()
            },
        }

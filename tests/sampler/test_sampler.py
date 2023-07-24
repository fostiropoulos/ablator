import re
import typing as ty

import numpy as np
import optuna
import pandas as pd
import pytest
import torch

# NOTE these appear unused but are used by eval(kwargs[*])
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from scipy.stats import ks_2samp

from ablator.config.mp import SearchAlgo, SearchSpace
from ablator.config.utils import flatten_nested_dict
from ablator.main.hpo import GridSampler, OptunaSampler
from ablator.main.hpo.base import BaseSampler
from ablator.main.state import Trial, TrialState
from ablator.main.hpo.grid import _expand_search_space

BUDGET = 100
REPETITIONS = 10


def mock_train(config):
    lr = config["train_config.optimizer_config"]["arguments"]["lr"]
    if lr != 0.1:
        perf = lr**2
    else:
        perf = config["b"] ** 2
    return {"loss": perf, "b": config["b"], "lr": lr}


def mock_train_optuna(trial: optuna.Trial):
    b = trial.suggest_float("b", -10, 10)
    if np.random.choice(2):
        return b**2
    lr = trial.suggest_float("lr", 0, 1)
    return lr**2


def search_space(n_bins=None):
    return {
        "train_config.optimizer_config": SearchSpace(
            subspaces=[
                {"sub_configuration": {"name": "sgd", "arguments": {"lr": 0.1}}},
                {
                    "sub_configuration": {
                        "name": "adam",
                        "arguments": {
                            "lr": {
                                "value_range": (0, 1),
                                "value_type": "float",
                                "n_bins": n_bins,
                            },
                            "wd": 0.9,
                        },
                    }
                },
                {
                    "sub_configuration": {
                        "name": {"categorical_values": ["adam", "sgd"]},
                        "arguments": {
                            "lr": {
                                "value_range": (0, 1),
                                "value_type": "float",
                                "n_bins": n_bins,
                            },
                            "wd": 0.9,
                        },
                    }
                },
            ]
        ),
        "b": SearchSpace(value_range=(-10, 10), value_type="float", n_bins=n_bins),
    }


def grid_search_space(n_bins=10):
    return {
        "a": SearchSpace(
            subspaces=[
                {
                    "sub_configuration": {
                        "d": {
                            "value_range": (0, 1),
                            "value_type": "float",
                            "n_bins": n_bins,
                        },
                        "c": 0.9,
                    }
                },
                {
                    "sub_configuration": {
                        "d": {
                            "value_range": (0, 1),
                            "value_type": "float",
                            "n_bins": n_bins,
                        },
                        "c": {
                            "subspaces": [
                                {
                                    "sub_configuration": {
                                        "i": {
                                            "value_range": (0, 1),
                                            "value_type": "float",
                                            "n_bins": n_bins,
                                        }
                                    }
                                },
                                {
                                    "sub_configuration": {
                                        "i": {
                                            "value_range": (0, 1),
                                            "value_type": "float",
                                            "n_bins": n_bins,
                                        }
                                    }
                                },
                            ]
                        },
                    }
                },
            ]
        ),
        "b": SearchSpace(value_range=(-10, 10), value_type="float", n_bins=n_bins),
        "c": SearchSpace(value_range=(-10, 10), value_type="float", n_bins=n_bins),
    }


def make_sampler(search_algo: SearchAlgo) -> BaseSampler:
    if search_algo == SearchAlgo.grid:
        space = search_space(n_bins=10)
        s = GridSampler(search_space=space, configs=[], seed=1)
    else:
        space = search_space()
        s = OptunaSampler(
            search_space=space,
            optim_metrics={"loss": "min"},
            search_algo=search_algo,
            trials=[],
            seed=1,
        )
    return s


def _ablator_sampler(search_algo: SearchAlgo, budget=None, assert_config=False):
    budget = BUDGET if budget is None else budget
    sampler = make_sampler(search_algo)
    perfs = []
    for i in range(budget):
        try:
            trial_id, config, kwargs = sampler.eager_sample()
            if assert_config:
                if search_algo == "grid":
                    assert kwargs is None
                else:
                    assert set(kwargs) == {
                        "_opt_params",
                        "_opt_distributions_kwargs",
                        "_opt_distributions_types",
                    }

                assert trial_id == i
                assert [eval(str(v)) for v in sampler.search_space.values()]
                assert [v.contains(config[k]) for k, v in sampler.search_space.items()]
            sampler.unlock(drop=False)
        except StopIteration:
            break
        perf = mock_train(config)
        sampler.update_trial(
            trial_id,
            {"loss": perf["loss"]},
            state=TrialState.COMPLETE,
        )
        perfs.append(perf)
    return pd.DataFrame(perfs)


@pytest.mark.parametrize("search_algo", list(SearchAlgo.__members__.keys()))
def test_sampled_config(search_algo):
    _ablator_sampler(search_algo, budget=100, assert_config=True)


@pytest.mark.parametrize("search_algo", ["tpe", "random"])
def test_optuna_kwargs(search_algo):
    search_space = grid_search_space()
    s = OptunaSampler(
        search_space=search_space,
        optim_metrics={"loss": "min"},
        search_algo=search_algo,
        trials=[],
    )
    for i in range(10):
        trial_id, config, kwargs = s.eager_sample()
        s.unlock(drop=False)
        fconfig = flatten_nested_dict(config)
        for k, v in kwargs["_opt_params"].items():
            _k = re.sub("_[0-9]", "", k)
            assert fconfig[_k] == v


def _test_tpe_continue():
    budget = BUDGET

    space = search_space()
    s = OptunaSampler(
        search_space=space,
        optim_metrics={"loss": "min"},
        search_algo="tpe",
        trials=[],
    )
    trials = []
    perfs = []
    for i in range(budget // 2):
        try:
            trial_id, config, kwargs = s.eager_sample()
            s.unlock(drop=False)
        except StopIteration:
            break
        perf = mock_train(config)
        perfs.append(perf)
        s.update_trial(
            trial_id,
            {"loss": perf["loss"]},
            state=TrialState.COMPLETE,
        )
        kwargs = s.internal_repr(trial_id)
        trial = Trial(
            config_uid=None,
            config_param=None,
            aug_config_param=config,
            trial_num=trial_id,
            state=TrialState.COMPLETE,
            metrics=[{"loss": perf["loss"]}],
            **kwargs
        )
        trials.append(trial)
    s = OptunaSampler(
        search_algo="tpe",
        search_space=space,
        optim_metrics={"loss": "min"},
        trials=trials,
    )
    for i in range(budget // 2, budget):
        try:
            trial_id, config, kwargs = s.eager_sample()
            assert trial_id == i
            s.unlock(drop=False)
        except StopIteration:
            break
        perf = mock_train(config)
        perfs.append(perf)
        s.update_trial(
            trial_id,
            {"loss": perf["loss"]},
            state=TrialState.COMPLETE,
        )
    return pd.DataFrame(perfs)


def test_tpe_continue():
    disc_tpe_df = pd.concat([_test_tpe_continue() for i in range(REPETITIONS)])
    tpe_df = pd.concat([_ablator_sampler("tpe") for i in range(REPETITIONS)])
    # optuna_df = pd.concat([_optuna_sampler("tpe") for i in range(REPETITIONS)])
    loss_disc = _get_top_n(disc_tpe_df)
    loss = _get_top_n(tpe_df)
    assert abs(loss - loss_disc) < 0.0001


def _update_tpe():
    sampler = make_sampler("tpe")
    perfs = []
    for i in range(BUDGET):
        trial_id, config, _ = sampler.eager_sample()
        sampler.unlock(drop=False)
        perf = mock_train(config)
        perfs.append(perf)
    for i, perf in enumerate(perfs):
        sampler.update_trial(
            i,
            {"loss": perf["loss"]},
            state=TrialState.COMPLETE,
        )
    return pd.DataFrame(perfs)


def test_update_tpe_error():
    sampler = make_sampler("tpe")
    trial_id, *_ = sampler.eager_sample()
    try:
        trial_id, *_ = sampler.eager_sample()
    except:
        sampler.unlock(drop=True)

    assert len(sampler._study.trials) == 0

    trial_id, *_ = sampler.eager_sample()
    sampler.unlock(drop=False)
    assert len(sampler._study.trials) == 1

    trial_id, *_ = sampler.eager_sample()
    sampler.unlock(drop=False)
    assert (
        len(sampler._study.trials) == 2 and trial_id == len(sampler._study.trials) - 1
    )
    try:
        sampler.update_trial(
            trial_id,
            {"loss2": 0.01},
            state=TrialState.COMPLETE,
        )
        assert False
    except ValueError as e:
        assert "metric keys {'loss2'} do not match optim_keys {'loss'}" in str(e)
    sampler.update_trial(
        trial_id,
        {"loss": 0.01},
        state=TrialState.COMPLETE,
    )
    assert sampler._study.trials[trial_id].values == [0.01]
    s2 = OptunaSampler(
        search_space=search_space(),
        optim_metrics={"loss1": "min", "loss2": "min"},
        search_algo="tpe",
        trials=[],
    )
    trial_id, *_ = s2.eager_sample()

    s2.unlock(drop=False)
    s2.update_trial(
        trial_id,
        {"loss2": 2, "loss1": 1},
        state=TrialState.COMPLETE,
    )

    assert s2._study.trials[trial_id].values == [1, 2]
    s2.update_trial(
        trial_id,
        {"loss1": 4, "loss2": 3},
        state=TrialState.COMPLETE,
    )
    assert s2._study.trials[trial_id].values == [4, 3]


def _optuna_sampler(sampler: ty.Literal["random", "tpe"]):
    if sampler == "random":
        sampler = optuna.samplers.RandomSampler()
    elif sampler == "tpe":
        sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler)  # Create a new study.
    study.optimize(mock_train_optuna, n_trials=BUDGET)
    return pd.DataFrame([{**{"loss": t.values[0]}, **t.params} for t in study.trials])


def _get_top_n(df: pd.DataFrame, top_n=None):
    top_n = int(BUDGET * 0.1) if top_n is None else top_n
    return df.sort_values("loss").iloc[:top_n].mean()["loss"].item()


def test_tpe():
    tpe_df = pd.concat([_ablator_sampler("tpe") for i in range(REPETITIONS)])
    optuna_df = pd.concat([_optuna_sampler("tpe") for i in range(REPETITIONS)])
    loss = _get_top_n(tpe_df)
    opt_loss = _get_top_n(optuna_df)
    assert abs(loss - opt_loss) < 0.0001


def test_random():
    rand_df = pd.concat([_ablator_sampler("random") for i in range(REPETITIONS)])
    optuna_rand_df = pd.concat([_optuna_sampler("random") for i in range(REPETITIONS)])
    loss = _get_top_n(rand_df)
    opt_loss = _get_top_n(optuna_rand_df)
    assert abs(loss - opt_loss) < 0.001


def test_update_tpe():
    # Test whether lazy updates of TPE cause reduction in performance (Expected as it samples at random when not available) however not exactly random as it does not sample from approx close configurations
    update_tpe = pd.concat([_update_tpe() for i in range(REPETITIONS)])
    rand_df = pd.concat([_ablator_sampler("random") for i in range(REPETITIONS)])
    tpe_df = pd.concat([_ablator_sampler("tpe") for i in range(REPETITIONS)])
    tpe2_df = pd.concat([_ablator_sampler("tpe") for i in range(REPETITIONS)])
    loss = _get_top_n(rand_df)
    update_tpe_loss = _get_top_n(update_tpe)
    tpe_loss = _get_top_n(tpe_df)
    tpe2_loss = _get_top_n(tpe2_df)
    assert abs(loss - update_tpe_loss) < 0.0001 and abs(
        update_tpe_loss - tpe_loss
    ) > abs(tpe2_loss - tpe_loss)


def test_grid_sampler(assert_error_msg):
    msg = assert_error_msg(
        lambda: GridSampler(
            search_space={"b": SearchSpace(value_range=(-10, 10), value_type="float")}
        )
    )
    assert (
        msg
        == "Invalid search space for b. `n_bins` must be specified for SearchSpace(value_range=(-10.0, 10.0), value_type='float')."
    )
    search_space_size = 10
    space = {
        "b": SearchSpace(
            value_range=(-10, 10), value_type="float", n_bins=search_space_size
        )
    }
    sampler = GridSampler(search_space=space)
    n_configs = len(sampler.configs)
    assert n_configs == search_space_size
    search_space_size = 20
    space = {
        "b": SearchSpace(
            value_range=(-10, 10), value_type="float", n_bins=search_space_size
        )
    }
    sampler = GridSampler(search_space=space)
    n_configs = len(sampler.configs)
    assert n_configs == search_space_size

    idx = 0
    dropped_trials = 0
    try:
        trial_id, *_ = sampler.eager_sample()
        idx += 1
        assert trial_id == 0
        sampler.eager_sample()
        assert False
    except:
        sampler.unlock(drop=False)
        trial_id, *_ = sampler.eager_sample()
        idx += 1
        assert trial_id == 1
        sampler.unlock(drop=True)
        dropped_trials += 1
        trial_id, *_ = sampler.eager_sample()
        idx += 1
        assert trial_id == 1
        sampler.unlock(drop=True)
        dropped_trials += 1
        assert True
    # sampled 3 times but dropped 2 configs
    assert (
        len(sampler.configs) == n_configs - idx
        and len(sampler.sampled_configs) == idx - dropped_trials
    )

    sampler = GridSampler(search_space=space)
    n_configs = len(sampler.configs)

    for i in range(n_configs * 2):
        sampler.eager_sample()
        sampler.unlock(drop=False)
    # the sampler should be able to reset when exceeding the limit of configs
    assert len(sampler.configs) == 0

    # Assert repeating
    sampler = GridSampler(search_space=space)
    n_configs = len(sampler.configs)
    configs = []
    for i in range(n_configs * 2):
        trial_id, cfg, _opt_kwargs = sampler.eager_sample()
        assert _opt_kwargs is None
        configs.append(cfg["b"])
        sampler.unlock(drop=False)
    assert len(np.unique(configs)) == n_configs

    grid_df = _ablator_sampler("grid", budget=BUDGET)
    grid2_df = _ablator_sampler("grid", budget=BUDGET * 100)

    assert ks_2samp(grid_df["loss"], grid2_df["loss"]).pvalue > 0.1
    assert _get_top_n(grid_df) < 0.01
    assert _get_top_n(grid_df, top_n=BUDGET // 2) > _get_top_n(
        grid2_df, top_n=BUDGET // 2
    )

    space = grid_search_space()
    sampler = GridSampler(space)
    assert len(sampler.configs) == 21000
    cfgs = []
    for i in range(100):
        _, cfg, _ = sampler.eager_sample()
        sampler.unlock(drop=False)
        cfgs.append(cfg)
    sampler2 = GridSampler(space, configs=cfgs)
    assert len(sampler2.configs) == len(sampler.configs) and len(sampler.configs) == (
        21000 - 100
    )


def test_optuna():
    try:
        s = OptunaSampler(
            search_space={},
            optim_metrics={},
            search_algo="tpe",
            trials=[],
        )
        assert False
    except Exception as e:
        assert "Need to specify 'optim_metrics' with `OptunaSampler`" in str(e)
    try:
        s = OptunaSampler(
            search_space={},
            optim_metrics={"test": "test"},
            search_algo="xxx",
            trials=[],
        )
        assert False
    except ValueError as e:
        assert "'xxx' is not a valid SearchAlgo" in str(e)


def test_expand_search_space():
    space = {"b": SearchSpace(value_range=(-10, 10), value_type="float", n_bins=10)}
    vals = np.array([s["b"] for s in _expand_search_space(space)])

    assert len(vals) == 10
    assert vals.max() == 10 and vals.min() == -10
    assert np.isclose(np.linspace(-10, 10, 10), vals).all()

    space = {"b": SearchSpace(value_range=(-10, 10), value_type="int", n_bins=10)}
    int_vals = np.array([s["b"] for s in _expand_search_space(space)])

    assert len(int_vals) == 10
    assert int_vals.max() == 10 and int_vals.min() == -10
    assert int_vals.dtype == "int"
    assert not np.isclose(vals[1:-1], int_vals[1:-1]).any()

    space = {"b": SearchSpace(value_range=(-10, 10), value_type="int", n_bins=1000)}
    int_vals = np.array([s["b"] for s in _expand_search_space(space)])
    assert len(int_vals) == 21
    assert (int_vals == np.arange(-10, 11)).all()


if __name__ == "__main__":
    from tests.conftest import _assert_error_msg

    # test_expand_search_space()
    # test_optuna()
    # test_tpe()
    # test_tpe_continue()
    # test_random()
    # test_update_tpe()
    # test_optuna_kwargs("tpe")
    # test_sampled_config("tpe")
    test_grid_sampler(_assert_error_msg)
    breakpoint()
    print()

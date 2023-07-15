import typing as ty

import numpy as np
import optuna
import pandas as pd

from ablator.config.mp import SearchAlgo, SearchSpace
from ablator.main.hpo import GridSampler, OptunaSampler
from ablator.main.hpo.base import BaseSampler
from ablator.main.state import TrialState, Trial
from ablator.config.utils import flatten_nested_dict
import random


import torch

# NOTE these appear unused but are used by eval(kwargs[*])
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
BUDGET = 100
REPETITIONS = 10
# TODO test eager_sample


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


def search_space():
    return {
        "train_config.optimizer_config": SearchSpace(
            subspaces=[
                {"sub_configuration": {"name": "sgd", "arguments": {"lr": 0.1}}},
                {
                    "sub_configuration": {
                        "name": "adam",
                        "arguments": {"lr": {"value_range": (0, 1)}, "wd": 0.9},
                    }
                },
                {
                    "sub_configuration": {
                        "name": {"categorical_values": ["adam", "sgd"]},
                        "arguments": {"lr": {"value_range": (0, 1)}, "wd": 0.9},
                    }
                },
            ]
        ),
        "b": SearchSpace(value_range=(-10, 10)),
    }


def grid_search_space():
    return {
        "a": SearchSpace(
            subspaces=[
                {"sub_configuration": {"d": {"value_range": (0, 1)}, "c": 0.9}},
                {
                    "sub_configuration": {
                        "d": {"value_range": (0, 1)},
                        "c": {
                            "subspaces": [
                                {"sub_configuration": {"i": {"value_range": (0, 1)}}},
                                {"sub_configuration": {"i": {"value_range": (0, 1)}}},
                            ]
                        },
                    }
                },
            ]
        ),
        "b": SearchSpace(value_range=(-10, 10)),
        "c": SearchSpace(value_range=(-10, 10)),
    }


def make_sampler(search_algo: SearchAlgo) -> BaseSampler:
    space = search_space()
    if search_algo in {SearchAlgo.discrete, SearchAlgo.grid}:
        s = GridSampler(search_space=space, search_algo=search_algo, configs=[])
    else:
        s = OptunaSampler(
            search_space=space,
            optim_metrics={"loss": "min"},
            search_algo=search_algo,
            trials=[],
        )
    return s


def _ablator_sampler(search_algo: SearchAlgo, budget=None):
    budget = BUDGET if budget is None else budget
    sampler = make_sampler(search_algo)
    perfs = []
    for i in range(budget):
        try:
            trial_id, config, kwargs = sampler.eager_sample()
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


def _update_tpe_error():
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


def _get_top_n(df: pd.DataFrame):
    top_n = int(BUDGET * 0.1)
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
    assert abs(loss - opt_loss) < 0.0003


def test_update_tpe():
    # Test whether lazy updates of TPE cause reduction in performance (Expected as it samples at random when not available) however not exactly random as it does not sample from approx close configurations
    _update_tpe_error()
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


def test_grid_sampler():
    space = {"b": SearchSpace(value_range=(-10, 10))}
    sampler = GridSampler(search_space=space, search_algo="discrete")
    n_configs = len(sampler.configs)
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

    sampler = GridSampler(search_space=space, search_algo="discrete")
    n_configs = len(sampler.configs)

    for i in range(n_configs * 2):
        sampler.eager_sample()
        sampler.unlock(drop=False)
    # the sampler should be able to reset when exceeding the limit of configs
    assert len(sampler.configs) == 0

    def _assert_stop_iter():
        sampler = GridSampler(search_space=space, search_algo="grid")
        n_configs = len(sampler.configs)
        for i in range(n_configs * 2):
            try:
                sampler.eager_sample()
                sampler.unlock(drop=False)
            except StopIteration:
                assert 0 == len(sampler.configs)
                return
        # the sampler should NOT be able to reset when exceeding the limit of configs
        assert False

    _assert_stop_iter()
    grid_df = _ablator_sampler("grid", budget=BUDGET)
    grid_df = _ablator_sampler("discrete", budget=BUDGET)
    grid2_df = _ablator_sampler("grid", budget=BUDGET * 100)
    grid3_df = _ablator_sampler("grid", budget=BUDGET * 100)
    assert np.isclose(grid3_df["loss"].mean(), grid2_df["loss"].mean())
    assert _get_top_n(grid_df) < 0.1

    space = grid_search_space()
    sampler = GridSampler("discrete", space)
    assert len(sampler.configs) == 21000
    cfgs = []
    for i in range(100):
        _, cfg, _ = sampler.eager_sample()
        sampler.unlock(drop=False)
        cfgs.append(cfg)
    sampler2 = GridSampler("discrete", space, configs=cfgs)
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


if __name__ == "__main__":
    # test_optuna()
    # test_tpe()
    # test_tpe_continue()
    test_random()
    # test_update_tpe()
    # test_grid_sampler()
    breakpoint()
    print()

import io
import shutil
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from ablator import ModelConfig, OptimizerConfig, RunConfig, TrainConfig
from ablator.config.mp import ParallelConfig, SearchAlgo, SearchSpace
from ablator.main.state import ExperimentState, TrialState
from ablator.modules.loggers.file import FileLogger


def _clean_path(p: Path):
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir()


class MockModel(ModelConfig):
    b: float = 0


class MockParallelConfig(ParallelConfig):
    model_config: MockModel


_search_space = {
    "train_config.optimizer_config": SearchSpace(
        subspaces=[
            {"sub_configuration": {"name": "sgd", "arguments": {"lr": 0.1}}},
            {
                "sub_configuration": {
                    "name": "adam",
                    "arguments": {
                        "lr": {"value_range": (0, 1), "value_type": "float"},
                        "weight_decay": 0.9,
                    },
                }
            },
            {
                "sub_configuration": {
                    "name": "adam",
                    "arguments": {
                        "lr": {
                            "subspaces": [
                                {"value_range": (0, 1), "value_type": "float"},
                                {"value_range": (0, 1), "value_type": "float"},
                            ]
                        },
                        "weight_decay": 0.9,
                    },
                }
            },
        ]
    ),
    "model_config.b": SearchSpace(value_range=(-10, 10), value_type="float"),
}


def make_config(search_space, search_algo):
    optimizer_config = OptimizerConfig(name="sgd", arguments={"lr": 0.1})
    train_config = TrainConfig(
        dataset="test",
        batch_size=128,
        epochs=2,
        optimizer_config=optimizer_config,
        scheduler_config=None,
    )

    is_grid_sampler = search_algo in {"grid"}
    optim_metrics = {"acc": "max"} if not is_grid_sampler else None
    return MockParallelConfig(
        train_config=train_config,
        model_config=MockModel(),
        verbose="silent",
        device="cpu",
        amp=False,
        search_space=search_space,
        optim_metrics=optim_metrics,
        total_trials=100,
        concurrent_trials=100,
        gpu_mb_per_experiment=0,
        search_algo=search_algo,
    )


@pytest.mark.parametrize("search_algo", list(SearchAlgo.__members__.keys()))
def test_state(tmp_path: Path, search_algo, assert_error_msg):
    assert_error_msg(
        lambda: SearchSpace(
            value_range=[0, 2],
        ),
        "`value_type` is required for `value_range` of SearchSpace",
    )
    assert_error_msg(
        lambda: SearchSpace(categorical_values=[0, "1", 0.122], value_type="int"),
        "Can not specify `value_type` without `value_range`.",
    )
    assert_error_msg(
        lambda: SearchSpace(sub_configuration={"test": "test"}, n_bins=10),
        "Can not specify `n_bins` without `value_range` or `categorical_values`.",
    )
    assert_error_msg(
        lambda: SearchSpace(
            subspaces=[SearchSpace(value_range=[0, 2], value_type="int")], n_bins=10
        ),
        "Can not specify `n_bins` without `value_range` or `categorical_values`.",
    )
    assert_error_msg(
        lambda: SearchSpace(
            value_range=[0, 1, 2], categorical_values=[0, "1", 0.122], value_type="int"
        ),
        "Incompatible lengths for value_range between [0, 1, 2] and type_hint: (<class 'str'>, <class 'str'>)",
    )
    assert_error_msg(
        lambda: SearchSpace(
            value_range=[0, 1], categorical_values=[0, "1", 0.122], value_type="float"
        ),
        "Must specify only one of 'value_range', 'subspaces', 'categorical_values' and / or 'sub_configurations' for SearchSpace.",
    )
    assert_error_msg(
        lambda: SearchSpace(),
        "Must specify only one of 'value_range', 'subspaces', 'categorical_values' and / or 'sub_configurations' for SearchSpace.",
    )
    search_space = {"some_var": SearchSpace(value_range=[0, 1], value_type="float")}

    config = make_config(search_space, search_algo)
    default_vals = config.make_dict(config.annotations, flatten=True)
    default_vals = [v for v in default_vals if not v.startswith("search_space")]
    _clean_path(tmp_path)
    assert_error_msg(
        lambda: ExperimentState(tmp_path, config),
        f"SearchSpace parameter some_var was not found in the configuration {sorted(list(default_vals))}.",
    )

    config.search_space = {
        "train_config.optimizer_config.name": SearchSpace(categorical_values=[0])
    }
    config.ignore_invalid_params = False

    _clean_path(tmp_path)
    assert_error_msg(
        lambda: ExperimentState(tmp_path, config).sample_trial(),
        "Invalid trial parameters {'train_config.optimizer_config.name': '0'}",
    )


@pytest.mark.parametrize("search_algo", list(SearchAlgo.__members__.keys()))
def test_sample_limits(tmp_path: Path, search_algo, assert_error_msg, capture_output):
    search_space = {
        "train_config.optimizer_config.name": SearchSpace(categorical_values=[0])
    }
    config = make_config(search_space, search_algo)
    config.ignore_invalid_params = True

    error_upper_bound = 20

    _clean_path(tmp_path)
    assert_error_msg(
        lambda: ExperimentState(tmp_path, config).sample_trial(),
        f"Reached maximum limit of misconfigured trials, {error_upper_bound} with {error_upper_bound} invalid trials.",
    )

    _clean_path(tmp_path)
    config.search_space = {
        "train_config.optimizer_config.arguments.lr": SearchSpace(
            categorical_values=[0, 1]
        )
    }
    state = ExperimentState(tmp_path, config, sampler_seed=1)
    for i in range(4):
        state.sample_trial()
    # this is because there are 2 categorical values
    assert len(set([t.config_uid for t in state.valid_trials()])) == 2
    _clean_path(tmp_path)
    config.search_space = {
        "train_config.optimizer_config.name": SearchSpace(
            categorical_values=["sgd", 0]
        ),
        "train_config.optimizer_config.arguments.lr": SearchSpace(
            value_range=[0, 1], value_type="float", n_bins=100
        ),
    }
    state = ExperimentState(tmp_path, config, logger=FileLogger())
    out, err = capture_output(lambda: [state.sample_trial() for i in range(10)])
    assert (
        "ignoring: {'train_config.optimizer_config.name': '0'," in out and len(err) == 0
    )

    _clean_path(tmp_path)
    config.search_space = {
        "train_config.optimizer_config.arguments.lr": SearchSpace(
            value_range=[0, 1],
            value_type="float",
            n_bins=1000,
        ),
    }
    s = ExperimentState(tmp_path, config)
    n_trials = 0
    assert len(s.valid_trials()) == n_trials
    s.sample_trial()
    n_trials += 1
    assert len(s.valid_trials()) == 1
    [s.sample_trial() for i in range(100)]
    n_trials += 100
    assert len(s.valid_trials()) == n_trials
    s.update_trial_state(0, None, state=TrialState.PRUNED)
    s.update_trial_state(53, None, state=TrialState.PRUNED)

    pruned_trials = s.get_trials_by_state(TrialState.PRUNED)
    assert {t.trial_num for t in pruned_trials} == {0, 53}

    assert len(s.valid_trials()) == n_trials
    s.update_trial_state(0, None, state=TrialState.PRUNED_INVALID)
    assert len(s.valid_trials()) == n_trials - 1
    pruned_trials = s.get_trials_by_state(TrialState.PRUNED)
    assert {t.trial_num for t in pruned_trials} == {53}

    s.update_trial_state(
        0,
        None,
        state=TrialState.RUNNING,
    )
    s.update_trial_state(
        0,
        None,
        state=TrialState.FAIL,
    )
    s.update_trial_state(
        0,
        {"acc": 0},
        state=TrialState.COMPLETE,
    )


@pytest.mark.parametrize("search_algo", list(SearchAlgo.__members__.keys()))
def test_state_resume(tmp_path: Path, search_algo, assert_error_msg):
    search_space = {
        "train_config.optimizer_config.arguments.lr": SearchSpace(
            value_range=[0, 1], value_type="float", n_bins=100
        ),
    }

    config = make_config(search_space, search_algo)
    s = ExperimentState(tmp_path, config)
    assert_error_msg(
        lambda: ExperimentState(tmp_path, config),
        f"{tmp_path.joinpath(f'{config.uid}_state.db')} exists. Please remove before starting another experiment or set `resume=True`.",
    )

    prev_trials = s.valid_trials_id()
    s = ExperimentState(tmp_path, config, resume=True)
    assert prev_trials == s.valid_trials_id()

    _clean_path(tmp_path)
    search_space = {
        "train_config.optimizer_config.arguments.lr": SearchSpace(
            value_range=[0, 1], value_type="float"
        ),
    }
    config.to_dict()
    s = ExperimentState(tmp_path, config)

    if config.optim_metrics is not None:
        assert_error_msg(
            lambda: s.update_trial_state(0, {"acc": 0}),
            "Trial 0 was not found.",
        )
        s.sample_trial()
        msg = assert_error_msg(
            lambda: s.update_trial_state(0, {"aaa": 0}),
            # ,
        )
        _str = "Expected to find acc in returned model metrics. Instead found: {'aaa'}"
        assert msg == (f'"{_str}"')
        for perf in [{"acc": np.nan}, {"acc": 0}, {"acc": None}]:
            trial_id, config = s.sample_trial()
            s.update_trial_state(trial_id, perf, TrialState.COMPLETE)
            trial_id, config = s.sample_trial()
            trial_id, config = s.sample_trial()
            trial_id, config = s.sample_trial()
            trial_id, config = s.sample_trial()

        assert_error_msg(
            lambda: s.update_trial_state(0, {"acc": "A"}, TrialState.COMPLETE),
            "ufunc 'isfinite' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''",
        )
    else:
        assert_error_msg(
            lambda: s.update_trial_state(0, None),
            "Trial 0 was not found.",
        )


def mock_train(config):
    lr = config.train_config.optimizer_config.arguments.lr
    b = config.model_config.b
    if lr != 0.1:
        perf = lr**2
    else:
        perf = b**2
    return {"acc": -perf, "b": b, "lr": lr}


def mock_train_optuna(trial: optuna.Trial):
    b = trial.suggest_float("b", -10, 10)
    if np.random.choice(2):
        return b**2
    lr = trial.suggest_float("lr", 0, 1)
    return lr**2


@pytest.fixture()
def search_space():
    return _search_space


BUDGET = 100


def _run_search_algo(s: ExperimentState):
    perfs = []
    for i in range(BUDGET):
        trial_id, config = s.sample_trial()
        perf = mock_train(config)
        s.update_trial_state(
            trial_id,
            {"acc": perf["acc"]},
            state=TrialState.COMPLETE,
        )
        perfs.append(perf)
    return pd.DataFrame(perfs)


def _get_top_n(df: pd.DataFrame):
    top_n = int(BUDGET * 0.1)
    return df.sort_values("acc")[::-1].iloc[:top_n].mean()["acc"].item()


def test_mock_run(tmp_path: Path, search_space):
    dfs = []
    for i in range(10):
        _clean_path(tmp_path)
        config = make_config(search_space, "tpe")
        s = ExperimentState(tmp_path, config)
        dfs.append(_run_search_algo(s))
    tpe_df = pd.concat(dfs)

    dfs = []
    for i in range(10):
        _clean_path(tmp_path)
        config = make_config(search_space, "random")
        s = ExperimentState(tmp_path, config)
        dfs.append(_run_search_algo(s))
    rand_df = pd.concat(dfs)
    loss_tpe = _get_top_n(tpe_df)
    loss_rand = _get_top_n(rand_df)
    assert abs(loss_tpe - loss_rand) > 1e-05


if __name__ == "__main__":
    from tests.conftest import _assert_error_msg, _capture_output

    tmp_path = Path("/tmp/state")
    test_mock_run(tmp_path, _search_space)

    for search_algo in ["random"]:
        _clean_path(tmp_path)
        test_state(tmp_path, search_algo, _assert_error_msg)

        _clean_path(tmp_path)
        test_sample_limits(tmp_path, search_algo, _assert_error_msg, _capture_output)

        _clean_path(tmp_path)
        test_state_resume(tmp_path, search_algo, _assert_error_msg)

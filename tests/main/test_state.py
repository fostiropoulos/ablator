from pathlib import Path

import shutil

import numpy as np
import optuna
import pandas as pd
import pytest


from ablator import ModelConfig, OptimizerConfig, RunConfig, TrainConfig
from ablator.config.mp import ParallelConfig, SearchAlgo, SearchSpace
from ablator.main.state import ExperimentState, TrialState
import io
from contextlib import redirect_stderr, redirect_stdout

from ablator.modules.loggers.file import FileLogger

optimizer_config = OptimizerConfig(name="sgd", arguments={"lr": 0.1})
train_config = TrainConfig(
    dataset="test",
    batch_size=128,
    epochs=2,
    optimizer_config=optimizer_config,
    scheduler_config=None,
)


def capture_output(fn):
    out = io.StringIO()

    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        fn()

    return out.getvalue(), err.getvalue()


def _clean_path(p: Path):
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir()


class MockModel(ModelConfig):
    b: float = 0


class MockParallelConfig(ParallelConfig):
    model_config: MockModel


def make_config(search_space, search_algo):
    is_grid_sampler = search_algo in {"grid", "discrete"}
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
        # cpus_per_experiment=0.1,
        search_algo=search_algo,
    )


@pytest.mark.parametrize("search_algo", list(SearchAlgo.__members__.keys()))
def test_state(tmp_path: Path, search_algo, assert_error_msg):
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
        "train_config.optimizer_config.name": SearchSpace(
            categorical_values=[0], value_type="int"
        )
    }
    config.ignore_invalid_params = False

    _clean_path(tmp_path)
    assert_error_msg(
        lambda: ExperimentState(tmp_path, config).sample_trial(),
        "Invalid trial parameters {'train_config.optimizer_config.name': '0'}",
    )


@pytest.mark.parametrize("search_algo", list(SearchAlgo.__members__.keys()))
def test_sample_limits(tmp_path: Path, search_algo, assert_error_msg):
    search_space = {
        "train_config.optimizer_config.name": SearchSpace(
            categorical_values=[0], value_type="int"
        )
    }
    config = make_config(search_space, search_algo)
    config.ignore_invalid_params = True
    config.sample_duplicate_params = False

    is_grid_sampler = search_algo in {"grid", "discrete"}
    if search_algo == "grid":
        error_upper_bound = 1
    else:
        error_upper_bound = 20

    assert_error_msg(
        lambda: ExperimentState(tmp_path, config).sample_trial(),
        f"Reached maximum limit of misconfigured trials, {error_upper_bound}.\nFound 0 duplicate and {error_upper_bound} invalid trials.",
    )

    _clean_path(tmp_path)
    config.search_space = {
        "train_config.optimizer_config.arguments.lr": SearchSpace(
            categorical_values=[0, 1], value_type="int"
        )
    }
    state = ExperimentState(tmp_path, config)
    state.sample_trial()
    state.sample_trial()
    if search_algo == "grid":
        assert_error_msg(
            lambda: state.sample_trial(),
            f"Reached maximum number of trials, for sampler {state.sampler.__class__.__name__}",
        )
    else:
        assert_error_msg(
            lambda: state.sample_trial(),
            f"Reached maximum limit of misconfigured trials, 20.\nFound 20 duplicate and 0 invalid trials.",
        )
    _clean_path(tmp_path)
    config.search_space = {
        "train_config.optimizer_config.name": SearchSpace(
            categorical_values=["sgd", 0], value_type="int"
        ),
        "train_config.optimizer_config.arguments.lr": SearchSpace(
            value_range=[0, 1], value_type="float"
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
            n_bins=1000 if is_grid_sampler else None,
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
        {"acc": 0} if not is_grid_sampler else None,
        state=TrialState.COMPLETE,
    )
    if config.optim_metrics is None:
        assert_error_msg(
            lambda: s.update_trial_state(
                0,
                {"acc": 0},
                state=TrialState.COMPLETE,
            ),
            f"Can not specificy 'metrics' when not setting 'config.optim_metrics'.",
        )


@pytest.mark.parametrize("search_algo", list(SearchAlgo.__members__.keys()))
def test_state_resume(tmp_path: Path, search_algo, assert_error_msg):
    search_space = {
        "train_config.optimizer_config.arguments.lr": SearchSpace(
            value_range=[0, 1],
            value_type="float",
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


def search_space():
    return {
        "train_config.optimizer_config": SearchSpace(
            subspaces=[
                {"sub_configuration": {"name": "sgd", "arguments": {"lr": 0.1}}},
                {
                    "sub_configuration": {
                        "name": "adam",
                        "arguments": {
                            "lr": {"value_range": (0, 1)},
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
                                    {"value_range": (0, 1)},
                                    {"value_range": (0, 1)},
                                ]
                            },
                            "weight_decay": 0.9,
                        },
                    }
                },
            ]
        ),
        "model_config.b": SearchSpace(value_range=(-10, 10)),
    }


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


def test_mock_run(tmp_path: Path):
    space = search_space()
    dfs = []
    for i in range(10):
        _clean_path(tmp_path)
        config = make_config(space, "tpe")
        s = ExperimentState(tmp_path, config)
        dfs.append(_run_search_algo(s))
    tpe_df = pd.concat(dfs)

    dfs = []
    for i in range(10):
        _clean_path(tmp_path)
        config = make_config(space, "random")
        s = ExperimentState(tmp_path, config)
        dfs.append(_run_search_algo(s))
    rand_df = pd.concat(dfs)
    loss_tpe = _get_top_n(tpe_df)
    loss_rand = _get_top_n(rand_df)
    assert abs(loss_tpe - loss_rand) > 1e-05


if __name__ == "__main__":
    from tests.conftest import _assert_error_msg

    tmp_path = Path("/tmp/state")
    # _clean_path(tmp_path)
    # test_mock_run(tmp_path)
    # for search_algo in list(SearchAlgo.__members__.keys()):
    for search_algo in ["tpe"]:
        # _clean_path(tmp_path)
        # test_state(tmp_path, search_algo)

        # _clean_path(tmp_path)
        # test_sample_limits(tmp_path, search_algo)

        _clean_path(tmp_path)
        test_state_resume(tmp_path, search_algo, _assert_error_msg)

from pathlib import Path
import tempfile
from collections import OrderedDict

import numpy as np


from ablator import ModelConfig, OptimizerConfig, RunConfig, TrainConfig
from ablator.main.configs import ParallelConfig, SearchSpace
from ablator.main.state import ExperimentState, TrialState, parse_metrics
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


def assert_error_msg(fn, error_msg):
    try:
        fn()
        assert False
    except Exception as excp:
        if not error_msg == str(excp):
            raise excp


def capture_output(fn):
    out = io.StringIO()

    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        fn()

    return out.getvalue(), err.getvalue()


def test_state(tmp_path: Path):
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
        "Can not specify value_range and categorical_values for SearchSpace.",
    )
    search_space = {"some_var": SearchSpace(value_range=[0, 1], value_type="float")}
    config = ParallelConfig(
        train_config=train_config,
        model_config=ModelConfig(),
        verbose="silent",
        device="cpu",
        amp=False,
        search_space=search_space,
        optim_metrics={"acc": "max"},
        total_trials=100,
        concurrent_trials=100,
        gpu_mb_per_experiment=0,
        cpus_per_experiment=0.1,
    )
    default_vals = config.make_dict(config.annotations, flatten=True)
    with tempfile.TemporaryDirectory() as fp:
        assert_error_msg(
            lambda: ExperimentState(Path(fp), config),
            f"SearchSpace parameter some_var was not found in the configuration {sorted(list(default_vals.keys()))}.",
        )

    search_space = {
        "train_config.optimizer_config.name": SearchSpace(
            categorical_values=[0], value_type="int"
        )
    }
    config = ParallelConfig(
        train_config=train_config,
        model_config=ModelConfig(),
        verbose="silent",
        device="cpu",
        amp=False,
        search_space=search_space,
        optim_metrics={"acc": "max"},
        total_trials=100,
        concurrent_trials=100,
        ignore_invalid_params=False,
        gpu_mb_per_experiment=0,
        cpus_per_experiment=0.1,
    )
    with tempfile.TemporaryDirectory() as fp:
        assert_error_msg(
            lambda: ExperimentState(Path(fp), config),
            "Invalid trial parameters {'train_config.optimizer_config.name': '0'}",
        )

    config = ParallelConfig(
        train_config=train_config,
        model_config=ModelConfig(),
        verbose="silent",
        device="cpu",
        amp=False,
        search_space=search_space,
        optim_metrics={"acc": "max"},
        total_trials=1,
        concurrent_trials=100,
        ignore_invalid_params=True,
        gpu_mb_per_experiment=0,
        cpus_per_experiment=0.1,
    )
    with tempfile.TemporaryDirectory() as fp:
        assert_error_msg(
            lambda: ExperimentState(Path(fp), config),
            "Reached maximum limit of misconfigured trials. 10\nFound 0 duplicate and 11 invalid trials.",
        )

    with tempfile.TemporaryDirectory() as fp:
        config.search_space = {
            "train_config.optimizer_config.arguments.lr": SearchSpace(
                categorical_values=[0, 1], value_type="int"
            )
        }
        config.total_trials = 3
        assert_error_msg(
            lambda: ExperimentState(Path(fp), config),
            "Reached maximum limit of misconfigured trials. 30\nFound 31 duplicate and 0 invalid trials.",
        )

    with tempfile.TemporaryDirectory() as fp:
        config.search_space = {
            "train_config.optimizer_config.name": SearchSpace(
                categorical_values=["sgd", 0], value_type="int"
            ),
            "train_config.optimizer_config.arguments.lr": SearchSpace(
                value_range=[0, 1], value_type="float"
            ),
        }
        config.total_trials = 10
        out, err = capture_output(
            lambda: ExperimentState(Path(fp), config, logger=FileLogger())
        )
        assert (
            "ignoring: {'train_config.optimizer_config.name': '0'," in out
            and len(err) == 0
        )

    with tempfile.TemporaryDirectory() as fp:
        config.search_space = {
            "train_config.optimizer_config.arguments.lr": SearchSpace(
                value_range=[0, 1], value_type="float"
            ),
        }
        config.total_trials = 10
        config.concurrent_trials = 1
        s = ExperimentState(Path(fp), config)
        assert len(s.all_trials) == 1
        s.sample_trials(3)
        assert len(s.all_trials) == 4
        s.update_trial_state(
            s.all_trials[0].uid, None, state=TrialState.PRUNED_DUPLICATE
        )
        assert len(s.pruned_duplicate_trials) == 1
        assert len(s.pruned_errored_trials) == 0
        assert len(s.all_trials) == 3

        s.update_trial_state(s.all_trials[0].uid, None, state=TrialState.PRUNED_INVALID)
        assert len(s.pruned_errored_trials) == 1
        assert len(s.pruned_duplicate_trials) == 1
        assert len(s.all_trials) == 2
        assert s.n_trials_remaining == 8
        s.sample_trials(3)
        assert s.n_trials_remaining == 5
        some_uid = s.all_trials[0].uid
        for state, trial_attr_name in zip(
            [
                TrialState.RUNNING,
                #TrialState.WAITING,
                TrialState.FAIL,
                TrialState.COMPLETE,
            ],
            ["running_trials", "failed_trials", "complete_trials"],
        ):
            val_before = len(getattr(s, trial_attr_name))
            s.update_trial_state(
                some_uid,
                None if state != TrialState.COMPLETE else {"acc": 0},
                state=state,
            )
            val_after = len(getattr(s, trial_attr_name))
            assert val_before + 1 == val_after
        assert_error_msg(
            lambda: ExperimentState(Path(fp), config),
            f"{s.experiment_dir.joinpath(f'{config.uid}_optuna.db')} exists. Please remove before starting a study.",
        )
        prev_trials = s.all_trials_uid
        s = ExperimentState(Path(fp), config, resume=True)
        assert prev_trials == s.all_trials_uid

    with tempfile.TemporaryDirectory() as fp:
        search_space = {
            # "train_config.optimizer_config.name": SearchSpace(
            #     categorical_values=["sgd"], value_type="int"
            # ),
            "train_config.optimizer_config.arguments.lr": SearchSpace(
                value_range=[0, 1], value_type="float"
            ),
        }
        config = ParallelConfig(
            train_config=train_config,
            model_config=ModelConfig(),
            verbose="silent",
            device="cpu",
            amp=False,
            search_space=search_space,
            optim_metrics={"acc": "max"},
            total_trials=1,
            concurrent_trials=100,
            ignore_invalid_params=True,
            gpu_mb_per_experiment=0,
            cpus_per_experiment=0.1,
        )
        config.to_dict()
        config.total_trials = 10
        config.concurrent_trials = 5
        s = ExperimentState(Path(fp), config)
        some_uid = s.all_trials[0].uid
        assert_error_msg(
            lambda: s.update_trial_state(some_uid, {"aaa": 0}),
            "Different specified metric directions `{'acc'}` and `{'aaa'}`",
        )
        for trial, some_uid in zip(
            [{"acc": np.nan}, {"acc": 0}, {"acc": None}], s.all_trials_uid
        ):
            s.update_trial_state(some_uid, trial, TrialState.COMPLETE)

        assert_error_msg(
            lambda: s.update_trial_state(some_uid, {"acc": "A"}, TrialState.COMPLETE),
            "ufunc 'isfinite' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''",
        )

def test_parse_metrics():
    metric_directions = OrderedDict([('a', 'max'), ('b', 'max')])
    metrics = {'a': 1, 'b': np.nan}
    parsed = parse_metrics(metric_directions, metrics)
    assert parsed == OrderedDict([('a', 1.0), ('b', -np.inf)])

if __name__ == "__main__":
    import shutil

    tmp_path = Path("/tmp/state")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir()
    # test_state(tmp_path)
    test_parse_metrics()

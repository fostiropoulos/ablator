from pathlib import Path

import numpy as np
import pytest

from ablator import PlotAnalysis
from ablator.analysis.results import Results
from ablator.main.mp import ParallelTrainer
from ablator.main.state.store import TrialState


@pytest.mark.order(0)
@pytest.mark.mp
def test_all(
    tmp_path: Path,
    assert_error_msg,
    working_dir,
    error_wrapper,
    make_config,
):
    # We set up an experiment of 9 trials where some of them end up crashing in purpose
    # We then check whether the error trials meets the expected number
    # whether the analysis artifacts agree with that. finally whether we can resume the experiment.
    # The test represents an end-to-end use-case of ablator.
    n_trials = 9
    config = make_config(tmp_path, search_space_limit=n_trials)
    config.experiment_dir = tmp_path.joinpath("ablator")
    config.total_trials = n_trials
    ablator = ParallelTrainer(wrapper=error_wrapper, run_config=config)
    ablator.launch(working_dir)

    complete_configs = ablator.experiment_state.get_trial_configs_by_state(
        TrialState.COMPLETE
    )
    failed_configs = ablator.experiment_state.get_trial_configs_by_state(
        TrialState.FAIL
    )
    lrs = np.array(
        [c.train_config.optimizer_config.arguments.lr for c in complete_configs]
    )
    bad_lrs = np.array(
        [c.train_config.optimizer_config.arguments.lr for c in failed_configs]
    )
    config = ablator.run_config
    n_trials = config.total_trials
    LR_ERROR_LIMIT = config.model_config.lr_error_limit
    n_complete = np.sum(
        np.linspace(0, 19, int(n_trials**0.5)) < LR_ERROR_LIMIT
    ) * int(n_trials**0.5)
    n_failed = np.sum(np.linspace(0, 19, int(n_trials**0.5)) > LR_ERROR_LIMIT) * int(
        n_trials**0.5
    )
    assert len(complete_configs) == n_complete
    assert len(failed_configs) == n_failed
    assert (lrs < LR_ERROR_LIMIT).all()
    assert (bad_lrs > LR_ERROR_LIMIT).all()
    assert len(failed_configs) == n_trials - n_complete
    msg = assert_error_msg(
        lambda: ablator._init_state(working_dir),
    )
    assert (
        "Experiment Directory " in msg
        and config.experiment_dir in msg
        and "exists" in msg
    )

    prev_trials = len(ablator.experiment_state.valid_trials())
    ablator.launch(working_dir, resume=True)
    assert len(ablator.experiment_state.valid_trials()) == prev_trials
    ablator.run_config.total_trials += 4
    ablator.launch(working_dir, resume=True)
    assert (len(ablator.experiment_state.valid_trials()) != prev_trials) and (
        len(ablator.experiment_state.valid_trials()) == ablator.total_trials
    )
    config = ablator.run_config
    results = Results(config, ablator.experiment_dir)

    # we add `ray_cluster` fixture so that it is scheduled with the slow tests.
    PlotAnalysis(results, optim_metrics={"val_loss": "min"})
    categorical_name_remap = {
        "model_config.mock_param": "Some Parameter",
    }
    numerical_name_remap = {
        "train_config.optimizer_config.arguments.lr": "Learning Rate",
    }
    analysis = PlotAnalysis(
        results,
        save_dir=tmp_path.as_posix(),
        cache=True,
        optim_metrics={"val_loss": "min"},
    )
    attribute_name_remap = {**categorical_name_remap, **numerical_name_remap}
    analysis.make_figures(
        metric_name_remap={
            "val_loss": "Val. Loss",
        },
        attribute_name_remap=attribute_name_remap,
    )
    assert all(
        tmp_path.joinpath("violinplot", "val_loss", f"{file_name}.png").exists()
        for file_name in categorical_name_remap
    )

    assert all(
        tmp_path.joinpath("linearplot", "val_loss", f"{file_name}.png").exists()
        for file_name in numerical_name_remap
    )


if __name__ == "__main__":
    from tests.conftest import run_tests_local
    from tests.test_plugins.model import (
        WORKING_DIR,
        MyErrorCustomModel,
        TestWrapper,
        _make_config,
    )

    error_wrapper = TestWrapper(MyErrorCustomModel)

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]

    kwargs = {
        "error_wrapper": error_wrapper,
        "working_dir": WORKING_DIR,
        "make_config": _make_config,
    }
    run_tests_local(test_fns, kwargs)

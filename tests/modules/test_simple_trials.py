import tempfile
from pathlib import Path
import pytest

from ablator import ModelConfig, OptimizerConfig, TrainConfig
from ablator.main.configs import ParallelConfig, SearchSpace
from ablator.main.state import ExperimentState

optimizer_config = OptimizerConfig(name="sgd", arguments={"lr": 0.1})
train_config = TrainConfig(
    dataset="test",
    batch_size=128,
    epochs=2,
    optimizer_config=optimizer_config,
    scheduler_config=None,
)

def test_sample_trials():
    with tempfile.TemporaryDirectory() as fp:
        search_space = {
            "train_config.optimizer_config.arguments.lr": SearchSpace(
                value_range=[0, 1], value_type="float"
            ),
        }
        config = ParallelConfig(
            train_config=train_config,
            model_config=ModelConfig(),
            verbose="silent",
            device="gpu",
            amp=False,
            search_space=search_space,
            optim_metrics={"acc": "max"},
            total_trials=10,
            concurrent_trials=1,
            ignore_invalid_params=True,
        )
        experiment_state = ExperimentState(Path(fp), config)

        # Verify the initial number of trials.
        initial_trial_count = len(experiment_state.all_trials)
        assert initial_trial_count == 1

        # Sample more trials.
        experiment_state.sample_trials(5)

        # Verify that the correct number of trials were added.
        final_trial_count = len(experiment_state.all_trials)
        assert final_trial_count == initial_trial_count + 5, "Incorrect number of trials added by sample_trials."
        print("Successful!")


if __name__ == "__main__":
    pass
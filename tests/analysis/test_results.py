import io
import os
from pathlib import Path
import pandas as pd
import pytest
from unittest.mock import patch
from ablator.analysis.results import Results, read_result
from ablator.config.proto import RunConfig
from ablator.config.main import ConfigBase
from ablator.analysis.main import _parse_results
from ablator.config.mp import ParallelConfig


def test_missing_config_raises_file_not_found_error(tmp_path):
    # Create a temporary directory but do not create the default_config.yaml file inside it
    experiment_dir = tmp_path / "fake_experiment"
    experiment_dir.mkdir()

    # Assert that initializing Results raises a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        results = Results(config=ParallelConfig, experiment_dir=experiment_dir)


def test_parse_with_results(parallel_config, mock_experiment_directory):
    results = Results(parallel_config, mock_experiment_directory)
    df, cats, nums, metrics = _parse_results(results)

    # Basic assertions to ensure data is returned
    assert not df.empty
    assert isinstance(cats, list)
    assert isinstance(nums, list)
    assert isinstance(metrics, dict)


def test_results_init_and_make_data(parallel_config, mock_experiment_directory):
    # Create an instance of the Results class

    results = Results(parallel_config, mock_experiment_directory)

    # Assert the attributes after initialization
    assert results.experiment_dir == mock_experiment_directory
    assert isinstance(results.config, parallel_config)
    assert isinstance(results.metric_map, dict)
    assert isinstance(results.data, pd.DataFrame)
    assert isinstance(results.config_attrs, list)
    assert isinstance(results.search_space, dict)
    assert isinstance(results.numerical_attributes, list)
    assert isinstance(results.categorical_attributes, list)


def test_results_config_not_parallel():
    # Create a test configuration that is not a subclass of ParallelConfig
    class TestConfig(ConfigBase):
        pass

    # Create a temporary experiment directory
    experiment_dir = "./tmp_experiment"
    os.makedirs(experiment_dir, exist_ok=True)
    try:
        results = Results(config=TestConfig, experiment_dir=experiment_dir)
    except ValueError as e:
        assert str(e) == "Configuration must be subclassed from ``ParallelConfig``. "
    else:
        raise AssertionError("ValueError was not raised")


def test_results_single_trial_config():
    # Create a test configuration that is a single-trial config
    class TestConfig(RunConfig):
        pass

    # Create a temporary experiment directory
    experiment_dir = "./tmp_experiment"
    os.makedirs(experiment_dir, exist_ok=True)

    # Check if ValueError is raised
    try:
        results = Results(config=TestConfig, experiment_dir=experiment_dir)
    except ValueError as e:
        assert (
            str(e)
            == "Provided a ``RunConfig`` used for a single-trial. Analysis "
            "is not meaningful for a single trial. Please provide a ``ParallelConfig``."
        )
    else:
        raise AssertionError("ValueError was not raised")


def test_metric_names(parallel_config, mock_experiment_directory):
    # Given the mock data for metric_map:
    mock_metric_map = {
        "key1": "val_loss",
        "key2": "train_loss",
        "key3": "val_acc",
        "key4": "train_acc",
    }
    results = Results(parallel_config, mock_experiment_directory)

    # Use patch.object() to mock the metric_map attribute of the results instance
    with patch.object(results, "metric_map", mock_metric_map):
        expected_metric_names = ["val_loss", "train_loss", "val_acc", "train_acc"]
        assert results.metric_names == expected_metric_names


def test_read_result_no_results_found():
    # Assuming experiment_dir is a non-existent directory
    experiment_dir = Path("nonexistent_directory")
    config_type = None

    # Use pytest.raises to check if the RuntimeError is raised
    with pytest.raises(RuntimeError) as exc_info:
        Results.read_results(config_type, experiment_dir)

    # Check the error message
    assert str(exc_info.value) == f"No results found in {experiment_dir}"


def test_assert_cat_attributes(
    capture_logger, capture_output, parallel_config, mock_experiment_directory
):
    out: io.StringIO = capture_logger()
    results = Results(parallel_config, mock_experiment_directory)

    # Creating a mock dataframe
    data = {
        "attr1": ["X", "X", "Y", "Z"],
        "attr2": ["X", "X", "X", "X"],
        "attr3": ["X", "Y", "Z", "Z"],
    }
    mock_df = pd.DataFrame(data)

    # Results object

    # Patching the data attribute of the results instance with our mock dataframe
    with patch.object(results, "data", mock_df):
        # Check for the expected warning for attr1 (which is balanced, so no warnings)
        stdout, stderr = capture_output(
            lambda: results._assert_cat_attributes(["attr2"])
        )
        expected_warning = "Imbalanced trials for attr attr2"
        assert expected_warning in out.getvalue()


def test_read_result_exception_handling(tmp_path):
    # Create dummy JSON and YAML files
    json_content = '[{"train_loss": 10.35, "val_loss": null, "current_epoch": 1}]'
    config_content = (
        'experiment_dir: "\\tmp\\results\\experiment_8925_9991"\ndevice: cpu'
    )

    json_file = tmp_path / "results.json"
    config_file = tmp_path / "config.yaml"

    json_file.write_text(json_content)
    config_file.write_text(config_content)

    # Mock the ConfigBase.load method to raise an exception
    with patch.object(
        ConfigBase, "load", side_effect=Exception("Mocked exception for testing")
    ):
        result = read_result(ConfigBase, json_file)
        assert result is None


if __name__ == "__main__":
    from tests.conftest import run_tests_local

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]
    run_tests_local(test_fns)

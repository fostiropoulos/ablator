import numpy as np
import pytest
import torch

import ablator.utils.base as base


def test_getitem_dummy():
    # check that the returned value is the same as the original dummy object.
    dummy = base.Dummy()
    assert dummy[0] is dummy, "Expected Dummy.__getitem__ to return self"
    assert dummy["key"] is dummy, "Expected Dummy.__getitem__ to return self"
    assert dummy[1, 2, 3] is dummy, "Expected Dummy.__getitem__ to return self"


def test_set_seed():
    seed = 42
    base.set_seed(seed)
    # Test if the seed is set successfully
    assert np.random.randint(1000) == np.random.RandomState(seed).randint(1000)


# Test parse_device if returns the correct device
def test_parse_device():
    assert base.parse_device("cpu") == "cpu"
    if torch.cuda.is_available():
        gpu_number = torch.cuda.device_count()
        for i in range(gpu_number):
            assert base.parse_device(i) == f"cuda:{i}"
            assert base.parse_device(f"cuda:{i}") == f"cuda:{i}"

        with pytest.raises(
            AssertionError,
            match=f"gpu cuda:{gpu_number + 1} does not exist on this machine",
        ):
            base.parse_device(gpu_number + 1)

        assert base.parse_device("cuda") == "cuda"
        assert base.parse_device(["cuda", "cpu"]) == "cuda"

    with pytest.raises(ValueError):
        base.parse_device("invalid")


def test_is_oom_exception():
    # Test that the function returns True for error messages indicating out of memory
    oom_errors = [
        "CUDA out of memory",
        "CUBLAS_STATUS_ALLOC_FAILED",
        "CUDA error: out of memory",
    ]
    for error in oom_errors:
        assert (
            base.is_oom_exception(RuntimeError(error)) is True
        ), f"Expected True for error message '{error}'"

    # Test that the function returns False for error messages not indicating out of memory
    non_oom_errors = [
        "Some other error",
        "CUDA error: unspecified launch failure",
    ]
    for error in non_oom_errors:
        assert (
            base.is_oom_exception(RuntimeError(error)) is False
        ), f"Expected False for error message '{error}'"


# Test If the apply_lambda_to_iter function correctly applies a lambda function
# (in this case, a function to square a number) to each element in a list
def test_apply_lambda_to_iter():
    input_list = [1, 2, 3, 4, 5]
    expected_output = [1, 4, 9, 16, 25]
    output = base.apply_lambda_to_iter(input_list, lambda x: x**2)
    assert output == expected_output
    values = dict(zip(input_list, input_list))

    output = base.apply_lambda_to_iter(values, lambda x: x**2)
    assert list(output.values()) == expected_output
    assert list(output.keys()) == input_list


def test_num_format_with_non_numeric_value():
    value = "not_a_number"
    # Test `num_format` with non-numeric value
    assert (
        base.num_format(value) == value
    ), "Non-numeric values should be returned as is"


if __name__ == "__main__":
    from tests.conftest import run_tests_local

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]
    run_tests_local(test_fns)

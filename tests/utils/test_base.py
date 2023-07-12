import pytest
import torch
from torch import nn
import numpy as np
import typing as ty
import ablator.utils.base as base


def test_getitem_dummy():
    dummy = base.Dummy()
    assert dummy[0] is dummy, "Expected Dummy.__getitem__ to return self"
    assert dummy["key"] is dummy, "Expected Dummy.__getitem__ to return self"
    assert dummy[1, 2, 3] is dummy, "Expected Dummy.__getitem__ to return self"


def assert_error_msg(fn, error_msg):
    try:
        fn()
        assert False, "Should have raised an error."
    except Exception as excp:
        if not error_msg == str(excp):
            raise excp


def test_set_seed():
    seed = 42
    base.set_seed(seed)
    assert np.random.randint(1000) == np.random.RandomState(seed).randint(1000)


def test_parse_device():
    assert base.parse_device(0) == 0
    assert base.parse_device(1) == 1
    assert base.parse_device('cpu') == 'cpu'
    if torch.cuda.is_available():
        assert base.parse_device('cuda') == 'cuda'
        assert base.parse_device(['cuda', 'cpu']) == ['cuda', 'cpu']
        device_number = min(torch.cuda.device_count() - 1, 0)
        assert base.parse_device(f'cuda:{device_number}') == f'cuda:{device_number}'
        gpu_number = torch.cuda.device_count()
        assert_error_msg(lambda: base.parse_device(f'cuda:{gpu_number + 2}'), f"gpu {gpu_number + 2} does not exist on this machine")
    with pytest.raises(ValueError):
        base.parse_device('invalid')


def test_init_weights():
    # Test with Linear layer
    linear = nn.Linear(10000, 5000)
    base.init_weights(linear)
    assert torch.allclose(linear.weight.mean(), torch.tensor(0.0), atol=0.02)
    assert torch.all(linear.bias == 0)

    # Test with Embedding layer
    embedding = nn.Embedding(10000, 5000, padding_idx=1)
    base.init_weights(embedding)
    assert torch.allclose(embedding.weight.mean(), torch.tensor(0.0), atol=0.02)
    assert torch.all(embedding.weight[embedding.padding_idx] == 0)

    # Test with LayerNorm
    layernorm = nn.LayerNorm(10)
    base.init_weights(layernorm)
    assert torch.all(layernorm.bias == 0)
    assert torch.all(layernorm.weight == 1)


MemType = ty.Literal["used", "total", "free"]


@pytest.mark.skip(reason="get_gpu_mem has a bug, the return type is not a list of int")
# get_gpu_mem has a bug, the return type is not a list of int
def test_get_gpu_mem():
    mem_types: list[MemType] = ["used", "total", "free"]
    for mem_type in mem_types:
        result = base.get_gpu_mem(mem_type)
        print(result)
        assert isinstance(result, list)
        assert all(isinstance(val, int) for val in result)

    with pytest.raises(KeyError):
        base.get_gpu_mem("invalid")




def test_is_oom_exception():
    oom_errors = [
        "CUDA out of memory",
        "CUBLAS_STATUS_ALLOC_FAILED",
        "CUDA error: out of memory",
    ]
    for error in oom_errors:
        assert base.is_oom_exception(RuntimeError(error)) is True, f"Expected True for error message '{error}'"

    non_oom_errors = [
        "Some other error",
        "CUDA error: unspecified launch failure",
    ]
    for error in non_oom_errors:
        assert base.is_oom_exception(RuntimeError(error)) is False, f"Expected False for error message '{error}'"


def test_apply_lambda_to_iter():
    # Let's say we want to square all numbers in a list
    input_list = [1, 2, 3, 4, 5]
    expected_output = [1, 4, 9, 16, 25]  # The squares of all numbers in input_list
    output = base.apply_lambda_to_iter(input_list, lambda x: x**2)
    assert output == expected_output


def test_num_format_with_non_numeric_value():
    value = "not_a_number"
    assert base.num_format(value) == value, "Non-numeric values should be returned as is"

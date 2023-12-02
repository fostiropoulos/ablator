import typing as ty

import pytest
import torch

from ablator.utils._nvml import get_gpu_mem

MemType = ty.Literal["used", "total", "free"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_get_gpu_mem():
    mem_types: list[MemType] = ["used", "total", "free"]
    for mem_type in mem_types:
        result = get_gpu_mem(mem_type)
        assert isinstance(result, dict)
        assert all(isinstance(val, int) for val in result)

    with pytest.raises(KeyError):
        get_gpu_mem("invalid")

import copy
import gc
import os
from pathlib import Path
import time
import uuid
import mock

import numpy as np
import pytest
import ray
import torch

from ablator.mp.gpu_manager import GPUManager, _get_gpus, unlock_gpu, wait_get_gpu
from ablator.utils.base import _get_gpu_info, get_cuda_processes, get_gpu_mem

GPU_MEM_UTIL = 100


n_gpus = min(torch.cuda.device_count(), 2)


def _is_p_init(cuda_processes):
    return any([len(p) == 0 for p in cuda_processes.values()])


def _running_processes(gpus, process_prefix):
    return any(
        [
            len([p for p in ps if process_prefix in p.process_name]) > 0
            for ps in gpus.values()
        ]
    )


def monitor_gpu_usage(
    current_cuda_ps, process_prefix="test_gpu_manager_"
) -> dict[int, list[int]]:
    cntr = 0
    cuda_ps: dict[int, list[int]] = {}

    while (
        len(cuda_ps) == 0
        or _is_p_init(cuda_ps)
        or _running_processes(current_cuda_ps, process_prefix)
    ):
        for k, ps in current_cuda_ps.items():
            if k not in cuda_ps:
                cuda_ps[k] = []
            for p in ps:
                if p.pid not in cuda_ps[k] and process_prefix in p.process_name:
                    cuda_ps[k].append(p.pid)

        cntr += 1
        if cntr > 30 / 0.4:
            assert False, "Could not terminate `monitor_gpu_usage` on time."
        time.sleep(0.1)
        current_cuda_ps = get_cuda_processes()
    return cuda_ps


def _make_futures(n_gpus: int, manager: GPUManager, remote_fn, process_prefix):
    futures = [
        ray.remote(
            num_gpus=0.001,
            num_cpus=0.001,
            max_calls=1,
            max_retries=0,
        )(remote_fn)
        .options(name=f"{process_prefix}_{i}")
        .remote(manager)
        for i in range(n_gpus)
    ]
    return futures


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The test is meant to evaluate allocation of cuda memory.",
)
def test_gpu_alignment(gpu_manager: GPUManager):
    """
    test that the GPU index numbers are aligned between torch to device and
    get_gpu_mem and get_cuda_processes.
    """

    init_cuda_ps = get_cuda_processes()
    least_busy_gpu = wait_get_gpu(gpu_manager, GPU_MEM_UTIL)
    unlock_gpu(gpu_manager, least_busy_gpu)
    t = torch.ones(10000, 10000).to(f"cuda:{least_busy_gpu}")
    new_lbgpu = wait_get_gpu(gpu_manager, GPU_MEM_UTIL)
    unlock_gpu(gpu_manager, new_lbgpu)
    assert (
        new_lbgpu != least_busy_gpu
    ), "Can not run test robustly when several processess are also running on cuda."
    mem = get_gpu_mem("used")
    assert np.argmin([mem[k] for k in init_cuda_ps]) == new_lbgpu
    cuda_ps = get_cuda_processes()
    assert len(cuda_ps[least_busy_gpu]) > len(init_cuda_ps[least_busy_gpu])
    gc.collect()
    torch.cuda.empty_cache()
    assert True


@pytest.mark.skipif(
    n_gpus < 2,
    reason="The test is meant to evaluate allocation of cuda memory.",
)
def test_lock_unlock(assert_error_msg, gpu_manager):
    """
    test that requesting and unlocking resources works
    as expected.
    """

    [unlock_gpu(gpu_manager, i) for i in range(n_gpus)]
    gpus = [wait_get_gpu(gpu_manager, 100, 2) for i in range(n_gpus)]

    assert sorted(gpus) == list(range(n_gpus))
    msg = assert_error_msg(lambda: wait_get_gpu(gpu_manager, 100, 2))
    assert "No available GPU." in msg
    unlock_gpu(gpu_manager, 0)
    assert wait_get_gpu(gpu_manager, 100, 2) == 0
    msg = assert_error_msg(lambda: wait_get_gpu(gpu_manager, 100, 2))
    assert msg == "No available GPU."
    rand_unlock = np.random.choice(n_gpus)
    unlock_gpu(gpu_manager, rand_unlock)
    assert wait_get_gpu(gpu_manager, 100, 2) == rand_unlock
    [unlock_gpu(gpu_manager, i) for i in range(n_gpus)]


@pytest.mark.skipif(
    n_gpus < 2,
    reason="The test is meant to evaluate allocation of cuda memory.",
)
def test_allocation(gpu_manager, remote_fn, n_remotes=n_gpus * 2):
    """
    Test that remotes are allocated to a unique GPU
    """
    assert n_remotes > n_gpus, "The test is only valid for n_remotes > n_gpus."
    process_prefix = str(uuid.uuid4())[:4]
    n_remotes = n_remotes // n_gpus

    current_cuda_ps = get_cuda_processes()
    futures = _make_futures(n_remotes, gpu_manager, remote_fn, process_prefix)
    cuda_ps = monitor_gpu_usage(current_cuda_ps, process_prefix)
    assert len(cuda_ps) == n_gpus
    remote_dist = [len(l) for l in cuda_ps.values()]
    # make sure that all the remotes executed on all devices
    assert sum(remote_dist) == n_remotes
    # make sure that all remotes were evenly distributed
    assert all([v == remote_dist[0] for v in remote_dist])
    gpus = ray.get(futures)
    assert sorted(gpus) == list(range(n_gpus))


if __name__ == "__main__":
    from tests.conftest import _assert_error_msg
    from tests.ray_models.model import _remote_fn

    with mock.patch("ablator.utils.base._get_gpu_info", lambda: _get_gpu_info()[:2]):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

        gpu_manager = GPUManager.remote([0, 1])
        test_allocation(gpu_manager, _remote_fn)
        # gpu_manager = GPUManager.remote([0, 1])
        # test_lock_unlock(_assert_error_msg, gpu_manager)
        gpu_manager = GPUManager.remote([0, 1])
        test_gpu_alignment(gpu_manager)

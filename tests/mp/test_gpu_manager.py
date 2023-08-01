import copy
import functools
import gc
import os
import shutil
import time
import uuid
from pathlib import Path

import mock
import numpy as np
import pytest
import ray
import torch

from ablator.modules.loggers.file import FileLogger
from ablator.mp import gpu_manager
from ablator.mp.gpu_manager import GPUManager, unlock_gpu, wait_get_gpu
from ablator.mp.train_remote import _apply_unlock_hook, train_main_remote
from ablator.utils.base import _get_gpu_info, get_cuda_processes, get_gpu_mem

wait_get_gpu = functools.partial(wait_get_gpu, expected_util_mb=100, max_timeouts=2)

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


def _make_remote(remote_fn, name, gpu_manager, kwargs):
    kwargs = copy.deepcopy(kwargs)
    if "gpu_id" not in kwargs:
        kwargs["gpu_id"] = wait_get_gpu(gpu_manager, process_name=name)
    return (
        ray.remote(
            num_gpus=0.001,
            num_cpus=0.001,
            max_calls=1,
            max_retries=0,
        )(remote_fn)
        .options(name=name)
        .remote(**kwargs)
    )


def _make_futures(n_gpus: int, manager: GPUManager, remote_fn, process_prefix):
    kwargs = dict(gpu_manager=manager)
    futures = [
        _make_remote(remote_fn, f"{process_prefix}_{i}", manager, kwargs)
        for i in range(n_gpus)
    ]
    return futures


def _make_future_gpu(remote_fn, name, gpu_manager, kwargs):
    return ray.get(_make_remote(remote_fn, name, gpu_manager, kwargs))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The test is meant to evaluate allocation of cuda memory.",
)
def test_gpu_alignment(very_patient_gpu_manager: GPUManager):
    """
    test that the GPU index numbers are aligned between torch to device and
    get_gpu_mem and get_cuda_processes.
    """

    init_cuda_ps = get_cuda_processes()
    fn_name = "test_gpu_alignment"
    least_busy_gpu = wait_get_gpu(very_patient_gpu_manager, process_name=fn_name)
    unlock_gpu(very_patient_gpu_manager, least_busy_gpu)
    t = torch.ones(10000, 10000).to(f"cuda:{least_busy_gpu}")
    new_lbgpu = wait_get_gpu(very_patient_gpu_manager, process_name=fn_name)
    unlock_gpu(very_patient_gpu_manager, new_lbgpu)
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
def test_lock_unlock(assert_error_msg, very_patient_gpu_manager, remote_fn):
    """
    test that requesting and unlocking resources works
    as expected.
    """

    fn_name = "test_lock_unlock"

    gpus = [
        wait_get_gpu(very_patient_gpu_manager, process_name=fn_name)
        for i in range(n_gpus)
    ]
    assert sorted(gpus) == list(range(n_gpus))
    msg = assert_error_msg(
        lambda: wait_get_gpu(very_patient_gpu_manager, process_name=fn_name)
    )
    assert "No available GPU." in msg
    unlock_gpu(very_patient_gpu_manager, 0)
    assert wait_get_gpu(very_patient_gpu_manager, process_name=fn_name) == 0
    msg = assert_error_msg(
        lambda: wait_get_gpu(very_patient_gpu_manager, process_name=fn_name)
    )
    assert msg == "No available GPU."
    rand_unlock = np.random.choice(n_gpus)
    unlock_gpu(very_patient_gpu_manager, rand_unlock)
    assert wait_get_gpu(very_patient_gpu_manager, process_name=fn_name) == rand_unlock
    [unlock_gpu(very_patient_gpu_manager, i) for i in range(n_gpus)]


@pytest.mark.skipif(
    n_gpus < 2,
    reason="The test is meant to evaluate allocation of cuda memory.",
)
def test_lock_unlock_by_id(very_patient_gpu_manager, remote_fn):
    # test unlocked by process name
    fn_name = "test_lock_unlock_by_process_name"
    gpu = wait_get_gpu(very_patient_gpu_manager, process_name=f"{fn_name}_0")
    gpu_two = wait_get_gpu(very_patient_gpu_manager, process_name=f"{fn_name}_1")
    assert gpu != gpu_two
    _make_future_gpu(
        remote_fn,
        f"{fn_name}_0",
        very_patient_gpu_manager,
        dict(gpu_id=gpu, gpu_manager=very_patient_gpu_manager),
    )
    gpus = ray.get(very_patient_gpu_manager._gpus.remote())

    assert gpus[gpu]._locking_process == None
    gpu_three = wait_get_gpu(very_patient_gpu_manager, process_name=f"{fn_name}_0")
    assert gpu == gpu_three

    gpus = ray.get(very_patient_gpu_manager._gpus.remote())
    assert gpus[gpu]._locking_process == f"{fn_name}_0"
    assert gpus[gpu_two]._locking_process == f"{fn_name}_1"
    assert gpus[gpu].lock_timeout == 10000000
    assert gpus[gpu_two].lock_timeout == 10000000


@pytest.mark.skipif(
    n_gpus < 2,
    reason="The test is meant to evaluate allocation of cuda memory.",
)
def test_lock_unlock_by_time_limit(gpu_manager):
    # impatient gpu manager
    fn_name = "test_lock_unlock_by_time_limit"
    gpu_one = wait_get_gpu(gpu_manager, process_name=fn_name)
    time.sleep(5)
    gpu_two = wait_get_gpu(gpu_manager, process_name=fn_name)
    gpu_three = wait_get_gpu(gpu_manager, process_name=fn_name)
    assert gpu_one == gpu_two
    assert gpu_two != gpu_three


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
    a = torch.randn(100).to("cuda")
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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Cuda is required to perform this test.",
)
def test_lock_unlock_hook(
    tmp_path: Path, wrapper, make_config, very_patient_gpu_manager
):
    # test unlocked by process name

    config = make_config(tmp_path.joinpath("mock_model"))
    uid = "some_uid"
    fault_tollerant = True
    crash_exceptions_types = None
    resume = False
    clean_reset = False
    progress_bar = None
    logger_path = tmp_path.joinpath("log.log")
    mp_logger = FileLogger(logger_path, verbose=True)
    gpu = wait_get_gpu(very_patient_gpu_manager)
    config.device = f"cuda:{gpu}"
    args = (
        wrapper,
        config,
        mp_logger,
        very_patient_gpu_manager,
        uid,
        fault_tollerant,
        crash_exceptions_types,
        resume,
        clean_reset,
        progress_bar,
    )
    _apply_unlock_hook(wrapper, very_patient_gpu_manager, gpu)
    wrapper.train(config, resume=resume, remote_progress_bar=progress_bar)
    assert wrapper._is_locked == False


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Cuda is required to perform this test.",
)
def test_lock_unlock_train_main_remote(
    tmp_path: Path, wrapper, make_config, very_patient_gpu_manager
):
    # test unlocked from within the train_main_remote

    config = make_config(tmp_path.joinpath("mock_model"))
    uid = "some_uid"
    logger_path = tmp_path.joinpath("log.log")
    mp_logger = FileLogger(logger_path, verbose=True)
    wrapper._uid = "train_main_remote_test"
    kwargs = dict(
        model=wrapper,
        run_config=config,
        mp_logger=mp_logger,
        gpu_manager=very_patient_gpu_manager,
        uid=uid,
        fault_tollerant=False,
    )

    gpu = wait_get_gpu(very_patient_gpu_manager, process_name="mock_lock")
    gpu = wait_get_gpu(very_patient_gpu_manager, process_name="mock_lock")
    # we unlock GPU 1 so that it is off-by one and the test
    # is challenging.
    unlock_gpu(very_patient_gpu_manager, 1)

    _make_future_gpu(
        train_main_remote, "train_main_remote_test", very_patient_gpu_manager, kwargs
    )
    gpus = ray.get(very_patient_gpu_manager._gpus.remote())
    gpu_two = wait_get_gpu(
        very_patient_gpu_manager, process_name="test_lock_unlock_train_main_remote"
    )
    assert 0 != gpu_two
    assert gpus[gpu_two]._locking_process == None
    assert gpus[0]._locking_process == "mock_lock"


if __name__ == "__main__":
    from tests.conftest import _assert_error_msg
    from tests.ray_models.model import (
        WORKING_DIR,
        MyCustomModel,
        MyDivCustomModel,
        MyErrorCustomModel,
        TestWrapper,
        _make_config,
        _remote_fn,
    )

    tmp_path = Path("/tmp/test_gpu_manager")
    shutil.rmtree(tmp_path, ignore_errors=True)
    wrapper = TestWrapper(MyErrorCustomModel)
    with mock.patch("ablator.utils.base._get_gpu_info", lambda: _get_gpu_info()[:2]):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

        gpu_manager = GPUManager.remote(60, [0, 1])
        test_gpu_alignment(gpu_manager)
        gpu_manager = GPUManager.remote(60, [0, 1])
        test_allocation(gpu_manager, _remote_fn)

        gpu_manager = GPUManager.remote(60, [0, 1])
        test_lock_unlock(_assert_error_msg, gpu_manager, _remote_fn)
        gpu_manager = GPUManager.remote(60, [0, 1])
        test_lock_unlock_by_id(gpu_manager, _remote_fn)
        gpu_manager = GPUManager.remote(60, [0, 1])
        test_lock_unlock_by_time_limit(gpu_manager)

        gpu_manager = GPUManager.remote(60, [0, 1])
        test_lock_unlock_train_main_remote(
            tmp_path,
            wrapper=wrapper,
            make_config=_make_config,
            gpu_manager=gpu_manager,
        )

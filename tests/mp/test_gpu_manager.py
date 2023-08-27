import copy
import functools
import os
import platform
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
import ray
import torch

from ablator.modules.loggers.file import FileLogger
from ablator.mp.gpu_manager import GPUManager, unlock_gpu, wait_get_gpu
from ablator.mp.train_remote import _apply_unlock_hook, train_main_remote
from ablator.utils._nvml import _get_gpu_info, get_cuda_processes

wait_get_gpu = functools.partial(wait_get_gpu, expected_util_mb=100, max_timeouts=2)

n_gpus = min(torch.cuda.device_count(), 2)


def _running_processes(gpus, process_prefix):
    return any(
        [
            len([p for p in ps if process_prefix in p.process_name]) > 0
            for ps in gpus.values()
        ]
    )


def monitor_gpu_usage(process_prefix="test_gpu_manager_") -> dict[int, list[int]]:
    if "microsoft-standard" in platform.uname().release:
        raise NotImplementedError(
            "Can not run `monitor_gpu_usage` on WSL as there is no process_name support."
        )
    start_time = time.time()
    while True:
        cuda_ps = get_cuda_processes()
        if _running_processes(cuda_ps, process_prefix):
            return cuda_ps
        if time.time() - start_time > 30:
            raise RuntimeError(
                f"Could not find running process {process_prefix} in time."
            )
        time.sleep(0.1)


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


def _make_future_gpu(remote_fn, name, gpu_manager, kwargs):
    return ray.get(_make_remote(remote_fn, name, gpu_manager, kwargs))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The test is meant to evaluate the allocation of cuda memory.",
)
def test_get_gpu_info():
    # This is testing a bug regarding _get_gpu_info  https://github.com/gpuopenanalytics/pynvml/issues/49
    torch.randn(100).to("cuda:0")
    pid = os.getpid()
    gpus = _get_gpu_info()
    assert pid in [p["pid"] for p in gpus[0]["processes"]]


@pytest.mark.skipif(
    n_gpus < 2 or "microsoft-standard" in platform.uname().release,
    reason="The test is meant to evaluate the allocation of cuda memory and is not applicable for Windows",
)
def test_gpu_alignment(
    very_patient_gpu_manager: GPUManager, ray_cluster, locking_remote_fn
):
    """
    test that the GPU index numbers are aligned between torch to device and
    get_gpu_mem and get_cuda_processes.
    """
    # UID to avoid conflict from concurrent tests.
    process_prefix = str(uuid.uuid4())[:4]
    gpu_one = wait_get_gpu(very_patient_gpu_manager, process_name=process_prefix)
    gpu_two = wait_get_gpu(very_patient_gpu_manager, process_name=process_prefix)

    gpus = ray.get(very_patient_gpu_manager._gpus.remote())
    assert (
        gpus[gpu_one]._locking_process is not None
        and gpus[gpu_two]._locking_process is not None
    )
    for i in [gpu_one, gpu_two]:
        p_name = f"{process_prefix}_{i}"
        _remote = _make_remote(
            locking_remote_fn,
            p_name,
            very_patient_gpu_manager,
            dict(gpu_id=i, gpu_manager=very_patient_gpu_manager),
        )
        cuda_ps = monitor_gpu_usage(p_name)
        assert any([p_name in p.process_name for p in cuda_ps[i]])
        unlock_gpu(very_patient_gpu_manager, i)
        assert ray.get(_remote)
    assert True


@pytest.mark.skipif(
    n_gpus < 2 or "microsoft-standard" in platform.uname().release,
    reason="The test is meant to evaluate the allocation of cuda memory and is not applicable to Windows.",
)
def test_allocation_bottleneck(very_patient_gpu_manager, remote_fn):
    """
    Test that remotes are allocated evenly among 2 GPUs when there is a bottleneck.
    """
    process_prefix = str(uuid.uuid4())[:4]
    n_remotes = 2 * n_gpus
    gpu_allocations = {i: [] for i in range(n_gpus)}
    for i in range(n_remotes):
        p_name = f"{process_prefix}_{i}"
        gpu_id = wait_get_gpu(
            very_patient_gpu_manager, process_name=p_name, max_timeouts=30
        )
        _remote = _make_remote(
            remote_fn,
            p_name,
            very_patient_gpu_manager,
            dict(gpu_id=gpu_id, gpu_manager=very_patient_gpu_manager),
        )
        cuda_ps = monitor_gpu_usage(p_name)
        assert any([p_name in p.process_name for p in cuda_ps[gpu_id]])

        gpu_allocations[gpu_id].append(_remote)
    assert all(
        (np.array(ray.get(gpu_allocations[i])) == i).all() for i in range(n_gpus)
    )


@pytest.mark.skipif(
    n_gpus < 2,
    reason="The test is meant to evaluate allocation of cuda memory.",
)
def test_lock_unlock(assert_error_msg, very_patient_gpu_manager, remote_fn):
    """
    test that requesting and unlocking resources works
    as expected and when there is no GPU available an error is thrown.
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
    assert all(
        [
            gpu._locking_process is None
            for gpu in ray.get(very_patient_gpu_manager._gpus.remote()).values()
        ]
    )


@pytest.mark.skipif(
    n_gpus != 1,
    reason="The test is meant to evaluate allocation of cuda memory.",
)
def test_lock_unlock_single_gpu(assert_error_msg, very_patient_gpu_manager, remote_fn):
    """
    test that requesting and unlocking resources works
    as expected and when there is no GPU available an error is thrown.
    """

    fn_name = "test_lock_unlock"
    gpu = wait_get_gpu(very_patient_gpu_manager, process_name=fn_name)
    assert gpu == 0
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
    assert all(
        [
            gpu._locking_process is None
            for gpu in ray.get(very_patient_gpu_manager._gpus.remote()).values()
        ]
    )


@pytest.mark.skipif(
    n_gpus != 1,
    reason="The test is meant to evaluate allocation of cuda memory.",
)
def test_lock_unlock_by_id_single_gpu(very_patient_gpu_manager, remote_fn):
    """This is the same test as test_lock_unlock_by_id but for a single GPU"""
    # test unlocked by process name
    fn_name = "test_lock_unlock_by_id_single_gpu"
    # We test whether we can lock / unlock using a remote
    gpu = wait_get_gpu(very_patient_gpu_manager, process_name=f"{fn_name}_0")
    _make_future_gpu(
        remote_fn,
        f"{fn_name}_0",
        very_patient_gpu_manager,
        dict(gpu_id=gpu, gpu_manager=very_patient_gpu_manager),
    )
    gpus = ray.get(very_patient_gpu_manager._gpus.remote())
    assert gpus[gpu]._locking_process == None

    # We only lock and check to see whether it updates the gpu_manager
    gpu = wait_get_gpu(very_patient_gpu_manager, process_name=f"{fn_name}_0")
    gpus = ray.get(very_patient_gpu_manager._gpus.remote())
    assert gpus[gpu]._locking_process == f"{fn_name}_0"
    assert gpus[gpu].lock_timeout == 10000000


@pytest.mark.skipif(
    n_gpus < 2,
    reason="The test is meant to evaluate allocation of cuda memory.",
)
def test_lock_unlock_by_id(very_patient_gpu_manager, remote_fn):
    # This test is same as "test_lock_unlock_by_id_single_gpu" but more comprehensive
    # it tests the GPU prioritization as well.

    # test unlocked by process name
    fn_name = "test_lock_unlock_by_id"
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
def test_lock_unlock_by_time_limit_mgpu(gpu_manager):
    # impatient gpu manager
    fn_name = "test_lock_unlock_by_time_limit_mgpu"
    gpu_one = wait_get_gpu(gpu_manager, process_name=fn_name)
    gpu_three = wait_get_gpu(gpu_manager, process_name=fn_name)
    assert gpu_one != gpu_three


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The test is meant to evaluate allocation of cuda memory.",
)
def test_lock_unlock_by_time_limit_single_gpu(gpu_manager):
    # Extending test_lock_unlock_by_time_limit_mgpu
    fn_name = "test_lock_unlock_by_time_limit_single_gpu"
    gpu_one = wait_get_gpu(gpu_manager, process_name=fn_name)
    time.sleep(5)
    gpu_two = wait_get_gpu(gpu_manager, process_name=fn_name)
    assert gpu_one == gpu_two


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
    wrapper.init_state(config, resume=resume, remote_progress_bar=progress_bar)
    wrapper.train()
    assert wrapper._is_locked == False


@pytest.mark.skipif(
    n_gpus < 2,
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


@pytest.mark.skipif(
    n_gpus != 1,
    reason="Cuda is required to perform this test.",
)
def test_lock_unlock_train_main_remote_single_gpu(
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
    gpu = wait_get_gpu(very_patient_gpu_manager, process_name="train_main_remote_test")
    kwargs["gpu_id"] = gpu
    gpus_before = ray.get(very_patient_gpu_manager._gpus.remote())
    assert gpus_before[0]._locking_process == "train_main_remote_test"
    _make_future_gpu(
        train_main_remote,
        "train_main_remote_test",
        very_patient_gpu_manager,
        kwargs,
    )
    gpus = ray.get(very_patient_gpu_manager._gpus.remote())
    assert gpus[0]._locking_process == None


if __name__ == "__main__":
    from tests.conftest import run_tests_local
    from tests.test_plugins.model import (
        MyErrorCustomModel,
        TestWrapper,
        _make_config,
        _remote_fn,
        _locking_remote_fn,
    )

    tmp_path = Path("/tmp/test_gpu_manager")
    wrapper = TestWrapper(MyErrorCustomModel)

    n_devices = min(2, torch.cuda.device_count())
    two_gpu_tests = [
        "test_allocation_bottleneck",
        "test_lock_unlock_by_time_limit_mgpu",
        "test_lock_unlock_by_id",
        "test_lock_unlock",
        "test_gpu_alignment",
        "test_lock_unlock_train_main_remote",
    ]

    one_gpu_tests = [
        "test_lock_unlock_by_id_single_gpu",
        "test_lock_unlock_by_time_limit_single_gpu",
        "test_lock_unlock_single_gpu",
        "test_lock_unlock_train_main_remote_single_gpu",
    ]
    l = locals()
    fn_names = [fn for fn in l if fn.startswith("test_")]
    if n_devices == 0:
        fn_names = []
    elif n_devices == 1:
        for _fn_name in two_gpu_tests:
            fn_names.remove(_fn_name)
    elif n_devices == 2:
        for _fn_name in one_gpu_tests:
            fn_names.remove(_fn_name)

    fn_names = ["test_lock_unlock_train_main_remote_single_gpu"]
    test_fns = [l[fn] for fn in fn_names]

    kwargs = {
        "wrapper": copy.deepcopy(wrapper),
        "make_config": _make_config,
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(n_devices))
    unpickable_kwargs = {
        "gpu_manager": lambda: GPUManager.remote(5, list(range(n_devices))),
        "very_patient_gpu_manager": lambda: GPUManager.remote(
            10000000, list(range(n_devices))
        ),
    }

    run_tests_local(test_fns, kwargs=kwargs, unpickable_kwargs=unpickable_kwargs)

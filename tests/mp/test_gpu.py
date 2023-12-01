import copy
import os
import pickle
import platform
import re
import time
from pathlib import Path

import mock
import numpy as np
import pytest
import ray
import torch

from ablator.mp.gpu import GPU, GPUError, ResourceManager, unlock_gpu, wait_get_gpu
from ablator.mp.node import run_actor_node
from ablator.mp.train_remote import (
    _apply_unlock_hook,
    _handle_exception,
    train_main_remote,
)
from ablator.mp.utils import get_node_ip, run_lambda
from ablator.utils._nvml import _get_gpu_info, get_gpu_mem
from ablator.utils.base import Dummy

MAX_TIMEOUT = 20

IS_CUDA_AVAILABLE = torch.cuda.is_available()

DEVICDE_COUNT = torch.cuda.device_count()
IS_MAC = "darwin" in platform.system().lower()


@pytest.mark.mp
def test_resource_manager_failing_gpu(ray_cluster):
    head_ip = get_node_ip()
    start_time = time.time()
    breakdown_time = 3

    def get_gpu_mem(*args, **kwargs):
        if (
            time.time() - start_time < MAX_TIMEOUT
            or time.time() - start_time > MAX_TIMEOUT + breakdown_time
        ):
            return {}
        else:
            return None

    with (
        mock.patch("ablator.mp.gpu.get_gpu_mem", get_gpu_mem),
        mock.patch("ablator.mp.utils.get_gpu_mem", get_gpu_mem),
    ):
        manager = ResourceManager(
            node_ip=head_ip,
            ray_address=ray_cluster.cluster_address,
            resource_lock_timeout=15,
            update_interval=1,
        )
        for _ in range(MAX_TIMEOUT * 2):
            resources = manager.resources()
            if not resources.is_active:
                break
            time.sleep(1)
        assert not resources.is_active
        time.sleep(breakdown_time + 2)
        resources = manager.resources()
        assert resources.is_active
        manager.stop()


@pytest.mark.mp
def test_run_lambda():
    head_ip = get_node_ip()
    assert not run_lambda(
        lambda: torch.cuda.is_available(),
        cuda=False,
        node_ip=head_ip,
    )


@pytest.mark.mp
@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Meant to test GPU allocation")
def test_run_lambda_cuda():
    head_ip = get_node_ip()
    assert run_lambda(
        lambda: torch.cuda.is_available(),
        cuda=True,
        node_ip=head_ip,
    )


# TODO debug / fix this
@pytest.mark.mp
@pytest.mark.skipif(
    IS_MAC,
    reason="MAC does not report accurate memory information for whatever reason.",
)
def test_resource_manager_process_mem(ray_cluster, remote_fn):
    head_ip = get_node_ip()
    manager = run_actor_node(
        ResourceManager,
        cuda=False,
        node=head_ip,
        kwargs=dict(
            node_ip=head_ip,
            ray_address=ray_cluster.cluster_address,
            update_interval=1,
            resource_lock_timeout=15,
        ),
    )
    init_resources = ray.get(manager.resources.remote())
    init_memory = init_resources.mem
    start_time = time.time()
    futures = []
    while True:
        futures.append(
            run_lambda(
                remote_fn,
                cuda=False,
                node_ip=head_ip,
                run_async=True,
            )
        )

        resources = ray.get(manager.resources.remote())
        if resources.mem > init_memory + 5:
            break
        elif time.time() - start_time > MAX_TIMEOUT * 10:
            break
        time.sleep(0.1)
    assert resources.mem > init_memory + 5
    ray.get(futures)
    ray.get(manager.stop.remote())


@pytest.mark.mp
def test_resource_manager_process_cpu(ray_cluster):
    head_ip = get_node_ip()
    manager = run_actor_node(
        ResourceManager,
        cuda=False,
        node=head_ip,
        kwargs=dict(
            node_ip=head_ip,
            ray_address=ray_cluster.cluster_address,
            resource_lock_timeout=15,
            update_interval=1,
        ),
    )
    for _ in range(MAX_TIMEOUT):
        init_resources = ray.get(manager.resources.remote())
        init_usage = np.mean(init_resources.cpu_usage)
        if init_usage < 70:
            break
        time.sleep(1)
    assert init_usage < 70, "High CPU utilization to run the test."
    start_time = time.time()
    futures = []

    def waste_cpu():
        start_time = time.time()
        while True:
            time.sleep(0.0001)
            if time.time() - start_time > MAX_TIMEOUT * 10:
                break

    while True:
        futures += [
            run_lambda(
                waste_cpu,
                cuda=False,
                node_ip=head_ip,
                run_async=True,
            )
        ] * 5

        resources = ray.get(manager.resources.remote())
        if np.mean(resources.cpu_usage) > init_usage + 5:
            break
        elif time.time() - start_time > MAX_TIMEOUT * 10:
            break
        time.sleep(1)
    assert np.mean(resources.cpu_usage) > init_usage + 5
    ray.get(manager.stop.remote())


@pytest.mark.mp
def test_gpu_locking(ray_cluster, n_gpus):
    GPUS = {i: (i + 1) * 100 for i in range(n_gpus)}

    def get_gpu_mem(*args, **kwargs):
        return GPUS

    head_ip = get_node_ip()
    with (
        mock.patch("ablator.mp.gpu.get_gpu_mem", get_gpu_mem),
        mock.patch("ablator.mp.utils.get_gpu_mem", get_gpu_mem),
    ):
        manager = ResourceManager(
            node_ip=head_ip,
            ray_address=ray_cluster.cluster_address,
            resource_lock_timeout=100,
            update_interval=1,
        )
        assert manager.request_gpu((n_gpus + 1) * 100) is None
        gpu = manager.request_gpu((n_gpus) * 100 - 1)
        assert gpu is not None
        assert gpu.device_id == n_gpus - 1
        assert gpu.locking_process_name is None
        assert manager.request_gpu((n_gpus) * 100 - 1) is None
        manager.unlock(gpu)
        gpu = manager.request_gpu(None, "x")
        assert gpu.locking_process_name == "x"
        manager.unlock(gpu)
        gpu_list = []
        for _ in range(n_gpus):
            gpu = manager.request_gpu()
            assert manager.gpus[gpu.device_id] == gpu
            assert gpu.free_mem == (gpu.device_id + 1) * 100
            gpu_list.append(gpu)

        assert [gpu.device_id for gpu in gpu_list] == list(range(n_gpus))[::-1]
        assert all(gpu.is_locked for gpu in gpu_list)
        # test locking / unlocking manually and order based on gpu
        # util
        assert manager.request_gpu() is None
        manager.unlock(gpu_list[-1])
        manager.unlock(gpu_list[0])
        assert manager.request_gpu() == gpu_list[0]
        assert manager.request_gpu() == gpu_list[-1]
        assert manager.request_gpu() is None
        # Assert automatic unlock based on timeout
        timeout = 10
        manager.stop()
        manager = ResourceManager(
            node_ip=head_ip,
            ray_address=ray_cluster.cluster_address,
            resource_lock_timeout=timeout,
            update_interval=1,
        )
        gpu_list = []
        while (gpu := manager.request_gpu()) is not None:
            gpu_list.append(gpu)
            if len(gpu_list) > n_gpus * 2:
                raise RuntimeError
        time.sleep(timeout - 5)
        assert any(gpu.is_locked for gpu in gpu_list)
        time.sleep(5)
        assert not any(gpu.is_locked for gpu in gpu_list)
        manager.stop()


def test_gpu_properties():
    timeout = 1
    gpu = GPU(device_id=1, free_mem=1000, max_mem=1000, lock_timeout=timeout)
    assert gpu.device_id == 1
    assert gpu.free_mem == 1000
    assert gpu.lock_timeout == timeout
    assert gpu.lock_timestamp is None
    assert gpu.locking_process_name is None
    assert not gpu.is_locked
    gpu.lock("X")
    assert gpu.locking_process_name == "X"
    time.sleep(timeout + 0.1)
    assert gpu.locking_process_name is None

    # test a longer timeout
    gpu = GPU(device_id=1, free_mem=1000, max_mem=1000, lock_timeout=1000)
    gpu.lock("X2")
    assert gpu.locking_process_name == "X2"
    time.sleep(timeout + 0.1)
    assert gpu.locking_process_name is not None
    gpu.unlock()
    assert gpu.locking_process_name is None
    assert not gpu.is_locked

    gpu = GPU(device_id=1, free_mem=1000, max_mem=1000, lock_timeout=1)
    gpu.lock("x")
    pickled_gpu = pickle.dumps(gpu)
    assert gpu.is_locked
    time.sleep(1)
    assert not gpu.is_locked
    _gpu = pickle.loads(pickled_gpu)
    assert _gpu.__dict__ == gpu.__dict__
    assert not _gpu.is_locked
    gpu = GPU(device_id=1, free_mem=1000, max_mem=1000, lock_timeout=1)
    with pytest.raises(ValueError, match="Can not change property `device_id`."):
        gpu.device_id = 1
    with pytest.raises(ValueError, match="Can not change property `max_mem`."):
        gpu.max_mem = 1
    gpu.free_mem = 1
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Can not specify larger `free_mem` than total GPU memory `max_mem` (1000)."
        ),
    ):
        gpu.free_mem = 1001
    with pytest.raises(
        ValueError,
        match="Can not specify larger `free_mem` than total GPU memory `max_mem`.",
    ):
        gpu = GPU(device_id=1, free_mem=1001, max_mem=1000, lock_timeout=1)


@pytest.mark.mp
def test_wait_get_gpu(ray_cluster, update_gpus_fixture, update_no_gpus_fixture, n_gpus):
    head_ip = get_node_ip()
    with mock.patch.object(ResourceManager, "heartbeat", update_no_gpus_fixture):
        manager = run_actor_node(
            ResourceManager,
            cuda=False,
            node=head_ip,
            kwargs=dict(
                node_ip=head_ip,
                ray_address=ray_cluster.cluster_address,
                resource_lock_timeout=15,
                update_interval=1,
            ),
        )
        with pytest.raises(GPUError, match="No available GPU."):
            gpu = wait_get_gpu(manager)
        ray.get(manager.stop.remote())
    with mock.patch.object(ResourceManager, "heartbeat", update_gpus_fixture):
        manager = run_actor_node(
            ResourceManager,
            cuda=False,
            node=head_ip,
            kwargs=dict(
                node_ip=head_ip,
                ray_address=ray_cluster.cluster_address,
                resource_lock_timeout=100000,
                update_interval=1,
            ),
        )
        start_time = time.time()
        # impossible request should return immediately
        with pytest.raises(GPUError, match="No available GPU."):
            gpu = wait_get_gpu(
                manager, expected_util_mb=n_gpus * 100 + 1, max_timeouts=5
            )
        end_time = time.time()
        # a single remote is scheduled for above which should take less than 10 seconds
        # and on an overloaded system. If this test fails it could mean
        # you need to clean up ray before running it again.
        assert end_time - start_time < 10

        gpu_list = []
        for i in range(n_gpus):
            gpu = wait_get_gpu(manager)
            gpu_list.append(gpu)
        assert [gpu.device_id for gpu in gpu_list] == list(range(n_gpus))[::-1]
        with pytest.raises(GPUError, match="No available GPU."):
            gpu = wait_get_gpu(manager, max_timeouts=3)
        unlock_gpu(manager, gpu_list[0])
        gpu = wait_get_gpu(manager, max_timeouts=3)
        assert gpu.device_id == gpu_list[0].device_id
        ray.get(manager.stop.remote())


@pytest.mark.mp
def test_lock_unlock_train_main_remote(
    tmp_path: Path, ray_cluster, wrapper, make_config, update_gpus_fixture
):
    """
    End-to-end test of the unlock hook.
    """
    head_ip = get_node_ip()

    config = make_config(tmp_path.joinpath("mock_model"))
    config.device = "cpu"
    mp_logger = Dummy()
    with mock.patch.object(ResourceManager, "heartbeat", update_gpus_fixture):
        manager = run_actor_node(
            ResourceManager,
            cuda=False,
            node=head_ip,
            kwargs=dict(
                node_ip=head_ip,
                ray_address=ray_cluster.cluster_address,
                resource_lock_timeout=100000,
                update_interval=1,
            ),
        )

        gpu = wait_get_gpu(manager, max_timeouts=5)
        assert ray.get(manager.gpu_dict.remote())[gpu.device_id].is_locked
        run_lambda(
            train_main_remote,
            node_ip=head_ip,
            fn_kwargs=dict(
                model=wrapper,
                run_config=config,
                mp_logger=mp_logger,
                resource_manager=manager,
                uid=0,
                gpu=gpu,
            ),
            run_async=True,
        )
        for _ in range(10):
            if not ray.get(manager.gpu_dict.remote())[gpu.device_id].is_locked:
                break
            time.sleep(1)
        assert not ray.get(manager.gpu_dict.remote())[gpu.device_id].is_locked

        # test whether an exception aka `_handle_exception` is going to release the GPU
        # when an error is raised.
        gpu = wait_get_gpu(manager, max_timeouts=5)
        assert ray.get(manager.gpu_dict.remote())[gpu.device_id].is_locked
        run_lambda(
            train_main_remote,
            node_ip=head_ip,
            fn_kwargs=dict(
                model=lambda: None,
                run_config=config,
                mp_logger=mp_logger,
                resource_manager=manager,
                uid=0,
                gpu=gpu,
            ),
            run_async=True,
        )
        for _ in range(10):
            if not ray.get(manager.gpu_dict.remote())[gpu.device_id].is_locked:
                break
            time.sleep(1)
        assert not ray.get(manager.gpu_dict.remote())[gpu.device_id].is_locked


@pytest.mark.mp
def test_lock_unlock_hook(
    tmp_path: Path, ray_cluster, wrapper, make_config, update_gpus_fixture
):
    """
    targeted test to _handle_exception and _apply_unlock_hook.
    """
    head_ip = get_node_ip()

    config = make_config(tmp_path.joinpath("mock_model"))
    config.device = "cpu"
    mp_logger = Dummy()
    with mock.patch.object(ResourceManager, "heartbeat", update_gpus_fixture):
        manager = run_actor_node(
            ResourceManager,
            cuda=False,
            node=head_ip,
            kwargs=dict(
                node_ip=head_ip,
                ray_address=ray_cluster.cluster_address,
                resource_lock_timeout=100000,
                update_interval=1,
            ),
        )

        def _gpu_is_locked(gpu: GPU):
            return ray.get(manager.gpu_dict.remote())[gpu.device_id].is_locked

        gpu = wait_get_gpu(manager, max_timeouts=5)

        assert _gpu_is_locked(gpu)

        assert not hasattr(wrapper, "_is_locked")
        _apply_unlock_hook(wrapper, manager, gpu)
        assert hasattr(wrapper, "_is_locked")
        assert wrapper._is_locked
        try:
            wrapper.train_step()
        except Exception:
            ...
        assert wrapper._is_locked
        assert _gpu_is_locked(gpu)
        try:
            wrapper.train_step()
        except Exception as e:
            _handle_exception(
                e,
                model=wrapper,
                run_config=config,
                mp_logger=mp_logger,
                resource_manager=manager,
                gpu=gpu,
                uid=0,
                fault_tollerant=True,
                crash_exceptions_types=None,
            )
        assert wrapper._is_locked
        assert not _gpu_is_locked(gpu)

        gpu = wait_get_gpu(manager, max_timeouts=5)
        assert _gpu_is_locked(gpu)

        try:
            wrapper.init_state(config)
            wrapper.train_step()
        except Exception:
            ...
        assert not wrapper._is_locked
        assert not _gpu_is_locked(gpu)
        ray.get(manager.stop.remote())


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The test is meant to evaluate the allocation of cuda memory.",
)
def test_get_gpu_info():
    """
    NOTE this test can be flaky if more CUDA processes are running or stopping in the background.
    """
    # This is testing a bug regarding _get_gpu_info  https://github.com/gpuopenanalytics/pynvml/issues/49
    gpus = _get_gpu_info()
    assert gpus[0]["minor_number"] == 0
    used_mem = gpus[0]["fb_memory_usage"]["used"]
    t = torch.randn(100_000_000).to("cuda:0")
    pid = os.getpid()
    gpus = _get_gpu_info()
    assert pid in [p["pid"] for p in gpus[0]["processes"]]
    assert gpus[0]["fb_memory_usage"]["used"] > used_mem


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Meant to test GPU allocation")
def test_get_gpu():
    """
    All of the tests in this module assume a certain behavior from `get_gpu_mem` which
    we test.
    """

    used = get_gpu_mem("used")
    free = get_gpu_mem("free")
    total = get_gpu_mem("total")
    for i in used:
        assert used[i] < total[i] and free[i] < total[i]


@pytest.mark.skipif(DEVICDE_COUNT < 2, reason="Meant to test GPU id allocation")
def test_get_multi_gpu():
    device_id = 1
    gpus = _get_gpu_info()
    assert gpus[device_id]["minor_number"] == device_id
    used_mem = gpus[device_id]["fb_memory_usage"]["used"]
    torch.randn(100).to(f"cuda:{device_id}")
    pid = os.getpid()
    gpus = _get_gpu_info()
    assert gpus[device_id]["fb_memory_usage"]["used"] > used_mem
    assert pid in [p["pid"] for p in gpus[device_id]["processes"]]


def test_get_no_gpu():
    """
    All of the tests in this module assume a certain behavior from `get_gpu_mem` which
    we test.

    This test still assumed get_gpu_info works correctly.
    """
    with mock.patch("ablator.utils._nvml._get_gpu_info", return_value=[]):
        used = get_gpu_mem("used")
        assert len(used) == 0


if __name__ == "__main__":
    from tests.conftest import run_tests_local
    from tests.test_plugins.model import (
        MockActor,
        MyErrorCustomModel,
        TestWrapper,
        update_gpus,
        update_no_gpus,
        N_GPUS,
    )

    wrapper = TestWrapper(MyErrorCustomModel)
    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]
    kwargs = {
        "mock_actor": MockActor,
        "wrapper": copy.deepcopy(wrapper),
        "update_gpus_fixture": update_gpus,
        "update_no_gpus_fixture": update_no_gpus,
        "n_gpus": N_GPUS,
    }
    run_tests_local(test_fns, kwargs=kwargs)

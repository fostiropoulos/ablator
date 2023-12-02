"""
Tests are meant to be supplementary to tests from `test_multi_node_cluster` for single node environments.
The purpose of these tests are to identify errors that would arise when running on single-node
i.e. a parallel experiment in the same machine for different platforms outside of Linux
(since these tests are covered by `test_multi_node_cluster` and can be redundant)
"""
import random
import time
import uuid
from pathlib import Path

import mock
import pytest
import ray

from ablator.mp.cluster import ClusterManager, get_username
from ablator.mp.gpu import GPUError, ResourceManager

MAX_TIMEOUT = 30


def wait_cluster_condition(manager: ClusterManager, target_nodes):
    for _ in range(MAX_TIMEOUT):
        nodes = manager.available_resources
        if len(nodes) == target_nodes:
            break
        time.sleep(1)
    assert len(nodes) == target_nodes, "Could not successfully set the cluster."


@pytest.mark.mp
def test_remote_cmd(tmp_path: Path, ray_cluster):
    manager = ClusterManager(
        tmp_path,
        sync_directory=tmp_path,
        ray_address=ray_cluster.cluster_address,
        timeout=15,
    )
    assert manager.run_cmd_head("whoami") == get_username()
    x, y = random.randint(0, 100), random.randint(0, 100)

    def fn(x, y):
        return (y, x)

    fn_kwargs = {"x": x, "y": y}
    y_out, x_out = manager.run_lambda_head(fn=fn, fn_kwargs=fn_kwargs)
    assert y_out == y and x_out == x
    remote = manager.run_lambda_head(fn=fn, fn_kwargs=fn_kwargs, run_async=True)
    y_out, x_out = ray.get(remote)
    assert y_out == y and x_out == x

    assert manager.run_lambda_head(uuid.getnode) == uuid.getnode()
    manager.stop()


@pytest.mark.mp
def test_get_gpu(tmp_path: Path, ray_cluster, update_gpus_fixture, n_gpus):
    with mock.patch.object(ResourceManager, "heartbeat", update_gpus_fixture):
        manager = ClusterManager(
            tmp_path,
            sync_directory=tmp_path,
            ray_address=ray_cluster.cluster_address,
            timeout=5,
            update_interval=1,
        )
        wait_cluster_condition(manager, ray_cluster.nodes)
        device_id = n_gpus - 1
        head_ip = manager.head_ip
        # right on the limit will not return any available nodes
        resources = manager.sorted_resources()
        gpu_available = resources[head_ip].gpu_free_mem[device_id]
        assert len(manager.sorted_resources(gpu_available)) == 0
        # below the limit will return all nodes
        resources = manager.sorted_resources(gpu_available - 1)
        assert len(resources) == ray_cluster.nodes
        assert next(iter(resources)) == manager.head_ip
        for node_ip in ray_cluster.node_ips():
            for gpu_id in range(n_gpus):
                gpu, actor = manager.get_gpu(node_ip, process_name=f"x_{gpu_id}")
                assert gpu.locking_process_name == f"x_{gpu_id}"
                assert gpu.device_id == n_gpus - gpu_id - 1
            with pytest.raises(GPUError):
                gpu, actor = manager.get_gpu(node_ip, process_name=f"x_{gpu_id}")

        with pytest.raises(RuntimeError, match="Could not find x in nodes"):
            manager.get_gpu("x", "x")
        manager.stop()


@pytest.mark.mp
def test_head_resource_error(tmp_path: Path, ray_cluster, inactive_resource):
    # we disable updating the other cluster nodes to avoid errors due
    # to inactive resources.

    with (
        mock.patch.object(ClusterManager, "heartbeat", lambda self: None),
        mock.patch.object(ResourceManager, "resources", inactive_resource),
    ):
        manager = ClusterManager(
            tmp_path,
            sync_directory=tmp_path,
            ray_address=ray_cluster.cluster_address,
            timeout=10,
            update_interval=1,
        )
        with pytest.raises(
            RuntimeError, match="Could not read the resources of the head node."
        ):
            manager.head_resources
    assert True


if __name__ == "__main__":
    from tests.conftest import run_tests_local
    from tests.test_plugins.model import update_gpus, N_GPUS, _inactive_resource

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]
    kwargs = {
        "update_gpus_fixture": update_gpus,
        "n_gpus": N_GPUS,
        "inactive_resource": _inactive_resource,
    }
    run_tests_local(test_fns, kwargs=kwargs)

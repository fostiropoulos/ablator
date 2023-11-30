"""
Tests for multi-node clusters are supported only for Linux. The reason being that
a cluster needs to have all nodes with the same operating system. Having a cluster
of Macs or WSL or Windows machines is an unrealistic scenario.

Additionally, we are only able to set-up mock docker nodes for Linux system. Mac
and WSL can run the same tests though docker but not natively, i.e. calling
`make tests` in the main folder (run in a docker) but not via `pytest` run
in their local.
"""
import copy
import platform
import random
import time
import uuid
from pathlib import Path

import mock
import numpy as np
import pytest
import ray
from ray import exceptions

from ablator.mp.cluster import ClusterManager
from ablator.mp.gpu import ResourceManager
from ablator.mp.node import run_actor_node
from ablator.mp.utils import (
    Resource,
    get_node_ip,
    get_ray_address,
    get_username,
    sort_resources,
    sort_resources_by_util,
)


IS_LINUX = (
    "microsoft-standard" not in platform.uname().release
    and "darwin" not in platform.system().lower()
)

MAX_TIMEOUT = 30

pytestmark = pytest.mark.skipif(
    not IS_LINUX,
    reason="Cluster Manager test for multi-node Unix platforms",
)


def wait_cluster_condition(manager: ClusterManager, target_nodes):
    for _ in range(MAX_TIMEOUT):
        nodes = manager.available_resources
        if len(nodes) == target_nodes:
            break
        time.sleep(1)
    assert len(nodes) == target_nodes, "Could not successfully set the cluster."


def flatten_resource(resource: ResourceManager):
    return np.array([resource.mem] + resource.cpu_usage)


@pytest.mark.mp
def test_resource_update(tmp_path: Path, ray_cluster):
    # test what happens when the mount dies mid-way and
    # the node has to be re-initialized
    manager = ClusterManager(
        tmp_path,
        sync_directory=tmp_path,
        ray_address=ray_cluster.cluster_address,
        timeout=15,
        update_interval=1,
    )
    wait_cluster_condition(manager, ray_cluster.nodes)
    assert manager.head_ip in manager.available_resources
    assert get_node_ip() == manager.head_ip and manager.head_ip in get_ray_address()
    assert len(manager.available_resources) == ray_cluster.nodes
    node = manager.healthy_nodes.pop()

    def abc():
        time.sleep(120)

    name = str(uuid.uuid4()).replace("-", "")
    og_resources = copy.deepcopy(manager.available_resources)
    for node_ip, r in og_resources.items():
        assert name not in r.running_tasks

    node.run_lambda(abc, run_async=True, name=name)
    for i in range(10):
        if name in manager.available_resources[node.node_ip].running_tasks:
            break
        time.sleep(1)

    for node_ip, r in manager.available_resources.items():
        if node_ip == node.node_ip:
            assert name in r.running_tasks
        else:
            assert name not in r.running_tasks
    time.sleep(2)
    # test that the resources were updated
    for k, v in manager.available_resources.items():
        assert (
            np.array(flatten_resource(og_resources[k])) != np.array(flatten_resource(v))
        ).any()
    manager.stop()


@pytest.mark.mp
def test_mount_dead(tmp_path: Path, ray_cluster):
    # test what happens when the mount dies mid-way and
    # the node can not be restarted
    with mock.patch("ablator.mp.node.Node.restart", lambda x: None):
        manager = ClusterManager(
            tmp_path,
            sync_directory=tmp_path,
            ray_address=ray_cluster.cluster_address,
            timeout=30,
            update_interval=1,
        )
        # wait for cluster to start
        wait_cluster_condition(manager, ray_cluster.nodes)
        nodes = manager.healthy_nodes
        child_node = nodes.pop()
        assert child_node in manager.healthy_nodes
        child_node.unmount()
        for _ in range(MAX_TIMEOUT * 3):
            if child_node not in manager.healthy_nodes:
                break
            time.sleep(1)
        assert child_node not in manager.healthy_nodes
        assert not child_node.is_alive()
        assert all(n.is_alive() for n in manager.healthy_nodes)
        wait_cluster_condition(manager, ray_cluster.nodes)
        manager.stop()
    # we need to reset the cluster
    ray_cluster.kill_nodes()
    ray.shutdown()


@pytest.mark.mp
@pytest.mark.skip
def test_robustness(tmp_path: Path, ray_cluster, mock_actor):
    # test what happens a node dies because:
    #       1. ray crashes
    #       2. docker dies
    #       3. network dies
    # ... following will the server be successfully restarted?
    # ... what happens to the pending remotes?
    # ... what happens to the remote mount storage?

    manager = ClusterManager(
        tmp_path,
        sync_directory=tmp_path,
        ray_address=ray_cluster.cluster_address,
        timeout=15,
    )

    # wait for cluster to start
    wait_cluster_condition(manager, ray_cluster.nodes)
    nodes = manager.healthy_nodes
    child_node = nodes.pop()

    # we create a mock actor on the server
    mock_remote = run_actor_node(mock_actor, cuda=False, node=child_node)
    # we test that it is alive
    assert ray.get(mock_remote.is_alive.remote())
    # we kill the node and restart it. this could have been
    # because of 1, 2 and 3 above.
    random_num = random.randint(0, 100)
    child_node.run_cmd(f"echo '{random_num}' >> {child_node.remote_dir}/{random_num}")
    child_node_old_id = child_node.node_id
    assert child_node.restart()

    assert (tmp_path / str(random_num)).read_text().strip("\n") == str(random_num)

    # wait for cluster to revive

    wait_cluster_condition(manager, ray_cluster.nodes)
    assert child_node_old_id not in [n.node_id for n in nodes]

    # We test the pending remote.
    with pytest.raises(
        exceptions.RayActorError,
        match="The actor .*died.",
    ):
        ray.get(mock_remote.is_alive.remote(), timeout=10)

    # we start a new one.
    mock_remote = run_actor_node(mock_actor, cuda=False, node=child_node)
    # we test it is alive.
    assert ray.get(mock_remote.is_alive.remote())
    assert child_node.run_cmd(f"cat {child_node.remote_dir}/{random_num}") == str(
        random_num
    )
    ray_cluster.kill_nodes()
    assert not child_node.restart()
    manager.stop()
    ray_cluster.kill_nodes()
    ray.shutdown()


def test_sort_resources():
    n_resources = 3
    resources = {
        n_resources
        - 1
        - i: Resource(
            cpu_usage=[(i + 1) * 10 * 3, 0, 0],
            mem=(i + 1) * 10,
            cpu_count=3,
            gpu_free_mem=[(i + 1) * 100, 101],
            running_tasks=[],
            is_active=True,
        )
        for i in range(n_resources)
    }
    assert (
        len(sort_resources(resources)) == n_resources
        and (
            np.array(list(sort_resources(resources).keys()))
            == np.arange(n_resources)[::-1]
        ).all()
    )
    assert len(sort_resources(resources, memory_perc_limit=9)) == 0
    assert len(sort_resources(resources, cpu_util_perc_limit=9)) == 0
    for i in range(n_resources):
        assert len(sort_resources(resources, memory_perc_limit=i * 10 + 1)) == i
        assert len(sort_resources(resources, cpu_util_perc_limit=i * 10 + 1)) == i
    # test excluding by gpu-util
    assert len(sort_resources(resources, gpu_util_requirement=102)) == n_resources - 1
    assert len(sort_resources(resources, gpu_util_requirement=n_resources * 100)) == 0
    assert (
        len(sort_resources(resources, gpu_util_requirement=n_resources * 100 - 1)) == 1
    )
    base_resource = Resource(
        cpu_usage=[30, 0, 0],
        mem=10,
        cpu_count=3,
        gpu_free_mem=[100, 101],
        running_tasks=["a", "b"],
        is_active=True,
    )
    resources = {i: base_resource for i in range(n_resources)}
    # we test when all resources are the same that their order is maintained in the dictionary.
    # i.e. {0: x, 1:x, 2:x} -> {0:x, 1:x, 2:x}
    assert (
        np.array(
            list(sort_resources_by_util(resources=resources, eval_gpu=True).keys())
        )
        == np.arange(n_resources)
    ).all()
    low_cpu_usage = copy.deepcopy(base_resource)
    low_cpu_usage.cpu_usage[0] -= 1

    low_mem = copy.deepcopy(base_resource)
    low_mem.mem -= 1

    low_running_tasks = copy.deepcopy(base_resource)
    low_running_tasks.running_tasks.pop()

    low_gpu_util = copy.deepcopy(base_resource)
    low_gpu_util.gpu_free_mem[0] += 100
    low_resources = {
        "cpu_usage": low_cpu_usage,
        "mem": low_mem,
        "running_tasks": low_running_tasks,
        "gpu_free_mem": low_gpu_util,
    }

    for attr in ["cpu_usage", "mem", "running_tasks"]:
        _resources = copy.deepcopy(resources)
        least_used = np.random.randint(n_resources)
        _resources[least_used] = low_resources[attr]
        assert least_used == next(iter(sort_resources_by_util(_resources, False)))

    _resources = copy.deepcopy(resources)
    least_used = next(iter(sort_resources_by_util(_resources, False)))

    while (least_used_gpu := np.random.randint(n_resources)) == least_used:
        continue
    assert least_used != least_used_gpu
    _resources[least_used_gpu] = low_resources["gpu_free_mem"]
    assert least_used == next(
        iter(sort_resources_by_util(resources=_resources, eval_gpu=False))
    )
    assert least_used_gpu == next(
        iter(sort_resources_by_util(resources=_resources, eval_gpu=True))
    )


@pytest.mark.mp
def test_remote_cmd(tmp_path: Path, ray_cluster):
    manager = ClusterManager(
        tmp_path,
        sync_directory=tmp_path,
        ray_address=ray_cluster.cluster_address,
        timeout=15,
    )
    assert manager.run_cmd_head("whoami") == get_username()
    # wait for cluster to start
    for i in range(MAX_TIMEOUT):
        nodes = manager.healthy_nodes
        if len(nodes) > 0:
            break
        time.sleep(1)
    assert len(nodes) > 0, "Could not successfully start the cluster."
    healthy_node = manager.healthy_nodes[0]
    assert healthy_node.run_cmd("whoami") == healthy_node._username
    x, y = random.randint(0, 100), random.randint(0, 100)

    def _swap(x, y):
        return (y, x)

    fn_kwargs = {"x": x, "y": y}
    y_out, x_out = manager.run_lambda_head(fn=_swap, fn_kwargs=fn_kwargs)
    assert y_out == y and x_out == x
    remote = manager.run_lambda_head(fn=_swap, fn_kwargs=fn_kwargs, run_async=True)
    y_out, x_out = ray.get(remote)
    assert y_out == y and x_out == x

    y_out, x_out = healthy_node.run_lambda(_swap, fn_kwargs=fn_kwargs)
    assert y_out == y and x_out == x
    remote = healthy_node.run_lambda(fn=_swap, fn_kwargs=fn_kwargs, run_async=True)
    y_out, x_out = ray.get(remote)
    assert y_out == y and x_out == x
    # test that the lambda runs on the dedicated ip
    node_id = healthy_node.run_lambda(uuid.getnode)
    assert node_id == healthy_node.run_lambda(uuid.getnode)
    assert node_id != manager.run_lambda_head(uuid.getnode)
    assert manager.run_lambda_head(uuid.getnode) == uuid.getnode()
    manager.stop()


if __name__ == "__main__":
    from tests.conftest import run_tests_local
    from tests.test_plugins.model import MockActor

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]
    kwargs = {"mock_actor": MockActor}
    run_tests_local(test_fns, kwargs=kwargs)

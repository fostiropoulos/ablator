from pathlib import Path
import platform
import pytest

import ray

from ablator.mp.node_manager import NodeManager
from ablator.mp.utils import Resource


@pytest.mark.skipif(
    "microsoft-standard" not in platform.uname().release,
    reason="Node Manager test for WSL",
)
def test_node_manager_wsl(tmp_path: Path, ray_cluster):
    timeout = 5
    n_nodes = ray_cluster.nodes
    manager = NodeManager(tmp_path, ray_address=ray_cluster.cluster_address)
    results = manager.run_cmd("whoami", timeout=timeout)
    test_ips = ray_cluster.node_ips()
    for node, result in results.items():
        node_username, node_ip = node.split("@")
        test_ips.remove(node_ip)
        assert result.strip() == node_username
    assert len(test_ips) == 0


@pytest.mark.skipif(
    "microsoft-standard" in platform.uname().release,
    reason="Node Manager test for Unix platforms",
)
def test_node_manager(tmp_path: Path, ray_cluster):
    timeout = 5
    n_nodes = ray_cluster.nodes
    assert n_nodes >= 1, "This test only supports >= 1 nodes"
    manager = NodeManager(tmp_path, ray_address=ray_cluster.cluster_address)
    results = manager.run_cmd("whoami", timeout=timeout)
    test_ips = ray_cluster.node_ips()
    for node, result in results.items():
        node_username, node_ip = node.split("@")
        test_ips.remove(node_ip)
        assert result.strip() == node_username
    assert len(test_ips) == 0

    ray_cluster.kill_nodes(1)
    n_nodes -= 1
    results = manager.run_cmd("whoami", timeout=timeout)

    assert len(results) == n_nodes + 1
    ray_cluster.append_nodes(1)
    n_nodes += 1
    results = manager.run_cmd("whoami", timeout=timeout)

    assert len(results) == n_nodes + 1
    ray_cluster.kill_nodes(1)
    n_nodes -= 1
    results = manager.run_cmd("whoami", timeout=timeout)
    assert len(results) == n_nodes + 1
    ray_cluster.kill_nodes()

    results = manager.run_cmd("whoami", timeout=timeout)
    assert len(results) == 1  # the head node
    ray_cluster.setUp()


@pytest.mark.skipif(
    "microsoft-standard" in platform.uname().release,
    reason="Node Manager test for Unix platforms",
)
def test_shutdown(tmp_path: Path, ray_cluster, assert_error_msg):
    n_nodes = ray_cluster.nodes
    assert n_nodes >= 1, "This test only supports >= 1 nodes"
    msg = assert_error_msg(
        lambda: NodeManager(tmp_path, ray_address=ray_cluster.cluster_ip)
    )
    assert (
        msg
        == "`ray_address` does not match currently running ray instance. Can not initialize ray twice."
    )
    manager = NodeManager(tmp_path, ray_cluster.cluster_address)
    results = manager.run_cmd(
        "whoami",
    )
    assert len(results) == ray_cluster.nodes + 1
    # NOTE test restarting ray and NodeManager
    ray_cluster.tearDown()
    ray.shutdown()
    new_ray_cluster = type(ray_cluster)(nodes=0)
    new_ray_cluster.setUp()
    manager = NodeManager(tmp_path, ray_address=new_ray_cluster.cluster_address)
    results = manager.run_cmd(
        "whoami",
    )
    assert len(results) == 1  # the head node
    new_ray_cluster.append_nodes(1)
    results = manager.run_cmd(
        "whoami",
    )

    assert len(results) == len(new_ray_cluster.node_ips())
    new_ray_cluster.kill_nodes()
    results = manager.run_cmd(
        "whoami",
    )
    assert len(results) == 1  # the head node
    new_ray_cluster.tearDown()
    ray_cluster.setUp()


def assert_resources_equal(
    resource_one: dict[str, Resource], resource_two: dict[str, Resource]
):
    resource_one_values = list(resource_one.values())
    resource_two_values = list(resource_two.values())
    assert all(isinstance(r, Resource) for r in resource_one_values)
    assert all(isinstance(r, Resource) for r in resource_two_values)
    assert all(
        resource_one_values[0].cpu_count == r.cpu_count for r in resource_two_values
    )
    assert all(
        resource_one_values[0].gpu_free_mem == r.gpu_free_mem
        for r in resource_two_values
    )


def test_resource_utilization(tmp_path: Path, ray_cluster):
    manager = NodeManager(tmp_path, ray_address=ray_cluster.cluster_address)

    init_resources = manager.utilization()
    for i in range(3):
        resources = manager.utilization()
        assert_resources_equal(resources, init_resources)
        assert_resources_equal(resources, resources)
        assert len(resources) == ray_cluster.nodes + 1  #  +1 head
        assert set(ray_cluster.node_ips()) == set(resources)
        init_resources = resources


if __name__ == "__main__":
    from tests.conftest import run_tests_local

    l = locals()
    fn_names = [fn for fn in l if fn.startswith("test_")]
    test_fns = [l[fn] for fn in fn_names]
    run_tests_local(test_fns)

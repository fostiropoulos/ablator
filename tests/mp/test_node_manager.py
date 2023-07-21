import shutil
from pathlib import Path
import unittest

import ray
from ablator.mp.node_manager import NodeManager, Resource


def test_node_manager(tmp_path: Path, ray_cluster):
    # TODO py-test clean-ups
    timeout = 15
    n_nodes = 2
    manager = NodeManager(tmp_path)
    results = manager.run_cmd("whoami", timeout=timeout)
    test_ips = ray_cluster.node_ips()
    for node, result in results.items():
        node_username, node_ip = node.split("@")
        test_ips.remove(node_ip)
        assert result.strip() == node_username
    assert len(test_ips) == 0
    ray_cluster.append_nodes(2)
    n_nodes += 2
    results = manager.run_cmd("whoami", timeout=timeout)

    assert (
        len(results) == len(ray_cluster.node_ips()) and len(results) == n_nodes + 1
    )  # +1 for the head node
    ray_cluster.kill_node(0)
    n_nodes -= 1
    results = manager.run_cmd("whoami", timeout=timeout)
    assert (
        len(results) == len(ray_cluster.node_ips()) and len(results) == n_nodes + 1
    )  # +1 for the head node
    ray_cluster.kill_all()

    results = manager.run_cmd("whoami", timeout=timeout)
    assert len(results) == 1  # the head node

    ray.shutdown()
    try:
        results = manager.run_cmd("whoami", timeout=5)
    except Exception as e:
        assert "Ray has not been started yet." in str(e)

    # NOTE test restarting ray and NodeManager

    ray_cluster = type(ray_cluster)(nodes=0)
    ray_cluster.setUp(Path(__file__).parent)
    manager = NodeManager(tmp_path, ray_address=ray_cluster.cluster_ip)
    results = manager.run_cmd("whoami", timeout=timeout)
    assert len(results) == 1  # the head node
    ray_cluster.append_nodes(2)
    results = manager.run_cmd("whoami", timeout=timeout)

    assert len(results) == len(ray_cluster.node_ips())
    ray_cluster.kill_all()
    results = manager.run_cmd("whoami", timeout=timeout)
    assert len(results) == 1  # the head node


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
    manager = NodeManager(tmp_path)

    init_resources = manager.utilization()
    for i in range(3):
        resources = manager.utilization()
        assert_resources_equal(resources, init_resources)
        assert_resources_equal(resources, resources)
        assert len(resources) == 3  # 2 nodes +1 head
        assert set(ray_cluster.node_ips()) == set(resources)
        init_resources = resources


if __name__ == "__main__":
    from tests.conftest import DockerRayCluster

    tmp_file = Path("/tmp/").joinpath("t")
    shutil.rmtree(tmp_file, ignore_errors=True)
    tmp_file.mkdir(exist_ok=True)
    ray_cluster = DockerRayCluster()
    ray_cluster.setUp(Path(__file__).parent)
    test_node_manager(tmp_file, ray_cluster)
    test_resource_utilization(tmp_file, ray_cluster)
    breakpoint()
    print()
    pass

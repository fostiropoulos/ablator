import copy
import io
import logging
import os
import platform
import random
import re
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import docker
import mock
import numpy as np
import pytest
import ray
import torch
from filelock import FileLock
from ray.util.state import list_nodes
from xdist.scheduler.loadscope import LoadScopeScheduling

from ablator import package_dir
from ablator.mp.gpu_manager import GPUManager
from ablator.utils.base import _get_gpu_info

DOCKER_TAG = "ablator"
pytest_plugins = ["ray_models.model"]


def _assert_error_msg(fn, error_msg=None):
    try:
        fn()
        raise RuntimeError(
            f"{fn} did not cause an error with. Expected message: {error_msg}"
        )
    except Exception as excp:
        if error_msg is not None and not error_msg == str(excp):
            raise RuntimeError(
                f"{fn} caused a different error. Expected message: {error_msg}\nFound {str(excp)}"
            ) from excp
        else:
            return str(excp)


@pytest.fixture
def assert_error_msg():
    return _assert_error_msg


def _capture_output(fn):
    out = io.StringIO()

    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        fn()

    return out.getvalue(), err.getvalue()


@pytest.fixture
def capture_output():
    return _capture_output


def make_node(docker_client: docker.DockerClient, img, cluster_address):
    cuda_args = {}

    if torch.cuda.is_available():
        cuda_args = dict(
            runtime="nvidia",
            device_requests=[
                {"driver": "nvidia", "count": -1, "capabilities": [["gpu"]]}
            ],
        )
    c = docker_client.containers.run(
        img,
        command="/bin/bash",
        detach=True,
        tty=True,
        stdin_open=True,
        pid_mode="host",
        **cuda_args,
    )
    res = c.exec_run(f"service ssh start")
    assert "Starting OpenBSD Secure Shell server" in res.output.decode()
    res = c.exec_run(f"ray start --address='{cluster_address}'")
    _out = res.output.decode()
    assert "Ray runtime started." in _out
    node_ip = list(
        set(
            re.findall(
                r"[0-9]+(?:\.[0-9]+){3}",
                _out.split("This node has an IP address of ")[-1],
            )
        )
    )[0]

    return c, node_ip


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--docker-tag",
        action="store",
        default="ablator",
        help="the docker tag to use for launching a machine.",
    )


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_dist = pytest.mark.skip(reason="distributed tests run only on linux.")
    dist_arg_names = ["main_ray_cluster", "ray_cluster", "ablator", "ablator_results"]
    for item in items:
        argnames = item._fixtureinfo.argnames
        if any(name in argnames for name in dist_arg_names):
            if not config.getoption("--runslow"):
                item.add_marker(skip_slow)
            elif platform.system().lower() != "linux":
                item.add_marker(skip_dist)


def build_docker_image(docker_client: docker.DockerClient):
    py_version = platform.python_version()
    img, *_ = docker_client.images.build(
        nocache=False,
        path=package_dir.parent.as_posix(),
        tag="ablator",
        buildargs={"PY_VERSION": py_version},
    )
    return img


def get_docker_image(docker_client, docker_tag):
    # NOTE the ray and python version must match between nodes.
    # try:
    # Currently building the image is too slow, we expect it to have been pre-build
    return docker_client.images.get(docker_tag)
    # except:
    #     return build_docker_image(docker_client)


class DockerRayCluster:
    def __init__(
        self, nodes=1, working_dir=Path(__file__).parent, docker_tag=DOCKER_TAG
    ) -> None:
        self.nodes = nodes
        self.working_dir = working_dir
        self.docker_tag = docker_tag
        try:
            self.api_client = docker.APIClient()
            self.client = docker.from_env()
        except Exception as e:
            raise RuntimeError(
                "Could not find a docker installation. Please make sure docker is in Path and can run be run by the current system user (non-root) e.g. `docker run hello-world`. Please refer to https://github.com/fostiropoulos/ablator/blob/main/DEVELOPER.md for detailed instructions."
            ) from e

    def tearDown(self):
        self.kill_nodes()

    def setUp(self):
        if not ray.is_initialized():
            # Borrowed from ray.tests.test_gcs_fault_tolerance.py
            timeout_config = dict(
                gcs_failover_worker_reconnect_timeout=2,
                gcs_rpc_server_reconnect_timeout_s=2,
                health_check_initial_delay_ms=0,
                health_check_period_ms=1000,
                health_check_timeout_ms=1000,
                health_check_failure_threshold=2,
            )
            ray_cluster = ray.init(
                address="local",
                runtime_env={"working_dir": self.working_dir},
                _system_config=timeout_config,
            )
            self.cluster_ip = ray_cluster.address_info["node_ip_address"]
            self.cluster_address = ray_cluster.address_info["address"]
        else:
            ray_cluster = ray.get_runtime_context()
            self.cluster_address = ray_cluster.gcs_address
            self.cluster_ip, _ = self.cluster_address.split(":")

        self.img = get_docker_image(self.client, self.docker_tag)
        containers = self._active_containers()
        if self.nodes > 0:
            node_diff = self.nodes - len(containers)  # + 1 for the head.
            if node_diff > 0:
                self.append_nodes(node_diff)
            elif node_diff < 0:
                self.kill_nodes(int(node_diff * -1))

    def _active_containers(self):
        containers = {}
        node_ips = self.node_ips()
        for c in self.api_client.containers(filters={"ancestor": self.docker_tag}):
            ip = c["NetworkSettings"]["Networks"]["bridge"]["IPAddress"]
            if ip in node_ips and ip not in self.cluster_address:
                containers[ip] = c
        return containers

    def node_ips(self):
        try:
            return list(
                set(
                    [
                        n.node_ip
                        for n in list_nodes(
                            self.cluster_address,
                            limit=5,
                            timeout=5,
                            raise_on_missing_output=False,
                        )
                        if n.state == "ALIVE"
                    ]
                )
            )
        except:
            return []

    def append_nodes(self, n):
        # NOTE Github actions imposes a limit on running Docker containers.
        # It would not be a good idea to exceed that limit.
        prev_nodes = len(self.node_ips())
        for _ in range(n):
            make_node(self.client, self.img, self.cluster_address)
        self._wait_nodes(prev_nodes, n)

    def _wait_nodes(self, prev_nodes, added_nodes):
        for _ in range(120):
            current_nodes = len(self.node_ips())
            if current_nodes - prev_nodes == added_nodes:
                return True
            time.sleep(1)
        raise RuntimeError("Could not update nodes to the cluster.")

    def kill_nodes(self, n_nodes: int | None = None):
        active_containers = self._active_containers()

        prev_nodes = len(self.node_ips())
        if n_nodes is None:
            n_nodes = len(active_containers)
        for i in list(active_containers.keys())[:n_nodes]:
            self.api_client.kill(active_containers[i])
        self._wait_nodes(prev_nodes, -1 * n_nodes)


@pytest.fixture(scope="session")
def main_ray_cluster(working_dir, pytestconfig):
    assert not ray.is_initialized(), "Can not run tests with ray initialized."
    docker_tag = pytestconfig.getoption("--docker-tag")
    cluster = DockerRayCluster(working_dir=working_dir, docker_tag=docker_tag)
    cluster.setUp()
    yield cluster
    cluster.tearDown()


@pytest.fixture(scope="function")
def gpu_lock(tmpdir_factory):
    tmp_path = Path(tmpdir_factory.getbasetemp())
    if not torch.cuda.is_available():
        yield False
    else:
        with FileLock(tmp_path.joinpath(".gpu")):
            yield True


def _gpu_manager(gpu_lock, ray_cluster, timeout=5):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        devices = copy.deepcopy(os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        devices = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    with mock.patch("ablator.utils.base._get_gpu_info", lambda: _get_gpu_info()[:2]):
        yield GPUManager.remote(timeout, [0, 1])
    os.environ["CUDA_VISIBLE_DEVICES"] = devices


@pytest.fixture(scope="function")
def gpu_manager(gpu_lock, ray_cluster):
    yield from _gpu_manager(gpu_lock, ray_cluster)


@pytest.fixture(scope="function")
def very_patient_gpu_manager(gpu_lock, ray_cluster):
    yield from _gpu_manager(gpu_lock, ray_cluster, 10000000)


@pytest.fixture(scope="function")
def ray_cluster(tmpdir_factory, main_ray_cluster: DockerRayCluster, gpu_lock):
    tmp_path = Path(tmpdir_factory.getbasetemp())
    with FileLock(tmp_path.joinpath(".ray_cluster")):
        main_ray_cluster.setUp()
        assert len(main_ray_cluster.node_ips()) == main_ray_cluster.nodes + 1
        yield main_ray_cluster
        if not ray.is_initialized():
            logging.warn(
                "Shutting down ray during a test, can cause slow-down to set-up the cluster again. "
            )
            main_ray_cluster.tearDown()


@pytest.fixture(scope="function", autouse=True)
def seed():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)


class MPScheduler(LoadScopeScheduling):
    # NOTE must schedule all tests that use ray in the same node because of concurrency problems
    # when having interacting ray instances.
    def _split_scope(self, nodeid):
        # Since these tests depend on DockerRayCluster we schedule
        # them together. They all perform a `FileLock` and
        # can not run concurrently.
        file_names = [
            "mp/test_node_manager.py",
            "mp/test_main.py",
            "modules/test_file_logger.py",
            "analysis/test_analysis.py",
            "mp/test_gpu_manager.py",
        ]
        if any(f in nodeid for f in file_names):
            self.log(f"Scheduling {nodeid} with mp-tests.")
            return "mp-tests"
        return nodeid


def pytest_xdist_make_scheduler(config, log):
    return MPScheduler(config, log)

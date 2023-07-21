import copy
import io
import platform
import random
import re
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import docker
import numpy as np
import pytest
import ray
import torch
from xdist.scheduler.loadscope import LoadScopeScheduling

from ablator import package_dir

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
        img, command="/bin/bash", detach=True, tty=True, stdin_open=True, **cuda_args
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


def build_docker_image(docker_client: docker.DockerClient):
    py_version = platform.python_version()
    img, *_ = docker_client.images.build(
        nocache=False,
        path=package_dir.parent.as_posix(),
        tag="ablator",
        buildargs={"PY_VERSION": py_version},
    )
    return img


def get_docker_image(docker_client):
    # NOTE the ray and python version must match between nodes.
    # try:
    # Currently building the image is too slow, we expect it to have been pre-build
    return docker_client.images.get(DOCKER_TAG)
    # except:
    #     return build_docker_image(docker_client)


class DockerRayCluster:
    def __init__(self, nodes=2, build=False) -> None:
        self.nodes = nodes
        self.build = build

    def tearDown(self):
        self.kill_all()

    def setUp(self, working_dir):
        self.ray_was_init = False
        self.client = docker.from_env()
        if not ray.is_initialized():
            ray_cluster = ray.init(
                address="local", runtime_env={"working_dir": working_dir}
            )
            self.cluster_ip = ray_cluster.address_info["node_ip_address"]
            self.cluster_address = ray_cluster.address_info["address"]
            self.ray_was_init = True
        else:
            ray_cluster = ray.get_runtime_context()
            self.cluster_address = ray_cluster.gcs_address
            self.cluster_ip, _ = self.cluster_address.split(":")
        self.img = get_docker_image(self.client)
        self.cs = {}
        self.append_nodes(self.nodes)
        time.sleep(1)

    @classmethod
    def system_clean(cls, clean_ray: bool):
        try:
            api_client = docker.APIClient()
        except Exception as e:
            raise RuntimeError(
                "Could not find a docker installation. Please make sure docker is in Path and can run be run by the current system user (non-root) e.g. `docker run hello-world`. Please refer to https://github.com/fostiropoulos/ablator/blob/main/DEVELOPER.md for detailed instructions."
            ) from e
        for c in api_client.containers(filters={"ancestor": DOCKER_TAG}):
            api_client.kill(c)
        if clean_ray and ray.is_initialized():
            ray.shutdown()

    def node_ips(self):
        return copy.deepcopy(list(self.cs.keys()) + [self.cluster_ip])

    def append_nodes(self, n=2):
        for _ in range(n):
            c, ip = make_node(self.client, self.img, self.cluster_address)
            self.cs[ip] = c

    def kill_node(self, idx):
        ip = list(self.cs.keys())[idx]
        self.cs[ip].kill()
        del self.cs[ip]

    def kill_all(self):
        for v in self.cs.values():
            v.kill()
        self.cs = {}

@pytest.fixture(scope="function")
def docker_ray_cluster():
    return DockerRayCluster

@pytest.fixture(scope="function")
def ray_cluster():
    if ray.is_initialized():
        ray.shutdown()
    cluster = DockerRayCluster()
    cluster.setUp(Path(__file__).parent)
    yield cluster
    cluster.tearDown()
    ray.shutdown()


@pytest.fixture(scope="function", autouse=True)
def seed():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)


class MPScheduler(LoadScopeScheduling):
    # NOTE must schedule all tests that use ray in the same node because of concurrency problems
    # when having interacting ray instances.
    def _split_scope(self, nodeid):
        file_names = ["test_node_manager.py", "test_mp.py", "test_file_logger.py", "test_analysis.py"]
        if any(f in nodeid for f in file_names):
            self.log(f"Scheduling {nodeid} with mp-tests.")
            return "mp-tests"
        return nodeid


def pytest_xdist_make_scheduler(config, log):
    return MPScheduler(config, log)

import copy
import inspect
import io
import logging
import os
import platform
import random
import re
import subprocess
import time
import typing as ty
from collections import abc
from contextlib import redirect_stderr, redirect_stdout
from multiprocessing import Lock
from pathlib import Path
from platform import uname

import docker
import numpy as np
import pytest
import ray
import torch
from ray.util.state import list_nodes
from xdist.scheduler.loadscope import LoadScopeScheduling

import ablator
from ablator import package_dir
from ablator.mp.utils import ray_init

IS_LINUX = (
    "microsoft-standard" not in uname().release
    and "darwin" not in platform.system().lower()
)

DOCKER_TAG = "ablator"
pytest_plugins = ["test_plugins.model"]

cuda_lock = Lock()

ray_lock = Lock()


def _assert_error_msg(fn, error_msg=None):
    try:
        fn()
        raise RuntimeError(
            f"{fn} did not cause an error with. Expected message: {error_msg}"
        )
    except Exception as excp:
        if error_msg is not None and not error_msg == str(excp):
            raise RuntimeError(
                f"{fn} caused a different error. Expected message: {error_msg}\nFound"
                f" {str(excp)}"
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


def _capture_logger():
    out = io.StringIO()
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(out))
    return out


@pytest.fixture
def capture_logger():
    return _capture_logger


@pytest.fixture
def capture_output():
    return _capture_output


def make_node(docker_client: docker.DockerClient, img, cluster_address):
    aux_args = {}
    driver_args = {"count": -1, "capabilities": [["gpu"]], "driver": "nvidia"}
    if torch.cuda.is_available():
        aux_args = dict(
            runtime="nvidia",
            device_requests=[driver_args],
        )
    c = docker_client.containers.run(
        img,
        command="/bin/bash",
        detach=True,
        tty=True,
        stdin_open=True,
        pid_mode="host",
        cap_add=["SYS_ADMIN"],
        devices=["/dev/fuse"],
        security_opt=["apparmor:unconfined"],
        **aux_args,
    )
    res = c.exec_run("service ssh start")
    assert "Starting OpenBSD Secure Shell server" in res.output.decode()
    res = c.exec_run(f"ray start --num-gpus=4 --address='{cluster_address}'")
    _out = res.output.decode()
    if "Ray runtime started." not in _out:
        raise RuntimeError(
            "Unable to start ray cluster. Potential reasons are different "
            "python versions. Please consult DEVELOPER.MD in the main repo "
            f"and inspect log: {_out}"
        )

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
        "--test-suite",
        choices=["mp", "fast", "remote"],
        default=None,
        help="run mp tests only",
    )
    parser.addoption(
        "--docker-tag",
        action="store",
        default="ablator",
        help="the docker tag to use for launching a machine.",
    )
    parser.addoption(
        "--build",
        action="store_true",
        default=False,
        help=(
            "Whether to build the docker container used for testing prior to running"
            " the tests."
        ),
    )
    parser.addoption(
        "--volume-name",
        action="store",
        default=None,
        help=(
            "the volume name to optionally use. Used only for docker-to-docker tests"
            " with shared volumes"
        ),
    )


@pytest.fixture
def volume_name(pytestconfig):
    return pytestconfig.getoption("--volume-name")


def should_skip(option_flag: str, markers: list[str], args: list[str]):
    # mp_args = ["main_ray_cluster", "ray_cluster"]
    is_marked = not any(m == option_flag for m in markers)
    # if option_flag == "mp" and any(arg in mp_args for arg in args):
    #     is_marked = False
    if is_marked:
        return True
    else:
        return False


def pytest_collection_modifyitems(config, items):
    option_flag = config.getoption("--test-suite")

    if option_flag is None:
        print(f"\nRunning: {len(items)} / {len(items)} tests.")
        return
    run_items = []
    for item in items:
        args = item._fixtureinfo.argnames
        markers = list(n.name for n in item.iter_markers())
        if not any(m in ["mp", "remote"] for m in markers):
            markers.append("fast")

        if should_skip(option_flag, markers=markers, args=args):
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        f"Test markers: {markers} does not contain the flag"
                        f" {option_flag}."
                    )
                )
            )
        else:
            run_items.append(item)
    print(f"\nRunning: {len(run_items)} / {len(items)} tests.")


def build_docker_image(docker_tag):
    # No build-kit support when using `docker_client`
    # to build the image.
    py_version = platform.python_version()
    p = subprocess.run(
        [
            "docker",
            "build",
            package_dir.parent.as_posix(),
            "--tag",
            docker_tag,
            f'--build-arg="PY_VERSION={py_version}"',
        ],
        capture_output=False,
    )
    if p.returncode != 0:
        logging.error("`docker build` encountered an error.")
        if p.stderr is not None:
            logging.error(p.stderr.decode())


def get_docker_image(docker_client, docker_tag):
    # NOTE the ray and python versions must match between nodes and local enviroment.
    return docker_client.images.get(docker_tag)


class DockerRayCluster:
    def __init__(
        self,
        working_dir: str,
        cluster_address=None,
        nodes=1,
        docker_tag=DOCKER_TAG,
        build=False,
    ) -> None:
        self.nodes = nodes
        self.working_dir = working_dir
        # TODO check if bug is fixed. The reason we turn off multi-node cluster for
        # WSL tests is that ray nodes die randomly
        if not IS_LINUX and nodes > 1:
            raise RuntimeError(
                "Does not support multi-node cluster environment on Windows."
            )
        if cluster_address is None:
            cluster_address = ray_setup(working_dir)
        self.cluster_address = cluster_address
        self.docker_tag = docker_tag
        if build:
            build_docker_image(docker_tag)
        try:
            self.api_client = docker.APIClient()
            self.client = docker.from_env()
        except Exception as e:
            raise RuntimeError(
                "Could not find a docker installation. Please make sure docker is in"
                " Path and can be run by the current system user (non-root) e.g."
                " `docker run hello-world`. Please refer to"
                " https://github.com/fostiropoulos/ablator/blob/main/DEVELOPER.md for"
                " detailed instructions."
            ) from e

    def tearDown(self):
        self.kill_nodes()
        ray.shutdown()

    def setUp(self):
        if not ray.is_initialized():
            self.kill_nodes()
            self.cluster_address = ray_setup(self.working_dir)
        self.img = get_docker_image(self.client, self.docker_tag)
        if self.nodes > 0:
            node_diff = self.nodes - len(self.node_ips())  # + 1 for the head.
            if node_diff > 0:
                self.append_nodes(node_diff)
            elif node_diff < 0:
                self.kill_nodes(int(node_diff * -1))
        return self

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
        except Exception:
            return []

    def append_nodes(self, n):
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


def get_main_ray_cluster(
    docker_tag: str, cluster_address: str, build: bool, working_dir: str
) -> DockerRayCluster:
    n_nodes = 2 if IS_LINUX else 1
    cluster = DockerRayCluster(
        nodes=n_nodes,
        docker_tag=docker_tag,
        cluster_address=cluster_address,
        build=build,
        working_dir=working_dir,
    )
    return cluster


@pytest.fixture(scope="session", autouse=True)
def is_good_os():
    if os.name == "nt":
        raise RuntimeError(
            "Can not run tests on Windows. Please consult DEVELOPER.md from the main"
            " repo."
        )
    elif "microsoft-standard" in uname().release:
        logging.warn(
            "Running LIMITED tests due to poor compatibility of Windows with Ray and"
            " Multi-Node environments."
        )


def ray_setup(working_dir):
    # config is borrowed from ray.tests.test_gcs_fault_tolerance.py
    if not ray.is_initialized():
        timeout_config = dict(
            gcs_failover_worker_reconnect_timeout=2,
            gcs_rpc_server_reconnect_timeout_s=2,
            health_check_initial_delay_ms=2000,
            health_check_period_ms=1000,
            health_check_timeout_ms=1000,
            health_check_failure_threshold=5,
        )
        ray_kwargs = dict(
            address="local",
            runtime_env={"working_dir": working_dir, "py_modules": [ablator]},
            _system_config=timeout_config,
        )
        ray_cluster = ray_init(**ray_kwargs)
        cluster_address = ray_cluster["gcs_address"]
    else:
        ray_cluster = ray.get_runtime_context()
        cluster_address = ray_cluster.gcs_address
    return cluster_address


@pytest.fixture(scope="session", autouse=True)
def main_ray_cluster(working_dir, pytestconfig, tmp_path_factory):
    docker_tag = pytestconfig.getoption("--docker-tag")
    build = pytestconfig.getoption("--build")
    subprocess.run(
        'mount -l -t fuse.rclone | grep %s | awk -F " " \'{print "fusermount -u " $3}\''
        " | bash"
        % tmp_path_factory.getbasetemp(),
        shell=True,
    )
    cluster_address = ray_setup(working_dir)
    cluster = get_main_ray_cluster(
        docker_tag=docker_tag,
        cluster_address=cluster_address,
        build=build,
        working_dir=working_dir,
    )
    cluster.setUp()
    yield cluster
    cluster.tearDown()
    ray.shutdown()


@pytest.fixture(scope="function")
def ray_cluster(main_ray_cluster: DockerRayCluster):
    with ray_lock:
        main_ray_cluster.setUp()
        assert len(main_ray_cluster.node_ips()) == main_ray_cluster.nodes
        yield main_ray_cluster


@pytest.fixture(scope="function", autouse=True)
def seed():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)


class MPScheduler(LoadScopeScheduling):
    # NOTE must schedule flaky tests that use ray in the same
    # worker because of concurrency problems when having many
    # ray instances running at the same time.
    def _split_scope(self, nodeid):
        # Since these tests depend on DockerRayCluster we schedule
        # them together. They all perform a `Lock` and
        # can not run concurrently.
        file_names = ["test_end_to_end.py"]
        if any(f in nodeid for f in file_names):
            self.log(f"Scheduling {nodeid} with mp-tests.")
            return "mp-tests"
        return nodeid


def pytest_xdist_make_scheduler(config, log):
    return MPScheduler(config, log)


def fns_requires_kwargs(
    test_fns: list[abc.Callable], *kwarg_names, **existing_kwargs
) -> bool:
    """
    Check whether the fns require any of the `kwargs_names` that is not
    present in the `existing_kwargs`. Used when a kwarg_name is missing to
    create it.

    Parameters
    ----------
    test_fns : list[abc.Callable]
        the functions inspect and determine their arguments
    kwarg_names : tuple[str]
        name of the kwarg arguments to find in the function signature
    existing_kwargs : dict[str, ty.Any]
        the existing kwargs provided

    Returns
    -------
    bool
        whether any of the kwarg_names is required by any of the test_fns.
    """
    return any(
        [
            p in list(kwarg_names) and p not in existing_kwargs
            for fn in test_fns
            for p in inspect.signature(fn).parameters
        ]
    )


def run_tests_local(
    test_fns: list[abc.Callable],
    kwargs: dict[str, ty.Any] = None,
    unpickable_kwargs: dict[str, ty.Any] = None,
):
    """
    Meant as a helper function for debugging tests. This helper also
    contains logic on how the tests are expected to run and how
    ablator is meant to be used. i.e. we first set-up the cluster and
    then run experiments. Essentially re-implements `pytest` but with
    the ability to run using a debugger. It can be difficult to debug
    tests and test-fixtures using debugging in pytest.

    Parameters
    ----------
    test_fns : list[abc.Callable]
        the functions to test
    kwargs : dict[str, ty.Any], optional
        the kwargs to pass to the functions. If a key matches that of the
        function signature parameter it is passed. These arguments are `deepcopy`'d for
        each function call, by default None
    unpickable_kwargs : dict[str, ty.Any], optional
        Similar to `kawrgs` but these kwargs can not be `deepcopy`d since they are not pickable
        hence they are passed unchanged on every test_fn call, by default None

    Raises
    ------
    ValueError
        If there is a missing keyword-argument in the `kwargs`
    """
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    import shutil

    from tests.test_plugins.model import (
        WORKING_DIR,
        _blocking_lock_remote,
        _locking_remote_fn,
        _make_config,
        _remote_fn,
    )

    if kwargs is None:
        kwargs = {}

    if unpickable_kwargs is None:
        unpickable_kwargs = {}
    cluster_address = ray_setup(Path(WORKING_DIR).parent)
    if fns_requires_kwargs(test_fns, "ray_cluster", **unpickable_kwargs):
        n_nodes = 2 if IS_LINUX else 1
        ray_cluster = DockerRayCluster(
            nodes=n_nodes,
            build=False,
            cluster_address=cluster_address,
            working_dir=Path(WORKING_DIR),
        )
        ray_cluster.setUp()
        unpickable_kwargs["ray_cluster"] = lambda: ray_cluster.setUp()

    for fn in test_fns:
        parameters = inspect.signature(fn).parameters

        tmp_path = Path("/tmp/test_exp")
        subprocess.run(
            'mount -l -t fuse.rclone | grep %s | awk -F " " \'{print "fusermount -u "'
            " $3}' | bash" % tmp_path,
            shell=True,
        )
        default_kwargs = {
            "tmp_path": tmp_path,
            "assert_error_msg": _assert_error_msg,
            "capture_output": _capture_output,
            "working_dir": WORKING_DIR,
            "make_config": _make_config,
            "remote_fn": _remote_fn,
            "locking_remote_fn": _locking_remote_fn,
            "capture_logger": _capture_logger,
            "blocking_lock_remote": _blocking_lock_remote,
        }

        if hasattr(fn, "pytestmark"):
            for mark in fn.pytestmark:
                if mark.name == "parametrize":
                    k, v = mark.args
                    default_kwargs[k] = v

        for k, v in kwargs.items():
            unpickable_kwargs[k] = lambda: copy.deepcopy(kwargs[k])

        _run_args = [{}]

        for k, v in parameters.items():
            if k in unpickable_kwargs:
                continue
            if k not in default_kwargs and v.default != inspect._empty:
                default_kwargs[k] = v.default
                for _args in _run_args:
                    _args[k] = default_kwargs[k]
            elif k not in default_kwargs:
                raise ValueError(f"Missing kwarg {k}.")
            elif isinstance(default_kwargs[k], (list, tuple, set, dict)):
                __run_args = copy.deepcopy(_run_args)
                _run_args = []
                for _args in __run_args:
                    for _v in default_kwargs[k]:
                        _args[k] = _v
                        _run_args.append(copy.deepcopy(_args))
            else:
                for _args in _run_args:
                    _args[k] = default_kwargs[k]

        for _args in _run_args:
            subprocess.run(
                'mount -l -t fuse.rclone | grep %s | awk -F " " \'{print "fusermount -u'
                " \" $3}' | bash" % tmp_path,
                shell=True,
            )
            shutil.rmtree(tmp_path, ignore_errors=True)
            tmp_path.mkdir()
            for k, v in unpickable_kwargs.items():
                if k in parameters:
                    _args[k] = unpickable_kwargs[k]()
            fn(**_args)


if __name__ == "__main__":
    # assert not should_skip("fast", ["xx", "xb", "fast"], [])
    assert should_skip("fast", ["mp", "xb"], [])
    assert should_skip("fast", ["xx", "xb"], [])
    assert not should_skip("mp", ["mp", "xb"], [])
    assert should_skip("mp", ["xx"], [])
    assert should_skip(
        "remote", ["mp"], ("tmp_path", "ray_cluster", "update_gpus_fixture", "n_gpus")
    )
    assert not should_skip(
        "remote",
        ["remote"],
        ("tmp_path", "ray_cluster", "update_gpus_fixture", "n_gpus"),
    )

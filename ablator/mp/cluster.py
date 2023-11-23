import copy
import logging
import subprocess
import time
import traceback
import typing as ty
from collections import OrderedDict, abc
from pathlib import Path

import ray

from ablator.config.remote import RemoteConfig
from ablator.mp.gpu import GPU, Resource, ResourceManager, wait_get_gpu
from ablator.mp.heart import Heart
from ablator.mp.node import (
    NODE_ID,
    NODE_IP,
    Node,
    RayNode,
    get_ray_nodes,
    run_actor_node,
    run_lambda_node,
)
from ablator.mp.utils import (
    get_ray_tasks,
    sort_resources,
    register_public_key,
    get_node_id,
    get_node_ip,
    get_ray_address,
    get_username,
    make_private_key,
    private_key_to_str,
    ray_init,
    run_ssh_cmd,
)

DEFAULT_TIMEOUT = 60
MISSED_BEATS_LIMIT = 3


# pylint: disable=too-many-instance-attributes
class ClusterManager(Heart):
    """
    a ray cluster manager that provides an interface for interacting
    with ray nodes. It automatically discovers ray nodes, set-ups
    ssh access to them, a shared mountable file-system. The cluster
    manager provides the ability to detect available resources and utilization
    on the machine and execute commands. It implements a fault-tolerant
    strategy where it restarts failed nodes. The manager should work for all
    systems out-of-the-box, however when a new node is connected to the ray cluster
    several assumptions are made, such that there is system compatibility with the
    head node and rmount is available on the remote node. Due to limited
    support of libraries and tools, multi-node clusters are only supported for Linux.

    Parameters
    ----------
    private_key_home : Path
        the path of where to store the private keys to the node.
    sync_directory : Path
        the synchronization directory.
    ray_address : str | None, optional
        the ray address of the cluster, by default None
    remote_config : RemoteConfig | None, optional
        the remote configuration used to synchronize files between cluster
        nodes. When left unspecified, an ssh configuration will be automatically created
        to the master node `sync_directory`, by default None
    timeout : int, optional
        a timeout by which to terminate, by default ``DEFAULT_TIMEOUT``
    update_interval : int, optional
        the interval by which to discover nodes and update them, by default 10

    Attributes
    ----------
    remote_config : RemoteConfig | None
        the remote configuration used to synchronize the artifacts between
        nodes.
    sync_directory : Path
        the directory of where to synchronize the artifacts
    healthy_nodes : list[Node]
        the list of healthy Nodes
    head_id : str
        the ID of the head node
    head_ip : str
        the IP of the head node relative to the cluster.
    head_resources : Resource
        the resources available on the head node.
    node_ips : list[str]
        the IPs of the ray nodes in the cluster
    available_resources : dict[NODE_IP, Resource]
        the available resources on the nodes.

    Raises
    ------
    RuntimeError
        when there is a mismatch between the ray address provided and the one found via
        the Python interface.
    """

    def __init__(
        self,
        private_key_home: Path,
        sync_directory: Path,
        ray_address: str | None = None,
        remote_config: RemoteConfig | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        update_interval: int = 10,
    ):
        self._private_key, self._public_key = make_private_key(private_key_home)
        if (
            ray.is_initialized()
            and ray_address is not None
            and ray_address != get_ray_address()
        ):
            raise RuntimeError(
                "`ray_address` does not match the currently running ray instance. Can"
                " not initialize ray twice."
            )
        if not ray.is_initialized():
            ray_init(address=ray_address)

        self._timeout = timeout
        self.ray_address = get_ray_address()
        self._head_node = RayNode(
            node_ip=get_node_ip(), node_id=get_node_id(), is_alive=True
        )
        self._username = get_username()
        self._healthy_nodes: list[Node] = []  # must be unique in terms of IP
        self._init_nodes: list[NODE_ID] = []  # must be unique in terms of ID
        self.remote_config: RemoteConfig | None = remote_config
        self.sync_directory = Path(sync_directory)
        self._resource_actor = run_actor_node(
            ResourceManager,
            node=self.head_ip,
            cuda=False,
            kwargs={
                "node_ip": self.head_ip,
                "resource_lock_timeout": self._timeout,
                "update_interval": update_interval,
                "ray_address": self.ray_address,
            },
        )
        self._update_interval = update_interval
        super().__init__(missed_heart_beats=MISSED_BEATS_LIMIT)

    @property
    def healthy_nodes(self) -> list[Node]:
        return copy.copy(self._healthy_nodes)

    @property
    def head_id(self) -> str:
        return self._head_node.node_id

    @property
    def head_ip(self):
        return self._head_node.node_ip

    def _make_remote_config(self) -> RemoteConfig:
        """
        We set-up access of the head-node via ssh for all nodes.
        We create a remote config that allows any other Node to connect to the head node to
        store artifacts via rmount. This is to sync files between all nodes.

        Returns
        -------
        RemoteConfig
            A remote configuration

        Raises
        ------
        RuntimeError
            If calling the function directly or more than once.
        RuntimeError
            If it is unable to set-up access via ssh on the head node.
        """
        if self.remote_config is not None:
            raise RuntimeError("Can not call _make_remote_config directly")

        register_public_key(self._public_key)
        key_pem = private_key_to_str(self._private_key)

        result: str = run_ssh_cmd(  # type: ignore[assignment]
            self.head_ip,
            self._username,
            self._private_key,
            cmd="whoami",
            timeout=self._timeout,
        )
        if result.strip("\n") != self._username:
            raise RuntimeError(
                "Misconfigured head node. Please make sure ssh is running on the "
                "ray cluster head node and that it is accessible on port 22. "
                f"i.e. `ssh {self._username}@{self.head_ip}`"
            )
        return RemoteConfig(
            ssh={
                "host": self.head_ip,
                "user": self._username,
                "port": 22,
                "key_pem": key_pem,
            },
            remote_path=self.sync_directory,
            local_path=self.sync_directory,
        )

    def _discover_nodes(self, timeout: int | None = None) -> list[Node]:
        if timeout is None:
            timeout = self._timeout
        discovered_nodes: list[Node] = []

        # NOTE this should never in principle run for the `head_node`.
        # the head-node should always be a healthy node and we keep track of
        # it with a dedicated variable.
        healthy_ids = [n.node_id for n in self._healthy_nodes if n.is_alive()]
        for ray_node in get_ray_nodes(
            self.ray_address,
            timeout=timeout,
            exclude_node_id=self.head_id,
        ):
            node_ip = ray_node.node_ip
            if (node_id := ray_node.node_id) not in healthy_ids:
                # a newly discovered node (could also be a previously dead node)
                if self.remote_config is None:
                    self.remote_config = self._make_remote_config()
                try:
                    node = Node(
                        node_ip=node_ip,
                        node_id=node_id,
                        private_key=self._private_key,
                        public_key=self._public_key,
                        remote_config=self.remote_config,
                        timeout=timeout,
                        update_interval=self._update_interval,
                    )
                    discovered_nodes.append(node)
                # pylint: disable=broad-exception-caught
                except Exception as e:
                    logging.error(
                        "Could not update node with %s. %s %s",
                        node_ip,
                        str(e),
                        traceback.format_exc(),
                    )

            else:
                idx = healthy_ids.index(node_id)
                node = self._healthy_nodes[idx]
                discovered_nodes.append(node)
        return discovered_nodes

    def _filter_discovered_nodes(self, discovered_nodes: list[Node]) -> list[Node]:
        parsed_node_ips = [n.node_ip for n in discovered_nodes]
        for node in self._healthy_nodes:
            if node.node_ip not in parsed_node_ips:
                node.restart()
        dead_node_idxs = []
        for node in discovered_nodes:
            if not node.is_alive():
                dead_node_idxs.append(node.node_id)
                node.restart()
        return discovered_nodes

    def heartbeat(self, timeout: int | None = None):
        if timeout is None:
            timeout = self._timeout
        # Step 1. find actively connected nodes to the cluster
        discovered_nodes = self._discover_nodes(timeout)
        # Step 2. find which nodes are not healthy. Based on:
        # 1. Not discovered in the ray cluster
        # 2. Not found to be mounted correctly.
        self._healthy_nodes = self._filter_discovered_nodes(discovered_nodes)

    def get_gpu(self, node_ip: str, process_name: str) -> tuple[GPU, ResourceManager]:
        """
        reserves a GPU on the `node_ip` for `process_name`.

        Parameters
        ----------
        node_ip : str
            the node ip to reserver a process at.
        process_name : str
            the reserving process name

        Returns
        -------
        tuple[GPU, ResourceManager]
            the GPU resource reserved with the corresponding ResourceManager

        Raises
        ------
        RuntimeError
            when it is unable to reserve a GPU at the Node.
        """
        actors = [(self.head_ip, self._resource_actor)]
        actors += [(n.node_ip, n.resource_actor) for n in self._healthy_nodes]
        for ip, resource_actor in actors:
            if ip == node_ip:
                gpu = wait_get_gpu(
                    manager=resource_actor,
                    expected_util_mb=None,
                    process_name=process_name,
                    max_timeouts=self._timeout,
                )
                return gpu, resource_actor

        raise RuntimeError(
            f"Could not find {node_ip} in nodes {[ip for ip, _ in actors]}"
        )

    @property
    def head_resources(self) -> Resource:
        for _ in range(10):
            node_resources: Resource = ray.get(
                self._resource_actor.resources.remote(), timeout=self._timeout
            )
            if node_resources.is_active:
                break
            time.sleep(1)
        if not node_resources.is_active:
            raise RuntimeError("Could not read the resources of the head node.")
        # end
        running_tasks = get_ray_tasks(
            ray_address=self.ray_address,
            node_id=self.head_id,
            timeout=self._timeout,
        )
        node_resources.running_tasks = running_tasks
        return node_resources

    @property
    def node_ips(self) -> list[str]:
        return [v.node_ip for v in self._healthy_nodes]

    @property
    def available_resources(self) -> dict[NODE_IP, Resource]:
        resource_dict = {self.head_ip: self.head_resources} | {
            n.node_ip: n.resources for n in self._healthy_nodes
        }
        return {k: v for k, v in resource_dict.items() if v.is_active}

    def sorted_resources(
        self, gpu_mem: int | None = None
    ) -> OrderedDict[NODE_IP, Resource]:
        """
        sort resources by utilization and optionally exclude the ones that do not meet
        a minimum requirement e.g. for `gpu_mem`.

        Parameters
        ----------
        gpu_mem : int | None, optional
            the GPU memory required for scheduling a task, by default None

        Returns
        -------
        OrderedDict[NODE_IP, Resource]
            the Node IP with the corresponding Resource ordered by utilization from
            least to most busy.
        """
        return sort_resources(
            resources=self.available_resources,
            gpu_util_requirement=gpu_mem,
        )

    def run_lambda_head(
        self,
        fn: abc.Callable,
        fn_kwargs: dict[str, ty.Any] | None = None,
        run_async: bool = False,
        cuda: bool = False,
        name: str | None = None,
    ) -> ty.Any | ray.ObjectRef:
        """
        run a Python lambda function on the Node and scheduled
        using ray. When running asynchronously a reference to the ray
        object is provided. Otherwise, it will return the function output.

        Parameters
        ----------
        fn : abc.Callable
            the function to run. Must be pickable.
        fn_kwargs : dict[str, ty.Any] | None, optional
            the keyword arguments to pass to the function, by default None
        run_async : bool, optional
            whether to wait for the lambda output or schedule asynchronously, by default False
        cuda : bool, optional
            whether the function requires CUDA, by default False
        name : str | None, optional
            the name of the scheduled function, by default None

        Returns
        -------
        ty.Any | ray.ObjectRef
            the function output or a reference to the ray object, depending on whether
            running asynchronously.
        """
        if fn_kwargs is None:
            fn_kwargs = {}
        return run_lambda_node(
            fn=fn,
            cuda=cuda,
            fn_kwargs=fn_kwargs,
            node=self.head_ip,
            timeout=self._timeout,
            run_async=run_async,
            name=name,
        )

    def run_cmd_head(
        self,
        cmd: str,
        run_async: bool = False,
    ) -> str | subprocess.Popen:
        """
        executes a bash command on the Node and optionally returns the output.

        Parameters
        ----------
        cmd : str
            the bash command to execute on the Node.
        run_async : bool, optional
            whether to wait for the command output, by default False

        Returns
        -------
        str | subprocess.Popen
            the stdout of the command or the background running process
        """
        # we purposefully leave it open for `run_async` option
        # pylint: disable=consider-using-with
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        )
        if run_async:
            return p

        stdout, stderr = p.communicate(timeout=self._timeout)
        err = stderr.decode().strip("\n")
        if len(err) > 0:
            logging.error(err)
        return stdout.decode().strip("\n")

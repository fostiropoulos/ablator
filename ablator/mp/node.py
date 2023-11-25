import logging
import time
import traceback
import typing as ty
from collections import abc
from dataclasses import dataclass
from pathlib import Path

import ray
from paramiko import PKey
from ray import exceptions as ray_exc
from ray.util.state import list_nodes
from ablator.config.remote import RemoteConfig
from ablator.mp.gpu import Resource, ResourceManager
from ablator.mp.utils import (
    register_public_key,
    get_ray_address,
    get_ray_tasks,
    run_ssh_cmd,
    run_lambda,
)

NODE_TYPE = ty.Union[str, "Node", "RayNode", None]


DEFAULT_TIMEOUT = 60
NODE_IP = str
NODE_ID = str


class DuplicateNodes(RuntimeError):
    """
    Error for when several nodes with the same IP but different
    ID are found connected on the cluster.
    """


@dataclass
class RayNode:
    """
    A RayNode that encompasses basic attributes
    useful to identifying and managing it.

    Attributes
    ----------
    node_ip: NODE_IP
        the IP of the node relative to the cluster.
    node_id: NODE_ID
        the ID of the ray cluster node
    is_alive: bool
        whether the node is alive.
    """

    node_ip: NODE_IP
    node_id: NODE_ID
    is_alive: bool


def run_actor_node(
    actor: type,
    cuda: bool,
    node: NODE_TYPE = None,
    kwargs: dict[str, ty.Any] | None = None,
) -> ray.ObjectRef:
    """
    schedules a class as a ray actor on the specified node.

    Parameters
    ----------
    actor : type
        the class to schedule as a ray actor.
    cuda : bool
        whether cuda should be used to schedule the actor
    node : NODE_TYPE, optional
        the node to schedule the actor at, by default None
    kwargs : dict[str, ty.Any] | None, optional
        keyword arguments supplied during the initialization
        of the actor, by default None

    Returns
    -------
    ray.ObjectRef
        a ray reference to the actor.
    """
    return run_lambda_node(
        fn=actor, cuda=cuda, node=node, fn_kwargs=kwargs, run_async=True, max_calls=None
    )


def run_lambda_node(
    fn: abc.Callable,
    cuda: bool,
    node: NODE_TYPE = None,
    timeout: int | None = None,
    run_async: bool = False,
    fn_kwargs: dict[str, ty.Any] | None = None,
    name: str | None = None,
    max_calls: int | None = 1,
) -> ty.Any | ray.ObjectRef:
    """
    run a Python lambda function on the Node and scheduled
    using ray. When running asynchronously a reference to the ray
    object is provided. Otherwise, it will return the function output.
    This is identical to `ablator.mp.utils.run_lambda` but specified by
    Node instead of IP.

    Parameters
    ----------
    fn : abc.Callable
        the function to run. Must be pickable.
    cuda : bool, optional
        whether the function requires CUDA, by default False
    node : NODE_TYPE, optional
        the the node to run the ray function to, by default None
    timeout : int | None, optional
        the timeout to apply when waiting for the results, by default None
    run_async : bool, optional
        whether to wait for the lambda output or schedule asynchronously, by default False
    fn_kwargs : dict[str, ty.Any] | None, optional
        the keyword arguments to pass to the function, by default None
    name : str | None, optional
        the name of the scheduled function, by default None
    max_calls : int | None, optional
        used to de-allocate memory for remotes (but is not applicable for ray actors), by default 1

    Returns
    -------
    ty.Any | ray.ObjectRef
        the result of the function or a reference to the ray object.
    """
    if isinstance(node, (Node, RayNode)):
        node = node.node_ip
    return run_lambda(
        fn=fn,
        node_ip=node,
        timeout=timeout,
        run_async=run_async,
        cuda=cuda,
        fn_kwargs=fn_kwargs,
        name=name,
        max_calls=max_calls,
    )


def get_ray_nodes(
    ray_address: str | None = None,
    timeout: int | None = None,
    exclude_node_id: str | None = None,
) -> list["RayNode"]:
    """
    discovers the currently connected nodes that are also alive.

    Parameters
    ----------
    ray_address: str | None, optional
        The address of the ray cluster to discover nodes on.
        When left unspecified it automatically discovers based on
        the run-time context, by default ``None``.
    timeout: int | None, optional
        The timeout to apply when discovering nodes.
        When left unspecified it uses ``ablator.mp.node.DEFAULT_TIMEOUT``, by default ``None``.
    exclude_node_id: str | None, optional
        When specified it excludes the specified Node from the list of returned Nodes.
        Useful to exclude the head-node, by default ``None``.

    Returns
    -------
    list[RayNode]
        list of RayNode that are currently active.

    Raises
    ------
    DuplicateNodes
        when several nodes with the same IP are found. For example when initializing ray multiple times
        on the same machine.
    """
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    if ray_address is None:
        ray_address = get_ray_address()
    nodes: list[RayNode] = []
    node_ips = []
    for n in list_nodes(address=ray_address, timeout=timeout):
        state: str = n.state
        nid: NODE_ID = n.node_id
        nip: NODE_IP = n.node_ip
        is_alive = state.lower() == "alive"
        if is_alive and (exclude_node_id is None or nid != exclude_node_id):
            nodes.append(RayNode(node_id=nid, node_ip=nip, is_alive=is_alive))

    node_ips = [n.node_ip for n in nodes]
    if len(node_ips) > len(set(node_ips)):
        raise DuplicateNodes(
            "Several ray nodes were found with the same IP. This can lead to unexpected"
            " behavior and is not supported."
        )
    return sorted(nodes, key=lambda x: x.node_id)


class MountServer:
    """
    A mount-point actor meant to run in the background. Contrary
    to mounting to the local-path, it mounts based relative to the home
    directory in an `ablator` subfolder.
    This is because the MountServer is meant to be scheduled to run
    in a remote machine where we have no observability of the file-system.

    Parameters
    ----------
    config : RemoteConfig
        The remote configuration to use to set-up the mount-point.

    Raises
    ------
    ImportError
        when rmount is not supported for the current system.
    """

    def __init__(self, config: RemoteConfig) -> None:
        config.local_path = str(
            Path.home().joinpath("ablator", *Path(config.local_path).parts[1:])
        )
        self.config = config
        self.remote_path: Path = Path(config.remote_path)
        self.local_path: Path = Path(config.local_path)
        # MountServer would only execute for Linux
        # pylint: disable=import-outside-toplevel
        try:
            from rmount import RemoteMount
        except ImportError as e:
            raise ImportError(
                "remote_config is only supported for Linux systems."
            ) from e

        self.backend = RemoteMount(
            config.get_config(),
            remote_path=self.remote_path,
            local_path=self.local_path,
            verbose=False,
        )

    def get_local_path(self):
        return self.local_path

    def mount(self):
        self.backend.mount()

    def wait_alive(self):
        while not self.backend.is_alive():
            time.sleep(1)
        return True

    def unmount(self):
        self.backend.unmount()

    def remote_files(self):
        return list(self.local_path.rglob("*"))


class Node:
    """
    A Node that belongs to a cluster with an interface for running commands to the node
    as well as maintaining active resources like a mount-point connection.

    Attributes
    ----------
    node_ip : str
        the node ip address relative to the cluster head.
    node_id : str
        the node ray id for the specific cluster.
    ray_address : str
        the ray cluster address
    remote_dir : Path
        the remote directory where artifacts between the Node and the cluster
        are stored.
    mount_server : MountServer | None
        the ``MountServer`` actor is responsible for maintaining a connection
        between the Node and the cluster head.
    resource_actor : ResourceManager
        the ``ResourceManager`` responsible for keeping track of the available resources on the node.
    node_name : str
        the node name formatted as `username@node_ip`
    resources : Resource
        the resources available on the node at the time of the request and are updated in the background.

    Parameters
    ----------
    node_ip : str
        the ip to the cluster node.
    node_id : str
        the id of the cluster node
    private_key : PKey
        the private key used to connect and run commands
        to the node.
    public_key : str
        the public key corresponding to the private key
        that will be added to the node.
    remote_config : RemoteConfig
        the remote configuration of where to sync experiment artifacts
    timeout : int | None, optional
        a timeout after which it will thrown an error.
        When unspecified it is set to `ablator.mp.node.DEFAULT_TIMEOUT`, by default None
    update_interval : int, optional
        the interval by which to run a heart-beat on the node, in seconds, by default 10

    Raises
    ------
    RuntimeError
        when we are unable to initialize the required resources on the Node.
    """

    def __init__(
        self,
        node_ip: str,
        node_id: str,
        private_key: PKey,
        public_key: str,
        remote_config: RemoteConfig,
        timeout: int | None = None,
        update_interval: int = 10,
    ):
        self.node_ip: NODE_IP = node_ip
        self.node_id: NODE_ID = node_id
        self.ray_address = get_ray_address()
        self._private_key = private_key
        if timeout is None:
            timeout = DEFAULT_TIMEOUT
        self._timeout = timeout

        self._remote_config = remote_config
        # ... re-initing the node does not hurt.
        self._username = run_lambda_node(
            register_public_key,
            cuda=False,
            node=node_ip,
            timeout=timeout,
            fn_kwargs={"public_key": public_key},
        )
        self.remote_dir: Path
        self.mount_server: MountServer | None = None
        self.resource_actor = run_actor_node(
            ResourceManager,
            cuda=False,
            node=node_ip,
            kwargs={
                "node_ip": node_ip,
                "resource_lock_timeout": self._timeout,
                "update_interval": update_interval,
            },
        )

        for _ in range(timeout):
            if self.resources.is_active:
                break
            time.sleep(1)
        if not self.resources.is_active:
            raise RuntimeError(f"Could not initialize ResourceManager for {node_ip}")
        self.mount()

    @property
    def node_name(self) -> str:
        return f"{self._username}@{self.node_ip}"

    @property
    def resources(self) -> Resource:
        # pylint: disable=broad-exception-caught
        try:
            node_resources: Resource = ray.get(
                self.resource_actor.resources.remote(), timeout=self._timeout
            )
            running_tasks = get_ray_tasks(
                ray_address=self.ray_address,
                node_id=self.node_id,
                timeout=self._timeout,
            )
            node_resources.running_tasks = running_tasks
        except Exception:
            node_resources = Resource(
                mem=101,
                cpu_usage=[101],
                cpu_count=0,
                gpu_free_mem=[0],
                running_tasks=[],
                is_active=False,
            )
        return node_resources

    def mount(self) -> bool:
        """
        creates a background `mount_server` process.

        Returns
        -------
        bool
            whether it was able to successfully mount the server.
        """
        self.mount_server = run_actor_node(
            MountServer,
            cuda=False,
            node=self.node_ip,
            kwargs={"config": self._remote_config},
        )
        self.remote_dir = ray.get(self.mount_server.get_local_path.remote())  # type: ignore[union-attr]
        self.mount_server.mount.remote()  # type: ignore[union-attr]
        return self._get_is_mount_alive()

    def unmount(self):
        """
        unmounts the `mount_server`
        """
        if self.mount_server is not None:
            ray.get(self.mount_server.unmount.remote(), timeout=self._timeout)
            self.mount_server = None

    def _is_ray_alive(self):
        nodes = get_ray_nodes(ray_address=self.ray_address, timeout=self._timeout)
        node_ips = [n.node_ip for n in nodes]
        node_ids = [n.node_id for n in nodes]
        if self.node_ip not in node_ips or node_ips.count(self.node_ip) > 1:
            return False
        node_idx = node_ips.index(self.node_ip)
        if node_ids[node_idx] != self.node_id:
            self.node_id = node_ids[node_idx]
            logging.warning(
                (
                    "Node id was updated for node %s and could be a result of cluster"
                    " instability."
                ),
                self.node_ip,
            )
        return True

    def _get_is_mount_alive(self):
        if self.mount_server is not None:
            return ray.get(
                self.mount_server.wait_alive.remote(),
                timeout=self._timeout,
            )
        return False

    # pylint: disable=broad-exception-caught
    def _is_mount_alive(self):
        try:
            return self._get_is_mount_alive()
        except ray_exc.GetTimeoutError:
            logging.error("mount for %s is dead. ", self.node_ip)
        except Exception:
            logging.error(
                "Unknown mount error for %s %s ", self.node_ip, traceback.format_exc()
            )
        return False

    def is_alive(self) -> bool:
        """
        tests whether both ray and the mount point are alive.

        Returns
        -------
        bool
            only if ray and the mount point are alive.
        """
        return self._is_mount_alive() and self._is_ray_alive()

    def _restart_ray(self):
        old_node_id = self.node_id
        try:
            self.stop()
            start_cmd = f"ray start --address={self.ray_address}"
            self.run_cmd(start_cmd)
        except Exception:
            logging.error("Node %s error %s", self.node_ip, traceback.format_exc())
            return False
        for _ in range(self._timeout):
            is_alive = False
            try:
                is_alive = self._is_ray_alive()
            except DuplicateNodes:
                # happens because ray did not have the time to unregister the dead
                # node.
                ...
            if is_alive and old_node_id != self.node_id:
                # the same ID could be the result of not having enough
                # time to update the node.
                time.sleep(1)
                return True
            time.sleep(1)
        logging.error("Failed to restart ray for Node %s.", self.node_ip)
        return False

    def restart(self) -> bool:
        """
        it restarts the node by first stopping the node. Then restarting ray and
        finally remounting the node. During remounting it attempts to mount 3 times,
        after which it returns False.

        Returns
        -------
        bool
            whether it successfully restarted the Node.
        """
        if self._restart_ray():
            for _ in range(3):
                try:
                    self.mount()
                    return True
                except Exception:
                    time.sleep(1)
        return False

    def stop(self) -> bool:
        """
        terminates the current node by unmounting any resources and stopping ray.

        Returns
        -------
        bool
            whether it successfully stopped the Node.

        Raises
        ------
        TimeoutError
            When it can not stop ray and unmount within a `timeout`.

        """
        try:
            self.unmount()
            kill_cmd = (
                "kill -9 $(ps -ef | awk '/[r]aylet .*--node_ip_address=%s/{print $2}')"
                % self.node_ip
            )
            self.run_cmd(kill_cmd)
            for _ in range(self._timeout):
                if not self.is_alive():
                    return True
                time.sleep(1)
            raise TimeoutError("Could not stop on time.")
        except Exception:
            logging.error("Node %s error %s", self.node_ip, traceback.format_exc())
        return False

    def run_lambda(
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
            node=self.node_ip,
            timeout=self._timeout,
            run_async=run_async,
            name=name,
        )

    def run_cmd(
        self,
        cmd: str,
        run_async: bool = False,
    ) -> str | None:
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
        str | None
            the stdout of the command
        """
        return run_ssh_cmd(
            node_ip=self.node_ip,
            username=self._username,
            private_key=self._private_key,
            cmd=cmd,
            run_async=run_async,
            timeout=self._timeout,
        )

import copy
import getpass
import io
import logging
import os
import socket
import typing as ty
from collections import OrderedDict, abc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import paramiko
import psutil
import ray
import torch
from paramiko import PKey
from ray.util.state import list_tasks
from ray.runtime_context import RuntimeContext
from ablator.utils._nvml import get_gpu_mem


@dataclass
class Resource:
    """
    A Resource dataclass used to group resources together

    Attributes
    ----------
    mem : float
        percentage of total memory utilization
    cpu_usage : list[float]
        percentage of cpu usage per core
    cpu_count : int
        number of cores available on the system.
    gpu_free_mem : list[int]
        free GPU memory in MiB listed by ascending device number
    running_Tasks : list[str]
        the list of ray processes running on the system
    is_active : bool
        whether the resource is currently active
    """

    mem: float
    cpu_usage: list[float]
    cpu_count: int
    gpu_free_mem: list[int]
    running_tasks: list[str] = field(default_factory=lambda: [])
    is_active: bool = True


def ray_init(**kwargs: ty.Any) -> RuntimeContext:
    """
    initialize ray with some reasonable defaults

    Parameters
    ----------
    **kwargs : ty.Any
        the keyword arguments to provide to initialize ray with. For full details
        please consult https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html

    Returns
    -------
    RuntimeContext
        the ray run-time context
    """
    env_cuda = (
        "CUDA_VISIBLE_DEVICES" not in os.environ
        or os.environ["CUDA_VISIBLE_DEVICES"] != ""
    )
    sys_cuda = torch.cuda.is_available()
    remote_connect = (
        "address" not in kwargs
        or kwargs["address"] is None
        or kwargs["address"] == "local"
    )
    if env_cuda and sys_cuda and remote_connect:
        # this is because WSL and other systems work poorly
        # with ray.
        kwargs["num_gpus"] = 1
    elif "address" in kwargs:
        kwargs["ignore_reinit_error"] = True
    return ray.init(**kwargs)


def get_ray_address() -> str:
    """
    the current ray address

    Returns
    -------
    str
        the address of the current ray cluster
    """
    return ray.get_runtime_context().gcs_address


def get_username() -> str:
    """
    The executing process system username.

    Returns
    -------
    str
        the system username
    """
    return getpass.getuser()


def get_node_ip() -> str:
    """
    The executing process ray node IP

    Returns
    -------
    str
        the ray node IP
    """
    return ray.util.get_node_ip_address()


def get_node_id() -> str:
    """
    The executing process ray node ID

    Returns
    -------
    str
        the ray node ID
    """
    return ray.get_runtime_context().get_node_id()


def sort_resource_gpu_util(resources: list[Resource]) -> list[int]:
    """
    helper function to sort a list of ``Resource`` based on the free GPU memory.

    Parameters
    ----------
    resources : list[Resource]
        the resources to sort.

    Returns
    -------
    list[int]
        the sorted list of the index from the least to the most used ``Resource``
    """
    free_gpu = np.array(
        [
            np.max(_resources.gpu_free_mem) if len(_resources.gpu_free_mem) else 0
            for _resources in resources
        ]
    )
    if (free_gpu[0] == free_gpu).all():
        return list(np.array([np.nan] * len(free_gpu)))
    return list(np.argsort(free_gpu)[::-1])


def sort_resource_cpu_util(resources: list[Resource]) -> list[int]:
    """
    helper function to sort a list of ``Resource`` based on the CPU utilization.

    Parameters
    ----------
    resources : list[Resource]
        the resources to sort.

    Returns
    -------
    list[int]
        the sorted list of the index from the least to the most used ``Resource``
    """
    cpu_usage = np.array([np.mean(_resources.cpu_usage) for _resources in resources])
    if (cpu_usage[0] == cpu_usage).all():
        return list(np.array([np.nan] * len(cpu_usage)))
    return list(np.argsort(cpu_usage))


def sort_resource_mem_util(resources: list[Resource]) -> list[int]:
    """
    helper function to sort a list of ``Resource`` based on the memory utilization.


    Parameters
    ----------
    resources : list[Resource]
        the resources to sort.

    Returns
    -------
    list[int]
        the sorted list of the index from the least to the most used ``Resource``
    """
    mem_arr = np.array([_resources.mem for _resources in resources])
    if (mem_arr[0] == mem_arr).all():
        return list(np.array([np.nan] * len(mem_arr)))
    return list(np.argsort(mem_arr))


def sort_resource_task_util(resources: list[Resource]) -> list[int]:
    """
    helper function to sort a list of ``Resource`` based on the number of tasks running.

    Parameters
    ----------
    resources : list[Resource]
        the resources to sort.

    Returns
    -------
    list[int]
        the sorted list of index from the least to the most used ``Resource``
    """
    n_running_tasks = np.array(
        [len(_resources.running_tasks) for _resources in resources]
    )
    if (n_running_tasks[0] == n_running_tasks).all():
        return list(np.array([np.nan] * len(n_running_tasks)))
    return list(np.argsort(n_running_tasks))


def sort_resources_by_util(
    resources: dict[str, Resource], eval_gpu: bool
) -> OrderedDict[str, Resource]:
    """
    Sort resources equally weighing between cpu_util, mem_util, number of tasks running and
    gpu_util, if `eval_gpu=True`.

    Parameters
    ----------
    resources : dict[str, Resource]
        the resources to order
    eval_gpu : bool
        whether to evaluate the Resource by their GPU utilization

    Returns
    -------
    OrderedDict[str, Resource]
        A dictionary of the same resources sorted by utilization.
    """
    node_ips = list(resources.keys())
    resources_list = list(resources.values())
    sorted_resources = OrderedDict()

    for _ in range(len(node_ips)):
        usage_lists = []
        if eval_gpu:
            usage_lists.append(sort_resource_gpu_util(resources_list))
        usage_lists.append(sort_resource_cpu_util(resources_list))
        usage_lists.append(sort_resource_mem_util(resources_list))
        usage_lists.append(sort_resource_task_util(resources_list))
        np_usage_lists = np.array(usage_lists)
        # remove inconclusive
        np_usage_lists = np_usage_lists[~np.isnan(np_usage_lists).all(1)]
        # usage_list x node_ip grid
        if len(np_usage_lists) > 0:
            least_used_idx = np_usage_lists[:, 0]
            least_used_idx, least_used_freq = np.unique(
                least_used_idx, return_counts=True
            )

            idx = int(least_used_idx[np.argmax(least_used_freq)])
        else:
            idx = 0
        node_ip = node_ips[idx]
        sorted_resources[node_ip] = resources[node_ip]
        del resources_list[idx]
        del node_ips[idx]

    return sorted_resources


def sort_resources(
    resources: dict[str, Resource],
    gpu_util_requirement: int | None = None,
    memory_perc_limit: int = 80,
    cpu_util_perc_limit: int = 80,
) -> OrderedDict[str, Resource]:
    """
    Sorts the nodes based on their available resources from
    the least used to the most used node. If a node does not meet the `gpu_util_requirement` or
    `memory_perc_limit` and `cpu_util_perc_limit` it is excluded from the list.

    Parameters
    ----------
    resources : dict[str, Resource]
        a dictionary of the nodes with their available resources
    gpu_util_requirement : int | None
        the GPU requirement for the task, by default ``None``.
    memory_perc_limit : int
        the percentage upper limit to memory utilization, by default ``80``.
    cpu_util_perc_limit : int
        the percentage upper limit to CPU utilization, by default 80

    Returns
    -------
    OrderedDict[str, Resource]
        the sorted list of Node IPs arranged from the least to most used.
    """

    sorted_resources = sort_resources_by_util(
        resources, gpu_util_requirement is not None
    )

    def _should_sample(node_ip):
        ray_cluster_gpu_limit = gpu_util_requirement is None or any(
            np.array(resources[node_ip].gpu_free_mem) > gpu_util_requirement
        )
        ray_cluster_cpu_limit = (
            np.mean(resources[node_ip].cpu_usage) < cpu_util_perc_limit
        )
        ray_cluster_mem_limit = resources[node_ip].mem < memory_perc_limit
        return ray_cluster_mem_limit and ray_cluster_cpu_limit and ray_cluster_gpu_limit

    available_sorted_nodes = OrderedDict()
    for node_ip in copy.deepcopy(sorted_resources):
        if _should_sample(node_ip):
            available_sorted_nodes[node_ip] = sorted_resources[node_ip]
    return available_sorted_nodes


def make_private_key(home_path: Path) -> tuple[PKey, str]:
    """
    creates a private key named `ablator_id_rsa` that can be used
    for ablator specific functionality, such as SSH between cluster-nodes.

    Parameters
    ----------
    home_path : Path
        the home directory of where to store the private key. The private
        key is added in the folder `.ssh` and is named `ablator_id_rsa`

    Returns
    -------
    tuple[PKey, str]
        the private and public key that was added to `home_path`.
    """
    pkey_path = Path(home_path).joinpath(".ssh", "ablator_id_rsa")
    pkey_path.parent.mkdir(exist_ok=True)

    if not pkey_path.exists():
        pkey = paramiko.RSAKey.generate(bits=2048)
        with open(
            os.open(
                pkey_path.as_posix(),
                flags=(os.O_WRONLY | os.O_CREAT | os.O_TRUNC),
                mode=0o600,
            ),
            "w",
            encoding="utf-8",
        ) as p:
            pkey.write_private_key(p)
    else:
        pkey = paramiko.RSAKey.from_private_key_file(pkey_path.as_posix())
    name = pkey.get_name()
    public_key = pkey.get_base64()
    hostname = socket.gethostname()
    node_ip = socket.gethostbyname(hostname)
    key = f"{name} {public_key} ablator-{hostname}@{node_ip}"
    return pkey, key


def register_public_key(
    public_key: str,
) -> str:
    """
    registers a public-key to be used to access the same machine and
    current user via SSH. It adds the `public_key` to `authorized_keys`.

    Parameters
    ----------
    public_key : str
        the public key to add to the system.

    Returns
    -------
    str
        the username corresponding to the added public key.
    """
    username = get_username()
    # check if key in authorized keys
    ssh_dir = Path.home().joinpath(".ssh")
    ssh_dir.mkdir(exist_ok=True)
    authorized_keys = ssh_dir.joinpath("authorized_keys")
    if authorized_keys.exists() and public_key in authorized_keys.read_text(
        encoding="utf-8"
    ):
        return username
    with authorized_keys.open("a", encoding="utf-8") as f:
        f.write(f"{public_key}\n")
    return username


def private_key_to_str(key: PKey) -> str:
    """
    Convert a private key object to a string.

    Parameters
    ----------
    key : PKey
        the private key object to convert to a string/

    Returns
    -------
    str
        the string representation of the private key.
    """
    strbuffer = io.StringIO()
    key.write_private_key(strbuffer)
    private_key = strbuffer.getvalue()
    return private_key


def get_ray_tasks(ray_address: str, node_id: str, timeout: int) -> list[str]:
    """
    list of running ray tasks on the specified node ID.

    Parameters
    ----------
    ray_address : str
        the cluster ray address
    node_id : str
        the node ID to discover the running tasks
    timeout : int
        the timeout to apply when waiting for the ray cluster to respond.

    Returns
    -------
    list[str]
        a list of the ray task names running on the ``Node``.
    """
    return [
        task.name
        for task in list_tasks(
            address=ray_address,
            filters=[
                ("state", "=", "RUNNING"),
                ("node_id", "=", node_id),
            ],
            timeout=timeout,
        )
    ]


def utilization() -> dict[str, dict[int, int] | float | list[float] | int]:
    """
    the system utilization

    Returns
    -------
    dict[str, dict[int, int] | float | list[float] | int]
        a dictionary with keys:
            - gpu_free_mem : dict[int, int]
                corresponding to the free GPU memory on the system.
            - mem : float
                the percentage of available system memory
            - cpu_usage : list[float]
                the usage of each CPU core/thread on the system.
            - cpu_count : int
                the number of CPU cores on the system.
    """
    free_gpu = get_gpu_mem("free")
    mem_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent(interval=2, percpu=True)
    cpu_count = psutil.cpu_count()
    return {
        "gpu_free_mem": free_gpu,
        "mem": mem_usage,
        "cpu_usage": cpu_usage,
        "cpu_count": cpu_count,
    }


def run_lambda(
    fn: abc.Callable,
    node_ip: str | None = None,
    timeout: int | None = None,
    run_async: bool = False,
    cuda: bool | None = None,
    fn_kwargs: dict[str, ty.Any] | None = None,
    name: str | None = None,
    max_calls: int | None = 1,
    **options: float | str | dict,
) -> ty.Any | ray.ObjectRef:
    """
    run a Python lambda function on the Node and scheduled
    using ray. When running asynchronously a reference to the ray
    object is provided. Otherwise, it will return the function output.

    Parameters
    ----------
    fn : abc.Callable
        the function to run. Must be pickable.
    node_ip : str | None, optional
        the IP of the node to run the ray function to, by default None
    timeout : int | None, optional
        the timeout to apply when waiting for the results, by default None
    run_async : bool, optional
        whether to wait for the lambda output or schedule asynchronously, by default False
    cuda : bool | None, optional
        whether the function requires CUDA, by default False
    fn_kwargs : dict[str, ty.Any] | None, optional
        the keyword arguments to pass to the function, by default None
    name : str | None, optional
        the name of the scheduled function, by default None
    max_calls : int | None, optional
        used to de-allocate memory for remotes (but is not applicable for ray actors), by default 1
    **options : float | str | dict
        Additional kwarg options to supply as run-time configurations to the remote function.

    Returns
    -------
    ty.Any | ray.ObjectRef
        the result of the function or a reference to the ray object.
    """
    if fn_kwargs is None:
        fn_kwargs = {}
    options["num_cpus"] = 0.001
    if cuda is None:
        cuda = torch.cuda.is_available()
    if not cuda:
        options["num_gpus"] = 0
    else:
        options["num_gpus"] = 0.001
    if name is not None:
        options["name"] = name
    if max_calls is not None:
        options["max_calls"] = max_calls

    if node_ip is not None:
        options["resources"] = {f"node:{node_ip}": 0.001}

    remote_fn = ray.remote(**options)(fn).remote(**fn_kwargs)
    if not run_async:
        return ray.get(
            remote_fn,
            timeout=timeout,
        )
    return remote_fn


def run_ssh_cmd(
    node_ip: str,
    username: str,
    private_key: PKey,
    cmd: str,
    timeout: int,
    run_async: bool = False,
) -> str | None:
    """
    executes a bash command via ssh and optionally returns the
    `stdout` of the command. When `stderr` is present in the command
    output it will log it as an error.

    Parameters
    ----------
    node_ip : str
        the IP of where to execute the command.
    username : str
        the username of where to execute the command.
    private_key : PKey
        the private key of where to connect to execute the command
    cmd : str
        the command to execute
    timeout : int
        the timeout in establishing a connection with SSH server
    run_async : bool, optional
        whether to run the command asynchronously and not wait for stdout/stderr but
        instead return None, by default False

    Returns
    -------
    str | None
        a string of the `stdout` from the command or `None` when the command is run
        asynchronously
    """
    client = paramiko.SSHClient()
    policy = paramiko.AutoAddPolicy()
    client.set_missing_host_key_policy(policy)
    client.connect(
        node_ip,
        username=username,
        pkey=private_key,
        timeout=timeout,
        banner_timeout=timeout,
        auth_timeout=timeout,
        channel_timeout=timeout,
    )
    _, _stdout, _stderr = client.exec_command(cmd)
    if run_async:
        return None
    error_msg = _stderr.read().decode().strip("\n")
    if len(error_msg) > 0:
        logging.error("Error for %s: %s", node_ip, error_msg)

    return _stdout.read().decode().strip("\n")

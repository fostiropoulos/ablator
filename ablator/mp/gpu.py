"""
GPU-Manager ray object.
When launching many remotes at once, they all detect the same free GPU because of race conditions.
A solution is to keep a shared memory where they can lock a GPU until after the initialization phase
of the model.
"""
import logging
import time
from collections import OrderedDict
import traceback
from typing import Any

import numpy as np
import ray
from ray.actor import ActorHandle
from ablator.mp.heart import Heart
from ablator.mp.utils import Resource, get_ray_address, utilization
from ablator.utils._nvml import get_gpu_mem

MISSED_BEATS_LIMIT = 3


class GPUError(Exception):
    pass


class GPU:
    """
    A GPU class used to reserve resources for scheduling GPU-intensive tasks.

    Parameters
    ----------
    device_id : int
        The device ordinal number in the computer.
    free_mem : int
        The free memory available on the GPU.
    max_mem : int
        The maximum memory available on the GPU.
    lock_timeout : int
        The timeout after which the GPU will be considered
        unlocked if it is not manually unlocked when requesting
        it for use.

    Attributes
    ----------
    device_id : int
        the GPU id number
    free_mem : int
        the free memory for the given GPU in MB
    is_locked : bool
        whether the GPU is currently pending memory allotment.
    max_mem : int
        The maximum available free memory on the GPU in MiB.
    free_mem : int
        The free available memory on the GPU in MiB.
    locking_process_name : str | None
        Optionally, the process name that requested to lock the GPU.
        Can be ``None`` even when the GPU is locked.
    lock_timestamp : float | None
        The timestamp of when the GPU was locked. It is
        ``None`` when the GPU is unlocked.

    Raises
    ------
    ValueError
        When incorrect parameters are provided to the GPU initialization, such
        as `max_mem` and `free_mem`.
    """

    def __init__(
        self, device_id: int, free_mem: int, max_mem: int, lock_timeout: int
    ) -> None:
        self.device_id: int
        # immutable attributes
        super().__setattr__("device_id", device_id)
        self.max_mem: int
        super().__setattr__("max_mem", max_mem)
        self.free_mem: int = free_mem
        self.lock_timeout: int = lock_timeout
        self.locking_process_name: str | None = None
        self.lock_timestamp: float | None = None
        if self.free_mem > self.max_mem:
            raise ValueError("GPU `max_mem` must be > `free_mem`")

    def __getattribute__(self, __name: str) -> Any:
        super().__getattribute__("_refresh")()
        return super().__getattribute__(__name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in {"device_id", "max_mem"}:
            raise ValueError(f"Can not change property `{__name}`.")
        if __name == "free_mem" and __value > super().__getattribute__("max_mem"):
            raise ValueError(
                "Can not specify larger `free_mem` than total GPU memory "
                f"`max_mem` ({super().__getattribute__('max_mem')})."
            )
        super().__getattribute__("_refresh")()
        super().__setattr__(__name, __value)

    @property
    def is_locked(self):
        self._refresh()
        return (
            super().__getattribute__("locking_process_name") is not None
            or super().__getattribute__("lock_timestamp") is not None
        )

    def lock(self, process_name):
        super().__setattr__("locking_process_name", process_name)
        super().__setattr__("lock_timestamp", time.time())

    def unlock(self):
        super().__setattr__("locking_process_name", None)
        super().__setattr__("lock_timestamp", None)

    # pylint: disable=broad-exception-caught
    def _refresh(self):
        try:
            if super().__getattribute__(
                "lock_timestamp"
            ) is not None and time.time() - super().__getattribute__(
                "lock_timestamp"
            ) > super().__getattribute__(
                "lock_timeout"
            ):
                super().__getattribute__("unlock")()
        except Exception:
            ...

    def __repr__(self) -> str:
        properties = ["device_id", "free_mem", "lock_timeout"]
        _init_string = ", ".join([f"{k}={getattr(self, k)}" for k in properties])
        return f"{type(self).__name__}({_init_string})"


def wait_get_gpu(
    manager: "ResourceManager",
    expected_util_mb: int | None = None,
    process_name: str | None = None,
    max_timeouts: int = 60,
) -> GPU:
    """
    a wrapper around ``ResourceManager.request_gpu`` that retries to get an available GPU based
    on the expected utilization and a timeout. It unlocks the given GPU when the `process_name`
    allocates memory to the allocated GPU or when the `lock_timeout` of ResourceManager is reached.

    Parameters
    ----------
    manager : ResourceManager
        the manager to request resources from.
    expected_util_mb : int | None
        the expected memory utilization in MB, by default ``None``
    process_name : str | None
        the name of the process to use to identify memory utilization, by default ``None``
    max_timeouts : int
        the seconds of timeouts after which to throw an error, by default 60

    Returns
    -------
    GPU
        the least used GPU number to be used with `.to(f"cuda:{i}")`

    Raises
    ------
    GPUError
        when there are no available GPUs within the timeout limit
    RuntimeError
        when the manager is not a ray actor object.
    """
    if not isinstance(manager, ActorHandle):
        raise RuntimeError("`wait_get_gpu` must be called with an Actor object.")

    least_used_gpu: GPU
    gpus: OrderedDict[int, GPU] = ray.get(manager.gpu_dict.remote())
    matching_gpu = expected_util_mb is None or any(
        gpu.max_mem >= expected_util_mb for gpu in gpus.values()
    )
    if len(gpus) == 0 or not matching_gpu:
        raise GPUError("No available GPU.")
    for _ in range(max_timeouts):
        if (
            least_used_gpu := ray.get(
                manager.request_gpu.remote(expected_util_mb, process_name)  # type: ignore[attr-defined]
            )
        ) is None:
            time.sleep(1)
            continue
        return least_used_gpu
    raise GPUError("No available GPU.")


def unlock_gpu(resource: "ResourceManager", gpu: GPU):
    """
    a wrapper around ``ResourceManager.unlock_gpu`` that unlocks the previously requested
    GPU id.


    Parameters
    ----------
    resource : ResourceManager
        the manager to unlock the requested .
    gpu : GPU
        the id of the GPU that will unlock.

    """
    ray.get(resource.unlock.remote(gpu))  # type: ignore[attr-defined]


class ResourceManager(Heart):
    """
    Resource class helps manage memory, CPU and GPU resources on a given machine.
    It maintains a state of the current GPUs available on the node,
    with their available memory. The class is meant to be used
    for synchronized request and management of resources between ray.remotes

    Parameters
    ----------
    node_ip : str
        the node IP where to monitor the resources
    resource_lock_timeout : int
        a timeout by which to unlock reserved resources
        if not manually unlocked.
    cuda_visible_devices : list[int] | None, optional
        the visible cuda devices to over-write the
        ones found by the system, by default None
    update_interval : int, optional
        the interval to update the resources by, by default 10
    ray_address : str | None, optional
        the address of the ray cluster where the node
        is connected, by default None

    Attributes
    ----------
    node_ip : str
        the node ip address
    ray_address : str
        the address of the ray cluster where the node
        is connected
    gpus : OrderedDict[int, GPU]
        the GPUs available on the Node ordered via their
        unique device number.
    mem : int
        the utilization of memory on the Node in percentage.
    cpu_usage : list[float]
        a list of CPU usage on the node, one for each thread
    cpu_count : int
        the number of CPUs available on the node
    is_active : bool
        whether the Node resources are active or outdated
    gpu_free_mem : list[int]
        the free GPU memory in MiB on the Node is ordered by the device number
        it corresponds to.
    cpu_mean_util : float
        the mean CPU utilization on the Node.
    """

    def __init__(
        self,
        node_ip: str,
        resource_lock_timeout: int,
        cuda_visible_devices: list[int] | None = None,
        update_interval: int = 10,
        ray_address: str | None = None,
    ) -> None:
        self._timeout = resource_lock_timeout
        self.node_ip = node_ip
        if ray_address is None:
            self.ray_address = get_ray_address()
        else:
            self.ray_address = ray_address

        gpus = {}

        max_mem = get_gpu_mem("total")
        for device_id, free_mem in get_gpu_mem("free").items():
            gpus[device_id] = GPU(
                device_id=device_id,
                free_mem=free_mem,
                max_mem=max_mem[device_id],
                lock_timeout=resource_lock_timeout,
            )
        if cuda_visible_devices is not None:
            gpus = {i: gpus[i] for i in cuda_visible_devices}
        self.gpus: OrderedDict[int, GPU] = OrderedDict(
            (k, v) for k, v in sorted(gpus.items(), key=lambda x: x[0])
        )

        self.mem: int = 0
        self.cpu_usage: list[float] = []
        self.cpu_count: int = 0
        self.is_active = True
        super().__init__(
            missed_heart_beats=MISSED_BEATS_LIMIT, heartbeat_interval=update_interval
        )

    @property
    def gpu_free_mem(self) -> list[int]:
        return [g.free_mem for g in self.gpus.values()]

    @property
    def cpu_mean_util(self) -> float:
        return float(np.mean(self.cpu_usage))

    def resources(self) -> Resource:
        return Resource(
            mem=self.mem,
            cpu_usage=self.cpu_usage,
            cpu_count=self.cpu_count,
            gpu_free_mem=self.gpu_free_mem,
            is_active=self.is_active,
        )

    # pylint: disable=broad-exception-caught
    def heartbeat(self, timeout: int | None = None):
        try:
            if timeout is None:
                timeout = self._timeout
            timeout = max(timeout, 5)
            util_dict = utilization()
            assert set(util_dict.keys()) == {
                "gpu_free_mem",
                "mem",
                "cpu_usage",
                "cpu_count",
            }
            for gpu_id, gpu_mem in util_dict["gpu_free_mem"].items():  # type: ignore[union-attr]
                if gpu_id in self.gpus:
                    self.gpus[gpu_id].free_mem = gpu_mem
            del util_dict["gpu_free_mem"]
            for k, v in util_dict.items():
                assert hasattr(self, k)
                setattr(self, k, v)

            self.is_active = True
        except Exception:
            self.mem = 0
            self.cpu_usage = []
            self.cpu_count = 0
            self.is_active = False
            logging.error(
                "Could not find node utilization %s. Ignoring. %s",
                self.node_ip,
                traceback.format_exc(),
            )

    def request_gpu(
        self, expected_util_mb: int | None = None, process_name: str | None = None
    ) -> GPU | None:
        """
        request an available GPU based on the expected utilization of the cuda
        process. We associate the `process_name` with the requested GPU, but do
        not use in any way.

        Parameters
        ----------
        expected_util_mb : int | None
            the expected utilization of the cuda process.
        process_name : str | None
            the id of the process requesting the GPU resources, by default ``None``

        Returns
        -------
        GPU | None
            the cuda device or None if there is no available device.
        """
        sorted_gpus = sorted(
            [gpu for gpu in self.gpus.values() if not gpu.is_locked],
            key=lambda x: x.free_mem,
        )
        if len(sorted_gpus) == 0 or (
            expected_util_mb is not None and sorted_gpus[-1].free_mem < expected_util_mb
        ):
            return None
        least_used_gpu = sorted_gpus[-1]
        self.gpus[least_used_gpu.device_id].lock(process_name)
        return self.gpus[least_used_gpu.device_id]

    def unlock(self, gpu: GPU):
        """
        unlocks the GPU given its device ID. The method is expected to be called once the GPU
        utilization is stabilized for the process that requested a GPU.

        Parameters
        ----------
        gpu : GPU
            the GPU to unlock.
        """
        self.gpus[gpu.device_id].unlock()

    def gpu_dict(self) -> OrderedDict[int, GPU]:
        """
        A function that returns `gpus` property and is used
        as a proxy for a remote function.

        Returns
        -------
        OrderedDict[int, GPU]
            the GPUs available on the Node.
        """
        return self.gpus

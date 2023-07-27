"""
GPU-Manager ray object.
When launching many remotes at once, they all detect the same free GPU because of race conditions.
A solution is to keep a shared memory where they can lock a GPU until after the initialization phase
of the model.
"""
from dataclasses import dataclass
import time
import ray
from ablator.utils.base import get_gpu_mem


class GPUError(Exception):
    pass


@dataclass
class GPU:
    """
    A GPU class used for internal purposes.


    Attributes
    ----------
    num : int
        the GPU id number
    free_mem : int | None
        the free memory for the given GPU in MB
    is_locked : bool
        whether the GPU is currently pending memory allotment.
    """

    num: int
    free_mem: int
    lock_timeout: int
    _locking_process: str | None = None
    _time_lock: float | None = None

    @property
    def is_locked(self):
        return self._locking_process is not None or self._time_lock is not None

    def lock(self, process_name):
        self._locking_process = process_name
        self._time_lock = time.time()

    def unlock(self):
        self._time_lock = None
        self._locking_process = None

    def update(self, free_mem: dict[int, int]):
        self.free_mem = free_mem[self.num]
        if not self.is_locked:
            return
        if (
            self._time_lock is not None
            and time.time() - self._time_lock > self.lock_timeout
        ):
            self.unlock()


def wait_get_gpu(
    manager: "GPUManager",
    expected_util_mb: int | None = None,
    process_name: str | None = None,
    max_timeouts: int = 60,
) -> int:
    """
    a wrapper around ``GPUManager.request_gpu`` that retries to get an available GPU based
    on the expected utilization and a timeout. It unlocks the given GPU when the `process_name`
    allocates memory to the allocated GPU or when the `lock_timeout` of GPUManager is reached.

    Parameters
    ----------
    manager : GPUManager
        the manager to request resources from.
    expected_util_mb : int | None
        the expected memory utilization in MB, by default ``None``
    process_name : str | None
        the name of the process to use to identify memory utilization, by default ``None``
    max_timeouts : int
        the seconds of timeouts after which to throw an error, by default 60
    Returns
    -------
    int
        the least used GPU number to be used with `.to(f"cuda:{i}")`

    Raises
    ------
    GPUError
        when there are no available GPUs within the timeout limit
    """
    timeouts = 0
    while timeouts < max_timeouts:
        if (
            least_used_gpu := ray.get(
                manager.request_gpu.remote(expected_util_mb, process_name)  # type: ignore
            )
        ) is None:
            time.sleep(1)
            timeouts += 1
            continue
        return least_used_gpu  # type: ignore
    raise GPUError("No available GPU.")


def unlock_gpu(manager: "GPUManager", gpu: int):
    """
    a wrapper around ``GPUManager.unlock_gpu`` that unlocks the previously requested
    GPU id.


    Parameters
    ----------
    manager : GPUManager
        the manager to unlock the requested .
    gpu : int
        the id of the GPU that will unlock.

    """
    ray.get(manager.unlock.remote(gpu))  # type: ignore


@ray.remote
class GPUManager:
    """
    GPUManager class helps manage GPU resources on a given machine.
    It maintains a state of the current GPUs available on the device,
    with their available memory. The class is meant to be be used
    for synchronized request and management of resources between ray.remotes
    """

    def __init__(
        self, lock_timeout: int = 60, cuda_visible_devices: list[int] | None = None
    ):
        gpus = {}
        for device_id, free_mem in get_gpu_mem("free").items():
            gpus[device_id] = GPU(
                num=device_id, free_mem=free_mem, lock_timeout=lock_timeout
            )
        if cuda_visible_devices is not None:
            gpus = {i: gpus[i] for i in cuda_visible_devices}
        self._gpu_dict: dict[int, GPU] = gpus

    def _gpus(self) -> dict[int, GPU]:
        return self._gpu_dict

    def _update_gpus(self):
        free_mem = get_gpu_mem("free")
        for gpu in self._gpu_dict.values():
            gpu.update(free_mem)

    def _sorted_unlocked_gpus(self) -> list[GPU]:
        self._update_gpus()
        return sorted(
            [gpu for gpu in self._gpu_dict.values() if not gpu.is_locked],
            key=lambda x: x.free_mem,
        )

    def request_gpu(
        self, expected_util_mb: int | None = None, process_name: str | None = None
    ) -> int | None:
        """
        request an available GPU based on the expected utilization of the cuda
        process. We associate the `process_name` with the requested GPU, but do
        not use in any way.

        Parameters
        ----------
        expected_util_mb : int
            the expected utilization of the cuda process.
        process_name : str
            the name of the process requesting the GPU resources, by default ``None``

        Returns
        -------
        int | None
            the cuda device or None if there is no available device.
        """
        gpus = self._sorted_unlocked_gpus()
        if len(gpus) == 0 or (
            expected_util_mb is not None and gpus[-1].free_mem < expected_util_mb
        ):
            return None
        least_used_gpu = gpus[-1].num
        gpus[-1].lock(process_name)
        return least_used_gpu

    def unlock(self, gpu_id: int):
        """
        unlocks the GPU given its num ID. the method is expected to be called once the GPU
        utilization is stablized for the process that requested a GPU.

        Parameters
        ----------
        gpu_id : int
            the GPU id to unlock.
        """
        self._gpu_dict[gpu_id].unlock()

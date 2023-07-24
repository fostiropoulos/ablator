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


@dataclass
class GPU:
    """
    A GPU class


    Attributes
    ----------
    num : int
        the GPU id number
    free_mem : int
        the free memory for the given GPU in MB
    is_locked : bool
        whether the GPU is currently pending memory allotment.
    """

    num: int
    free_mem: int
    is_locked: bool = False


def _get_gpus() -> dict[int, GPU]:
    return {
        i: GPU(num=i, free_mem=free_mem)
        for i, free_mem in enumerate(list(get_gpu_mem("free").values()))
    }


def wait_get_gpu(
    manager: "GPUManager", expected_util_mb: int, max_timeouts: int = 60
) -> int:
    """
    a wrapper around ``GPUManager.request_gpu`` that retries to get an available GPU based
    on the expected utilization and a timeout.

    Parameters
    ----------
    manager : GPUManager
        the manager to request resources from.
    expected_util_mb : int
        the expected memory utilization in MB
    max_timeouts : int, optional
        the seconds of timeouts after which to throw an error, by default 60

    Returns
    -------
    int
        the least used GPU number to be used with `.to(f"cuda:{i}")`

    Raises
    ------
    RuntimeError
        when there are no available GPUs within the timeout limit
    """
    timeouts = 0
    while timeouts < max_timeouts:
        if (
            least_used_gpu := ray.get(
                manager.request_gpu.remote(expected_util_mb)  # type: ignore
            )
        ) is None:
            time.sleep(1)
            timeouts += 1
            continue
        return least_used_gpu  # type: ignore
    raise RuntimeError("No available GPU.")


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
    ray.get(manager.unlock_gpu.remote(gpu))  # type: ignore


@ray.remote
class GPUManager:
    """
    GPUManager class helps manage GPU resources on a given machine.
    It maintains a state of the current GPUs available on the device,
    with their available memory. The class is meant to be be used
    for synchronized request and management of resources between ray.remotes
    """

    def __init__(self, cuda_visible_devices: list[int] | None = None):
        gpus = _get_gpus()
        if cuda_visible_devices is not None:
            gpus = {i: gpus[i] for i in cuda_visible_devices}
        self._gpu_dict: dict[int, GPU] = gpus

    def _gpus(self) -> dict[int, GPU]:
        return self._gpu_dict

    def _update_gpus(self):
        updated_gpus = _get_gpus()
        for gpu_idx in self._gpu_dict:
            self._gpu_dict[gpu_idx].free_mem = updated_gpus[gpu_idx].free_mem

    def _sorted_unlocked_gpus(self) -> list[GPU]:
        self._update_gpus()
        return sorted(
            [gpu for gpu in self._gpu_dict.values() if not gpu.is_locked],
            key=lambda x: x.free_mem
        )

    def request_gpu(self, expected_util_mb: int) -> int | None:
        """
        request an available GPU based on the expected utilization of the cuda
        process.

        Parameters
        ----------
        expected_util_mb : int
            the expected utilization of the cuda process.

        Returns
        -------
        int | None
            the cuda device or None if there is no available device.
        """
        gpus = self._sorted_unlocked_gpus()
        if len(gpus) == 0 or gpus[-1].free_mem < expected_util_mb:
            return None
        least_used_gpu = gpus[-1].num
        self._gpu_dict[least_used_gpu].is_locked = True
        return least_used_gpu

    def unlock_gpu(self, gpu: int):
        """
        unlocks the GPU given its num ID. the method is expected to be called once the GPU
        utilization is stablized for the process that requested a GPU.

        Parameters
        ----------
        gpu : int
            the GPU id to unlock.
        """
        self._gpu_dict[gpu].is_locked = False

import contextlib
import logging
import os
import typing as ty
from collections import namedtuple
from platform import uname

import psutil
import torch

CUDA_PROCESS = namedtuple("CUDA_PROCESS", ["process_name", "pid", "memory"])


try:
    # pylint: disable=unspecified-encoding
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        from pynvml import nvml

        nvml.nvmlInit()
        device_count = nvml.nvmlDeviceGetCount()
        HANDLES = []
        for _i in range(0, device_count):
            _handle = nvml.nvmlDeviceGetHandleByIndex(_i)
            HANDLES.append(_handle)
# pylint: disable=broad-exception-caught
except Exception:
    nvml = None


def _is_smi_bug() -> bool:
    # This is not a reported bug, but apparently there is a padding
    # to each process, causes a strange error.
    # The docs do not report any padding:
    # https://docs.nvidia.com/deploy/nvml-api/structnvmlProcessInfo__t.html#structnvmlProcessInfo__t
    # The byte-order also differs (e.g. pid, usedGpuMemory, gpuInstanceId, computeInstanceId)
    # Details here: https://github.com/gpuopenanalytics/pynvml/pull/48

    if not torch.cuda.is_available() or nvml is None:
        return False
    driver_version = nvml.nvmlSystemGetDriverVersion()
    if driver_version.startswith("535"):
        raise RuntimeError(
            f"Please change your nvidia-driver version `{driver_version}` as there is"
            " critical bug for version 535:"
            " https://github.com/gpuopenanalytics/pynvml/pull/48"
        )
    return False


def _handleError(err: "nvml.NVMLError") -> str:
    if err.value == nvml.NVML_ERROR_NOT_SUPPORTED:
        return "N/A"
    return str(err)


# flake8: noqa: E722
def getProcessName(pid):
    try:
        if os.name == "nt" or "microsoft-standard" in uname().release:
            return f"process_pid:{str(pid)}"
        process = psutil.Process(pid)
        return process.name()
    except:  # pylint: disable=bare-except
        return f"process_pid:{str(pid)}"


def _get_processes(handle) -> list[dict[str, str | int]]:
    processes = []
    try:
        procs = nvml.nvmlDeviceGetComputeRunningProcesses(handle)

        for p in procs:
            name = getProcessName(p.pid)
            processInfo = {}
            processInfo["pid"] = p.pid
            processInfo["process_name"] = name

            if p.usedGpuMemory is None:
                mem = 0
            else:
                mem = int(p.usedGpuMemory / 1024 / 1024)
            processInfo["used_memory"] = mem
            processInfo["unit"] = "MiB"
            processes.append(processInfo)

    except nvml.NVMLError as err:
        logging.error("Unable to read NVML processes error %s", str(err))

    return processes


def _get_gpu_info() -> list[dict[str, ty.Any]]:
    if nvml is None:
        return []

    assert not _is_smi_bug()
    deviceCount = nvml.nvmlDeviceGetCount()
    dictResult = []
    for i in range(0, deviceCount):
        gpuResults: dict[str, ty.Any] = {}
        fbMemoryUsage = {}
        handle = HANDLES[i]
        try:
            memInfo = nvml.nvmlDeviceGetMemoryInfo(handle)
            mem_total = memInfo.total / 1024 / 1024
            mem_used = memInfo.used / 1024 / 1024
            mem_free = memInfo.total / 1024 / 1024 - memInfo.used / 1024 / 1024
        except nvml.NVMLError as err:
            error = _handleError(err)
            mem_total = error
            mem_used = error
            mem_free = error

        fbMemoryUsage["total"] = mem_total
        fbMemoryUsage["used"] = mem_used
        fbMemoryUsage["free"] = mem_free
        fbMemoryUsage["unit"] = "MiB"
        gpuResults["fb_memory_usage"] = fbMemoryUsage
        gpuResults["minor_number"] = i
        gpuResults["processes"] = _get_processes(handle)
        dictResult.append(gpuResults)
    return dictResult


def get_cuda_processes() -> dict[int, list[CUDA_PROCESS]]:
    """
    Finds the currently running cuda processes on the system. Each process is a
    ``CUDA_PROCESS`` object that contains information on the process name, `pid` and
    the memory utilization.

    Returns
    -------
    dict[int, list[CUDA_PROCESS]]
        The key of each dictionary is the device-id, corresponding to a list of running CUDA processes.
    """
    gpus = _get_gpu_info()
    cuda_processes: dict[int, list[CUDA_PROCESS]] = {}
    for gpu in gpus:
        device_id = int(gpu["minor_number"])
        cuda_processes[device_id] = [
            CUDA_PROCESS(p["process_name"], p["pid"], p["used_memory"])
            for p in gpu["processes"]
        ]
    return cuda_processes


def get_gpu_mem(
    mem_type: ty.Literal["used", "total", "free"] = "total"
) -> dict[int, int]:
    """
    Get the memory information of all available GPUs.

    Parameters
    ----------
    mem_type : ty.Literal["used", "total", "free"], optional
        The type of memory information to retrieve, by default "total".

    Returns
    -------
    dict[int, int]
        A list of memory values for each GPU, depending on the specified memory type.
    """
    memory: dict[int, int] = {}
    gpus = _get_gpu_info()
    for gpu in gpus:
        device_id = int(gpu["minor_number"])
        memory[device_id] = int(gpu["fb_memory_usage"][mem_type])
    return memory

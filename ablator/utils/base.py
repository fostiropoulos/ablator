import contextlib
import os
import random
import sys
import typing as ty
from collections import namedtuple
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
import numpy as np
import torch
from ablator.utils._nvml import patch_smi

try:
    # pylint: disable=unspecified-encoding
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        from pynvml.smi import nvidia_smi as smi
        from pynvml import nvml

        _instance = smi.getInstance()
        _instance.DeviceQuery()
        # TODO: waiting for fix: https://github.com/pytorch/pytorch/issues/86493
# pylint: disable=broad-exception-caught
except Exception:
    smi = None

CUDA_PROCESS = namedtuple("CUDA_PROCESS", ["process_name", "pid", "memory"])


class Dummy:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self

    def __getitem__(self, *args, **kwargs):
        return self


def iter_to_numpy(iterable):
    """
    Convert elements of the input iterable to NumPy arrays if they are torch.Tensor objects.

    Parameters
    ----------
    iterable : Iterable
        The input iterable.

    Returns
    -------
    any
        The iterable with torch.Tensor elements replaced with their NumPy array equivalents.
    """
    return apply_lambda_to_iter(
        iterable,
        lambda v: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v,
    )


def iter_to_device(
    data_dict, device
) -> ty.Union[Sequence[torch.Tensor], dict[str, torch.Tensor]]:
    """
    Moving torch.Tensor elements to the specified device.

    Parameters
    ----------
    data_dict : dict or list
        The input dictionary or list containing torch.Tensor elements.
    device : torch.device | str
        The target device for the tensors.

    Returns
    -------
    ty.Union[Sequence[torch.Tensor], dict[str, torch.Tensor]]
        The input data with tensors moved to the target device.
    """
    return apply_lambda_to_iter(
        data_dict, lambda v: v.to(device) if isinstance(v, torch.Tensor) else v
    )


def apply_lambda_to_iter(iterable, fn: Callable):
    """
    Applies a given function ``fn`` to each element of an iterable data structure.

    This function recursively applies ``fn`` to elements within nested dictionaries or lists.
    It can be used for converting torch.Tensor elements to NumPy arrays or moving tensors
    to a specified device.

    Parameters
    ----------
    iterable : Iterable
        The input iterable.
    fn : Callable
        The function to apply to each element.

    Returns
    -------
    any
        The resulting data structure after applying ``fn`` to each element of the input ``iterable``.
        The type of the returned object matches the type of the input ``iterable``.
    """
    if isinstance(iterable, dict):
        return {
            k: apply_lambda_to_iter(v, fn) if isinstance(v, (Iterable)) else fn(v)
            for k, v in iterable.items()
        }
    if isinstance(iterable, list):
        return [apply_lambda_to_iter(v, fn) for v in iterable]

    return fn(iterable)


def set_seed(seed: int):
    """
    Set the random seed.

    Parameters
    ----------
    seed : int
        The random seed to set.

    Returns
    -------
    int
        The set random seed.
    """
    assert seed is not None, "Must provide a seed"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def get_lr(optimizer):
    """
    Get the learning rate from an optimizer.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer or dict
        The optimizer.

    Returns
    -------
    float
        The learning rate.
    """
    if isinstance(optimizer, dict):
        param_groups = optimizer["param_groups"]
    else:
        param_groups = optimizer.param_groups

    return param_groups[0]["lr"]


def debugger_is_active() -> bool:
    """
    Check if the debugger is currently active.

    Returns
    -------
    bool
        True if the debugger is active, False otherwise.

    Notes
    -----
    Return if the debugger is currently active
    """
    gettrace = getattr(sys, "gettrace", lambda: None)
    return gettrace() is not None


def get_latest_chkpts(checkpoint_dir: Path) -> list[Path]:
    """
    Get a list of all checkpoint files in a directory, sorted from the latest to the earliest.

    Parameters
    ----------
    checkpoint_dir : Path
        The directory containing checkpoint files.

    Returns
    -------
    list[Path]
        A list of the checkpoint files sorted by filename.
    """
    return sorted(list(checkpoint_dir.glob("*.pt")))[::-1]


def parse_device(device: ty.Union[str, list[str]]):
    """
    Parse a device string, an integer, or a list of device strings or integers.

    Parameters
    ----------
    device : ty.Union[str, list[str], int]
        The target device for the tensors.

    Returns
    -------
    any
        The parsed device string, integer, or list of device strings or integers.

    Raises
    ------
    ValueError
        If the device string is not one of {'cpu', 'cuda'} or doesn't start with 'cuda:'.
    AssertionError
        If cuda is not found on system or gpu number of device is not available.

    Examples
    --------
    >>> parse_device("cpu")
    'cpu'
    >>> parse_device("cuda")
    'cuda'
    >>> parse_device("cuda:0")
    'cuda:0'
    >>> parse_device(["cpu", "cuda"])
    ['cpu', 'cuda']
    >>> parse_device(["cpu", "cuda:0", "cuda:1", "cuda:2"])
    ['cpu', 'cuda:0', 'cuda:1', 'cuda:2']
    """
    if isinstance(device, str):
        if device == "cpu":
            return device
        if device == "cuda" or (device.startswith("cuda:") and device[5:].isdigit()):
            assert (
                torch.cuda.is_available()
            ), "Could not find a torch.cuda installation on your system."
            if device.startswith("cuda:"):
                gpu_number = int(device[5:])
                assert (
                    gpu_number < torch.cuda.device_count()
                ), f"gpu {device} does not exist on this machine"
            return device
        raise ValueError
    if isinstance(device, int):
        return device
    if isinstance(device, Iterable):
        return [parse_device(_device) for _device in device]

    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_gpu_info() -> list[dict[str, ty.Any]]:
    if smi is not None:
        try:
            instance = smi.getInstance()
            device = instance.DeviceQuery()
        # pylint: disable=broad-exception-caught
        except Exception:
            return []
    else:
        return []
    if not getattr(smi, "_is_id", False):
        patch_smi(smi, nvml)
    if "gpu" not in device:
        return []
    return sorted(device["gpu"], key=lambda x: x["minor_number"])


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
        if gpu["processes"] is None:
            cuda_processes[device_id] = []
            continue
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


def _num_e_format(value: int | np.integer | float | np.floating, width: int) -> str:
    # minimum width of 8 "x.xxe+nn" 6 fixed 1.xxe+00 and 2 available
    diff = 6
    if value < 0:
        diff += 1
    return f"{value:.{width-diff}e}"


def _num_format_int(value: int | np.integer, width: int) -> str:
    str_value = str(value)
    if len(str_value) > width:
        return _num_e_format(value, width)
    return f"{value:0{width}d}"


def _num_format_float(value: float | np.floating, width: int) -> str:
    str_value = str(value)
    if "e" in str_value:
        return _num_e_format(value, width)
    int_part, *float_part = str_value.split(".")
    int_len = len(int_part)
    if len(float_part):
        _float_part = float_part[0]
        float_len = len(_float_part)
        leading_zeros = float_len - len(_float_part.lstrip("0"))
    else:
        _float_part = None
        float_len = 0
        leading_zeros = 0
    if int_len > 0 and int_part != "0":
        # xxx.xxxx format
        if float_len + int_len >= width:
            return _num_e_format(value, width)
        return f"{value:.{width-int_len - 1}f}"
    if int_part == "0" and leading_zeros == float_len:
        return f"{value:.{width-2}f}"
    if len(str_value) < width:
        return f"{value:.{width-2}f}"
    return _num_e_format(value, width)


def is_oom_exception(err: RuntimeError) -> bool:
    """
    is_oom_exception checks whether the exception is caused by CUDA out of memory errors.

    Parameters
    ----------
    err : RuntimeError
        the exception to parse

    Returns
    -------
    bool
        whether the exception indicates out of memory error.
    """
    return any(
        x in str(err)
        for x in (
            "CUDA out of memory",
            "CUBLAS_STATUS_ALLOC_FAILED",
            "CUDA error: out of memory",
        )
    )


def num_format(
    value: str | int | float | np.integer | np.floating, width: int = 8
) -> str:
    """
    Format number to be no larger than `width` by converting to scientific
    notation when the `value` exceeds width either by informative decimal places
    or size.

    Parameters
    ----------
    value : int | float
        the value to format
    width : int, optional
        the width of the decimal places, by default 5

    Returns
    -------
    str
        The formatted string representation of the `value`
    """
    assert width >= 8
    if isinstance(value, (int, np.integer)):
        return _num_format_int(value, width)
    if isinstance(value, (float, np.floating)):
        return _num_format_float(value, width)
    return value

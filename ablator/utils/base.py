import contextlib
import os
import random
import sys
import typing as ty
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    # pylint: disable=unspecified-encoding
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        from pynvml.smi import nvidia_smi as smi
        _instance = smi.getInstance()
        _instance.DeviceQuery()
        # TODO: waiting for fix: https://github.com/pytorch/pytorch/issues/86493
# pylint: disable=broad-exception-caught
except Exception:
    smi = None


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


# TODO not a good initialization. Performs poorly for some networks.
def init_weights(module: nn.Module):
    """
    Initialize the weights of a module.

    Parameters
    ----------
    module : nn.Module
        The input module to initialize.

    Notes
    -----
    - If the module is a Linear layer, initialize weight values from a normal distribution N(mu=0, std=1.0).
    If biases are available, initialize them to zeros.

    - If the module is an Embedding layer, initialize embeddings with values from N(mu=0, std=1.0).
    If padding is enabled, set the padding embedding to a zero vector.

    - If the module is a LayerNorm layer, set all biases to zeros and all weights to 1.
    """
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def get_gpu_mem(
    mem_type: ty.Literal["used", "total", "free"] = "total"
) -> dict[str, int]:
    """
    Get the memory information of all available GPUs.

    Parameters
    ----------
    mem_type : ty.Literal["used", "total", "free"], optional
        The type of memory information to retrieve, by default "total".

    Returns
    -------
    list[int]
        A list of memory values for each GPU, depending on the specified memory type.
    """
    memory: dict[str, int] = {}
    if smi is not None:
        try:
            instance = smi.getInstance()
            device = instance.DeviceQuery()
        # pylint: disable=broad-exception-caught
        except Exception:
            pass
    else:
        return memory
    if "gpu" not in device:
        return memory
    for gpu in device["gpu"]:
        device_id = f"{gpu['product_name']}_{gpu['id']}"
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

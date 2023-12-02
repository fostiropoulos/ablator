import logging
import random
import sys
import time
import typing as ty
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
import numpy as np
import ray
import torch


class Dummy:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self

    def __getitem__(self, *args, **kwargs):
        return self


@ray.remote
class _Lock:
    def __init__(self) -> None:
        self.lock: bool = False

    def acquire(self):
        if not self.lock:
            self.lock = True
            return True
        return False

    def release(self):
        if not self.lock:
            raise ValueError("lock released too many times")
        self.lock = False


class Lock:
    def __init__(self, timeout: float | None = None) -> None:
        # pylint: disable=no-member
        self.lock = _Lock.remote()  # type: ignore[attr-defined]

        self.timeout = timeout

    def __enter__(self):
        self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def acquire(self):
        current_time = time.time()
        n = 0
        while True:
            if ray.get(self.lock.acquire.remote()):
                break
            if self.timeout is not None and time.time() - current_time > self.timeout:
                raise TimeoutError(
                    f"Could not obtain lock within {self.timeout:.2f} seconds"
                )
            n += 1
            if n % 10000 == 0:
                logging.warning(
                    (
                        "Lock(%s) takes an excessive time and it could be caused by a"
                        " deadlock."
                    ),
                    id(self),
                )

            time.sleep(0.1)

    def release(self):
        ray.get(self.lock.release.remote())


def iter_to_numpy(iterable: Iterable) -> ty.Any:
    """
    Convert elements of the input iterable to NumPy arrays if they are torch.Tensor objects.

    Parameters
    ----------
    iterable : Iterable
        The input iterable.

    Returns
    -------
    ty.Any
        The iterable with torch.Tensor elements replaced with their NumPy array equivalents.
    """
    return apply_lambda_to_iter(
        iterable,
        lambda v: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v,
    )


def iter_to_device(
    data_dict: Iterable, device: str
) -> ty.Union[Sequence[torch.Tensor], dict[str, torch.Tensor]]:
    """
    Moving torch.Tensor elements to the specified device.

    Parameters
    ----------
    data_dict : Iterable
        The input dictionary or list containing torch.Tensor elements.
    device : str
        The target device for the tensors.

    Returns
    -------
    ty.Union[Sequence[torch.Tensor], dict[str, torch.Tensor]]
        The input data with tensors moved to the target device.
    """
    return apply_lambda_to_iter(
        data_dict, lambda v: v.to(device) if isinstance(v, torch.Tensor) else v
    )


def apply_lambda_to_iter(iterable: Iterable, fn: Callable) -> ty.Any:
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
    ty.Any
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


def set_seed(seed: int) -> int:
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


def get_lr(optimizer: torch.optim.Optimizer | dict) -> float:
    """
    Get the learning rate from an optimizer.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer | dict
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


def parse_device(device: ty.Union[str, int]) -> str:
    """
    Parse a device string, an integer, or a list of device strings or integers.

    Parameters
    ----------
    device : ty.Union[str, int]
        The target device for the tensors.

    Returns
    -------
    str
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
        return parse_device(f"cuda:{device}")

    return "cuda" if torch.cuda.is_available() else "cpu"


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
    value : str | int | float | np.integer | np.floating
        the value to format
    width : int
        the width of the decimal places, by default ``8``.

    Returns
    -------
    str
        The formatted string representation of the `value`

    Examples
    --------
    >>> num_format(123456, width=8)
    123456
    >>> num_format(123456789, width=8)
    1.23e+08
    >>> num_format(1234.5678, width=8)
    1.23e+03
    >>> num_format(0.000012345, width=8)
    1.23e-05
    >>> num_format(np.float64(12345678.12345678), width=8)
    1.23e+07
    """
    assert width >= 8
    if isinstance(value, (int, np.integer)):
        return _num_format_int(value, width)
    if isinstance(value, (float, np.floating)):
        return _num_format_float(value, width)
    return value

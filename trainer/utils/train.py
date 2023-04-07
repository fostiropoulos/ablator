import random
import sys
from typing import Callable, Dict, Iterable, List, Sequence, Union

import numpy as np
import torch
import torchvision


class Dummy:
    def __init__(*args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self

    def __getitem__(self, *args, **kwargs):
        return self


def set_seed(seed: int):
    assert seed is not None, "Must provide a seed"
    # if seed is None:
    #     seed = random.getrandbits(32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def get_lr(optimizer):
    if type(optimizer) == dict:
        param_groups = optimizer["param_groups"]
    else:
        param_groups = optimizer.param_groups

    return param_groups[0]["lr"]


def parse_device(device: Union[str, List[str]]):
    if isinstance(device, str):
        if device in ["cpu", "cuda"]:
            return device
        elif device.startswith("cuda:"):
            return device
        else:
            raise ValueError
    elif isinstance(device, int):
        return device
    elif isinstance(device, Iterable):
        return [parse_device(_device) for _device in device]
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"


def get_device(m: torch.nn.Module):
    return next(m.parameters()).device


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    gettrace = getattr(sys, "gettrace", lambda: None)
    return gettrace() is not None


def get_cuda():
    return "cuda" if torch.cuda.is_available() else "cpu"


def iter_to_device(
    data_dict, device
) -> Union[Sequence[torch.Tensor], Dict[str, torch.Tensor]]:
    return apply_lambda_to_iter(
        data_dict, lambda v: v.to(device) if isinstance(v, torch.Tensor) else v
    )


def apply_lambda_to_iter(iterable, fn: Callable):
    if isinstance(iterable, dict):
        return {
            k: apply_lambda_to_iter(v, fn) if isinstance(v, (Iterable)) else fn(v)
            for k, v in iterable.items()
        }
    elif isinstance(iterable, list):
        return [apply_lambda_to_iter(v, fn) for v in iterable]
    else:
        return fn(iterable)


def iter_to_numpy(data_dict):
    return apply_lambda_to_iter(
        data_dict,
        lambda v: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v,
    )


def convert_to_HWC(tensor, input_format):  # tensor: numpy array
    assert len(set(input_format)) == len(
        input_format
    ), "You can not use the same dimension shordhand twice. \
        input_format: {}".format(
        input_format
    )
    assert len(tensor.shape) == len(
        input_format
    ), "size of input tensor and input format are different. \
        tensor shape: {}, input_format: {}".format(
        tensor.shape, input_format
    )
    input_format = input_format.upper()

    if len(input_format) == 4:
        index = [input_format.find(c) for c in "NCHW"]
        tensor_NCHW = tensor.transpose(index)
        tensor_CHW = torchvision.utils.make_grid(tensor_NCHW)
        return tensor_CHW.transpose(1, 2, 0)

    if len(input_format) == 3:
        index = [input_format.find(c) for c in "HWC"]
        tensor_HWC = tensor.transpose(index)
        if tensor_HWC.shape[2] == 1:
            tensor_HWC = np.concatenate([tensor_HWC, tensor_HWC, tensor_HWC], 2)
        return tensor_HWC

    if len(input_format) == 2:
        index = [input_format.find(c) for c in "HW"]
        tensor = tensor.transpose(index)
        tensor = np.stack([tensor, tensor, tensor], 2)
        return tensor

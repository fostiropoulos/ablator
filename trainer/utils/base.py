import typing as ty
from copy import deepcopy
from multiprocessing import Process

import numpy as np
import torch
from pynvml.smi import nvidia_smi

from trainer.config.main import Enum
# from trainer.main.configs import RunConfig
import random


class Dummy:
    def __init__(*args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self

    def __getitem__(self, *args, **kwargs):
        return self


def iter_to_numpy(iterable):
    return apply_lambda_to_iter(
        iterable,
        lambda v: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v,
    )


def apply_lambda_to_iter(iterable, fn: ty.Callable):
    if isinstance(iterable, dict):
        return {
            k: apply_lambda_to_iter(v, fn) if isinstance(v, (ty.Iterable)) else fn(v)
            for k, v in iterable.items()
        }
    elif isinstance(iterable, list):
        return [apply_lambda_to_iter(v, fn) for v in iterable]
    else:
        return fn(iterable)


def set_seed(seed: int):
    assert seed is not None, "Must provide a seed"
    # if seed is None:
    #     seed = random.getrandbits(32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


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


# def get_memory_use(model, config: RunConfig):
#     """get_memory_use returns the memory of the model and also works as a smoke_test"""

#     model: ModelWrapper
#     run_config = deepcopy(config)

#     run_config.train_config.verbose = False

#     torch.cuda.empty_cache()

#     smi = nvidia_smi.getInstance()

#     p: Process = model.mock_train(run_config, block=False)

#     mem_usages = []

#     while p.is_alive():
#         mem_usage = get_process_memory(run_config.uid, smi_instance=smi)
#         mem_usages.append(mem_usage)
#     del smi
#     del run_config
#     torch.cuda.empty_cache()
#     return max(mem_usages)


# def get_process_memory(process_name, smi_instance=None):
#     if smi_instance is None:
#         smi_instance = nvidia_smi.getInstance()
#     used_memory = 0
#     for gpu in smi_instance.DeviceQuery()["gpu"]:
#         if "processes" in gpu and gpu["processes"] is not None:
#             for process in gpu["processes"]:
#                 if process["process_name"] == process_name:
#                     used_memory += process["used_memory"]
#     return used_memory

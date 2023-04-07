

from copy import deepcopy
import typing as ty

import numpy as np
import torch

from trainer.config.main import Enum

try:
    from pynvml.smi import nvidia_smi as smi
except ImportError:
    smi = None
from multiprocessing import Process

from trainer.config.run import RunConfig


class MemType(Enum):
    used = "used"
    total = "total"
    free = "free"

def get_gpu_max_mem() -> ty.List[int]:
    return get_gpu_mem(mem_type=MemType.total)


def get_gpu_mem(mem_type: ty.Literal["used","total","free"] = "total") -> ty.List[int]:

    # TODO: waiting for fix: https://github.com/pytorch/pytorch/issues/86493
    instance = smi.getInstance()
    memory = []
    for gpu in instance.DeviceQuery()["gpu"]:
        memory.append(gpu["fb_memory_usage"][mem_type.value])
    return memory


def get_gpu_cur_mem():
    return get_gpu_mem(MemType.used)


def get_available_mem():
    return sum(get_gpu_cur_mem())


def get_least_used_gpu():
    return np.argmax(get_gpu_cur_mem())


def get_process_memory(process_name, nvidia_smi=None):
    if nvidia_smi is None:
        nvidia_smi = smi.getInstance()
    used_memory = 0
    for gpu in nvidia_smi.DeviceQuery()["gpu"]:
        if "processes" in gpu and gpu["processes"] is not None:
            for process in gpu["processes"]:
                if process["process_name"] == process_name:
                    used_memory += process["used_memory"]
    return used_memory


def get_memory_use(model, config: RunConfig):
    """get_memory_use returns the memory of the model and also works as a smoke_test"""
    from pynvml.smi import nvidia_smi
    from trainer import ModelWrapper
    model: ModelWrapper
    # TODO not clear how smi is used exactly.
    run_config = deepcopy(config)

    run_config.train_config.verbose = False

    torch.cuda.empty_cache()

    smi = nvidia_smi.getInstance()

    p: Process = model.mock_train(run_config, block=False)

    mem_usages = []

    while p.is_alive():
        mem_usage = get_process_memory(run_config.uid, nvidia_smi=smi)
        mem_usages.append(mem_usage)
    del smi
    del run_config
    torch.cuda.empty_cache()
    return max(mem_usages)

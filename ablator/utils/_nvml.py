import multiprocessing as mp
from ctypes import c_ssize_t, c_uint, c_ulonglong
from pathlib import Path

import torch
from joblib import Memory

cache_location = Path().home().joinpath(".cache", "ablator")
memory = Memory(cache_location, verbose=0)


def _mock_fn(q, l):
    torch.randn(100).to("cuda")
    q.put("done")
    l.acquire()


def _make_locks(n):
    locks = [mp.Lock() for i in range(n)]
    for l in locks:
        l.acquire()
    return locks


def _make_processes(locks, q):
    processes = [mp.Process(target=_mock_fn, args=(q, lock)) for lock in locks]
    for p in processes:
        p.start()
    pids = {p.pid for p in processes}
    return pids


def _sync_processes(q, n):
    for _ in range(n):
        q.get()


def _release_locks(locks):
    for lock in locks:
        lock.release()


def _get_running_pids(smi):
    instance = smi.getInstance()
    device = instance.DeviceQuery()
    running_pids = []
    for gpu in device["gpu"]:
        if gpu["processes"] is None:
            continue
        for p in gpu["processes"]:
            running_pids.append(p["pid"])
    return set(running_pids)


@memory.cache(ignore=["smi"])
def _is_smi_bug(smi, n_procs=2):
    # This is not a reported bug, but apparently there is a padding
    # to each process, causes a strange error.
    # The docs do not report any padding:
    # https://docs.nvidia.com/deploy/nvml-api/structnvmlProcessInfo__t.html#structnvmlProcessInfo__t
    # The byte-order also differs (e.g. pid, usedGpuMemory, gpuInstanceId, computeInstanceId)
    if not torch.cuda.is_available():
        return False
    try:
        torch.multiprocessing.set_start_method("spawn")
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    q = mp.Queue()
    locks = _make_locks(n_procs)
    pids = _make_processes(locks, q)
    assert len(pids) == n_procs
    _sync_processes(q, n_procs)
    running_pids = _get_running_pids(smi)
    _release_locks(locks)
    if len(pids.difference(running_pids)) == 0:
        return False

    return True


# pylint: disable=protected-access
def patch_smi(smi, nvml):
    smi._is_id = True
    if not _is_smi_bug(smi):
        return

    class c_nvmlProcessInfo_t(nvml._PrintableStructure):
        _fields_ = [
            ("pid", c_uint),
            ("usedGpuMemory", c_ulonglong),
            ("gpuInstanceId", c_uint),
            ("computeInstanceId", c_uint),
            ("pad", c_ssize_t),
        ]
        _fmt_ = {
            "usedGpuMemory": "%d B",
        }

    nvml.c_nvmlProcessInfo_t = c_nvmlProcessInfo_t

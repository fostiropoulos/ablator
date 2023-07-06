from pathlib import Path
from ablator.modules.storage.remote import RemoteConfig
import os
import getpass
import time
import sys
import torch
import pytest


def write_rand_tensors(tmp_path: Path, n=2):
    tensors = []
    for i in range(n):
        a = torch.rand(100)
        torch.save(a, tmp_path.joinpath(f"t_{i}.pt"))
        tensors.append(a)
    return tensors


def load_rand_tensors(tmp_path: Path, n=2):
    tensors = []
    for i in range(n):
        a = torch.load(tmp_path.joinpath(f"t_{i}.pt"))
        tensors.append(a)
    return tensors


def assert_tensor_list_eq(a, b):
    assert all([all(_a == _b) for _a, _b in zip(a, b)])

@pytest.mark.skipif(sys.platform == 'win32', reason="Rysnc does not support by Windows")
def test_remote(tmp_path: Path):
    username = getpass.getuser()
    hostname = "localhost"
    local_path = tmp_path.joinpath("local_path")
    local_path.mkdir()
    cfg = RemoteConfig(remote_path=tmp_path, username=username, hostname=hostname)
    tensors = write_rand_tensors(local_path)
    cfg.rsync_up(local_path, "remote_path", run_async=False)
    remote_tensors = load_rand_tensors(tmp_path.joinpath("remote_path", "local_path"))
    assert_tensor_list_eq(tensors, remote_tensors)

    time.sleep(0.5)

    new_remote_tensors = write_rand_tensors(
        tmp_path.joinpath("remote_path", "local_path")
    )
    cfg.rsync_down(local_path=local_path, destination="remote_path", run_async=False)

    new_local_tensors = load_rand_tensors(local_path)
    assert_tensor_list_eq(new_local_tensors, new_remote_tensors)


if __name__ == "__main__":
    import shutil
    tmp_path = Path("/tmp/remote_test")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(exist_ok=True)
    test_remote(tmp_path)
    breakpoint()

    pass

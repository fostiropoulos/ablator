from pathlib import Path

import pytest
from ablator.modules.storage.remote import RemoteConfig
import getpass
import time

import torch


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


@pytest.mark.skip("Remote Config is obsolete")
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
    from tests.conftest import run_tests_local

    l = locals()
    fn_names = [fn for fn in l if fn.startswith("test_")]
    test_fns = [l[fn] for fn in fn_names]
    run_tests_local(test_fns)

from pathlib import Path
import sys
import shutil
import os
import pytest

from ablator.config.rclone import RemoteRcloneConfig
from ablator.modules.storage.rclone import RcloneConfig
from ablator.mp.utils import make_rclone_config
import tests.ray_models.model

pytestmark = pytest.mark.skipif(
    sys.platform != 'linux',
    reason="Only runs on Linux"
)


def assert_error_msg(func: callable, msg: str):
    try:
        func()
        assert False
    except Exception as e:
        if not msg == str(e):
            raise e


def test_rclone(tmp_path: Path):

    remote_path = tmp_path / "mntremote"
    local_path = tmp_path/"mntlocal"
    shutil.rmtree(remote_path, ignore_errors=True)
    shutil.rmtree(local_path, ignore_errors=True)
    os.makedirs(remote_path, exist_ok=True)
    os.makedirs(local_path, exist_ok=True)
    remoteRcloneConfig = RemoteRcloneConfig(
        host="localhost",
        key_file="~/.ssh/id_rsa",
        user="",
        remote_path=str(remote_path),
    )
    remoteRcloneConfig.startMount(local_path, verbose=False)
    assert remoteRcloneConfig.rcloneWrapper.get_rclone_status() is True

    shutil.rmtree(remote_path, ignore_errors=True)
    shutil.rmtree(local_path, ignore_errors=True)
    os.makedirs(local_path, exist_ok=True)
    assert_error_msg(lambda: remoteRcloneConfig.startMount(local_path, verbose=False),
                     "rclone running with error, please check the config again!"),


def test_make_config(tmp_path: Path, make_config):
    run_config = make_config(tmp_path=tmp_path)
    run_config.remote_rclone_config = RemoteRcloneConfig(
        host="localhost",
        user="",
        key_file="~/.ssh/id_rsa",
    )
    make_rclone_config(run_config)


if __name__ == "__main__":
    test_rclone(Path.home())

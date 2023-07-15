from pathlib import Path
from ablator.main.configs import RemoteRcloneConfig
from ablator.modules.storage.rclone import RcloneConfig
import sys
import shutil
import os


def assert_error_msg(func: callable, msg: str):
    try:
        func()
        assert False
    except Exception as e:
        if not msg == str(e):
            raise e


def test_rclone(tmp_path: Path):
    if sys.platform != "linux":
        return

    remote_path = tmp_path / "mntremote"
    local_path = tmp_path/"mntlocal"
    shutil.rmtree(remote_path, ignore_errors=True)
    shutil.rmtree(local_path, ignore_errors=True)
    os.makedirs(remote_path, exist_ok=True)
    os.makedirs(local_path, exist_ok=True)
    gcs_config = RemoteRcloneConfig(
        host="localhost",
        key_file="~/.ssh/id_rsa",
        user="",
        remote_path=str(remote_path),
    )
    gcs_config.startMount(local_path, verbose=False)
    assert gcs_config.inspect_mount_status() is True

    shutil.rmtree(remote_path, ignore_errors=True)
    shutil.rmtree(local_path, ignore_errors=True)
    os.makedirs(local_path, exist_ok=True)
    assert_error_msg(lambda: gcs_config.startMount(local_path, verbose=False),
                     "rclone running with error, please check the config again!"),


if __name__ == "__main__":
    test_rclone(Path.home())

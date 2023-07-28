
from pathlib import Path
import sys
import subprocess

from pyrclone import RCloneWrapper
from ablator.config.mp import ParallelConfig
from ablator.config.main import ConfigBase, configclass
from ablator.config.rclone import allowed_rclone_remote_configs


def make_rclone_config(run_config: ParallelConfig):
    count = 0
    rclone_config = None
    for rclone_config_name in allowed_rclone_remote_configs:
        config = getattr(run_config, rclone_config_name)
        if config:
            count += 1
            rclone_config = config
    assert count <= 1, "You can just have one central remote repository"
    if rclone_config is not None:
        run_config.rclone_config = rclone_config


@configclass
class RcloneConfig(ConfigBase):
    config_name: str
    remote_path: str = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rcloneWrapper = RCloneWrapper({self.config_name: self.to_dict()})

    def get_remote_path_prefix(self) -> str:
        return f"{self.config_name}:{self.remote_path}"

    def startMount(self, expriement_dir: Path, verbose=False):
        self.rcloneWrapper.run_cmd("lsd", [self.get_remote_path_prefix()])
        self.experiment_dir = str(expriement_dir)
        if sys.platform == "win32":
            command = ["--rc",
                       "--gcs-bucket-policy-only",
                       "--vfs-cache-mode", "writes", "--dir-cache-time", "10s",
                       "--poll-interval", "10s", "-o", 'FileSecurity=D:P(A;;FA;;;WD)',
                       "--stats", "10s", "--transfers", "64"]
            if verbose is True:
                command.append("-vv")
            self.rcloneProcess = self.rcloneWrapper.mount(f"{self.get_remote_path_prefix()}", self.experiment_dir, command, verbose=verbose)
        else:
            subprocess.run(["umount", "-f", self.experiment_dir])
            command = [
                "--rc", '--gcs-bucket-policy-only', '--dir-cache-time', '10s', '--poll-interval', '10s', '--vfs-cache-mode', 'writes', "--transfers", "64"]
            if verbose is True:
                command.append("-vv")
            self.rcloneProcess = self.rcloneWrapper.mount(f"{self.get_remote_path_prefix()}", self.experiment_dir, command, verbose=verbose)
        if self.rcloneWrapper.get_rclone_status() is not True:
            raise RuntimeError("rclone is not running")

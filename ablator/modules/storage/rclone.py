from ablator.config.main import ConfigBase, configclass
from ablator.config.types import Stateless
from pyrclone import RCloneWrapper
from pathlib import Path
import sys
import threading
from ablator.modules.loggers.file import FileLogger
import time
import subprocess


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
        for i in range(10, 0, -1):
            print("wait for rclone mouting", i)
            time.sleep(1)
        if self.inspect_mount_status() is not True:
            raise RuntimeError("rclone is not running")

    def inspect_mount_status(self):
        if self.rcloneProcess is None:
            return False
        if self.rcloneProcess.poll() is not None:
            return False
        return True

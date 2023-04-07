from multiprocessing import Process
from pathlib import Path
import socket
from typing import Optional
from trainer.modules.logging.file import FileLogger

import trainer.utils.file as futils

from trainer.config.main import ConfigBase, configclass
import subprocess
import os


@configclass
class RsyncConfig(ConfigBase):
    username: str
    hostname: str
    path: str
    port: Optional[int] = None
    exclude_glob: Optional[str] = None
    exclude_chkpts: bool = False

    @property
    def full_path(self):
        return f"{self.username}@{self.hostname}:{self.path}"

    def __make_cmd_up(self, model_dir):
        username = self.username
        host = self.hostname
        path = Path(self.path) / Path(model_dir).parent.name
        cmd = f'rsync -art --rsync-path="mkdir -p {path} && rsync"'
        args = 'ssh -o "StrictHostKeyChecking=no"'
        if self.port is not None:
            args += f" -p {self.port}"
        cmd += f' -e "{args}"'
        if self.exclude_glob is not None:
            cmd += f' --exclude="{self.exclude_glob}"'
        if self.exclude_chkpts:
            cmd += f' --exclude="*.pt"'
        cmd += f" {model_dir}  {username}@{host}:{path}"
        return cmd

    def _make_cmd_down(self, model_dir, verbose=True):
        username = self.username
        host = self.hostname
        path = Path(self.path) / Path(model_dir).parent.name / Path(model_dir).name

        cmd = [f"rsync", "-art", f"--rsync-path", f"mkdir -p {path} && rsync"]
        if verbose:
            cmd[1] += "v"
        args = 'ssh -o "StrictHostKeyChecking=no"'
        if self.port is not None:
            args += f" -p {self.port}"
        cmd += [f"-e", f"{args}"]
        cmd += [f"{username}@{host}:{path}", str(Path(model_dir).parent)]
        return cmd

    def _rsync_in_process(
        self,
        model_dir,
    ):

        cmd = self.__make_cmd_up(model_dir)
        subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, preexec_fn=os.setsid)

    def rsync(
        self,
        model_dir,
        timeout=360,
        run_async=True,
    ):

        cmd = self.__make_cmd_up(model_dir)
        p = Process(target=futils.run_cmd_wait, args=(cmd, timeout))
        p.start()
        if not run_async:
            p.join()


@configclass
class GcpConfig(ConfigBase):
    bucket: str
    exclude_glob: Optional[str] = None
    exclude_chkpts: bool = False

    def _make_cmd_up(self, model_dir, bucket: Optional[str] = None, verbose=True):
        if bucket is None:
            destination = Path(self.bucket) / Path(model_dir).parent.name
        else:
            destination = Path(bucket)
        src = Path(model_dir).parent
        cmd = ["gsutil", "-m", f"rsync", "-r"]
        # if verbose:
        #     cmd[-1] += "v"
        if self.exclude_glob is not None:
            cmd += [f"--exclude", f"{self.exclude_glob}"]
        if self.exclude_chkpts:
            cmd += [f"--exclude", "*.pt"]
        cmd += [f"{src}", f"gs://{destination}"]
        return cmd

    def _make_cmd_down(self, model_dir, verbose=True):
        src = Path(self.bucket) / Path(model_dir).parent.name
        destination = Path(model_dir).parent
        cmd = ["gsutil", "-m", f"rsync", "-r"]
        # if verbose:
        #     cmd[-1] += "v"
        cmd += [f"gs://{src}", f"{destination}"]
        return cmd

    def rsync_up(
        self,
        model_dir,
        bucket: Optional[str] = None,
        logger: Optional[FileLogger] = None,
        verbose=True,
    ):
        cmd = self._make_cmd_up(model_dir, bucket)
        p = self._make_process(cmd, logger is not None)
        hostname = socket.gethostname()
        if logger is not None:
            logger.info(f"Rsync {hostname}:{cmd[-2]} to {cmd[-1]}")
        p.wait()

    def _make_process(self, cmd, verbose):
        if verbose:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL
        else:
            stdout = subprocess.PIPE
            stderr = subprocess.PIPE
        p = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, preexec_fn=os.setsid)
        return p

    def rsync_down(self, model_dir, logger: Optional[FileLogger] = None, verbose=True):
        cmd = self._make_cmd_down(model_dir)
        p = self._make_process(cmd, verbose)

        hostname = socket.gethostname()
        if logger is not None:
            logger.info(f"Rsync {cmd[-2]} to {hostname}:{cmd[-1]}")
        p.wait()

    def rsync_down_node(
        self,
        node_hostname,
        model_dir,
        logger: Optional[FileLogger] = None,
        verbose=True,
    ):

        zone_cmd = f'gcloud compute instances list --filter="name={node_hostname}" --format "get(zone)"'
        p = subprocess.Popen(
            zone_cmd, shell=True, stdout=subprocess.PIPE, preexec_fn=os.setsid
        )
        p.wait()
        _out = p.stdout.readline().decode()
        zone = _out.rstrip().split("/")[-1]

        cmd = self._make_cmd_down(model_dir)
        cmd = [
            "gcloud",
            "compute",
            "ssh",
            node_hostname,
            "--zone",
            zone,
            "--tunnel-through-iap",
            "--quiet",
            "--",
            "mkdir",
            "-p",
            f"{model_dir};",
        ] + cmd

        p = subprocess.Popen(
            " ".join(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
            shell=True,
        )
        if logger is not None:
            logger.info(f"Rsync {cmd[-2]} to {node_hostname}:{cmd[-1]}")

        p.wait()

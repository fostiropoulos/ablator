import os
import signal
import subprocess
import traceback
from multiprocessing import Process
from pathlib import Path


from ablator.config.main import ConfigBase, configclass
from ablator.config.types import Optional


def run_cmd_wait(cmd, timeout=300, raise_errors=False) -> None | str:
    # timeout is in seconds
    output = None
    with subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, preexec_fn=os.setsid
    ) as process:
        try:
            output = process.communicate(timeout=timeout)[0]
        except subprocess.TimeoutExpired as e:
            os.killpg(process.pid, signal.SIGINT)  # send signal to the process group
            output = process.communicate()[0]
            traceback.print_exc()
            if raise_errors:
                raise e
    return output


@configclass
class RemoteConfig(ConfigBase):
    remote_path: str
    username: str
    hostname: str
    port: Optional[int] = None
    exclude_glob: Optional[str] = None
    exclude_chkpts: bool = False

    def _make_cmd_up(self, local_path: Path, destination: str):
        username = self.username
        host = self.hostname
        path = Path(self.remote_path) / destination
        cmd = f'rsync -art --rsync-path="mkdir -p {path} && rsync"'
        args = 'ssh -o "StrictHostKeyChecking=no"'
        if self.port is not None:
            args += f" -p {self.port}"
        cmd += f' -e "{args}"'
        if self.exclude_glob is not None:
            cmd += f' --exclude="{self.exclude_glob}"'
        if self.exclude_chkpts:
            cmd += ' --exclude="*.pt"'
        cmd += f" {local_path}  {username}@{host}:{path}"
        return cmd

    def _make_cmd_down(self, local_path: Path, destination: str, verbose=True):
        username = self.username
        host = self.hostname
        path = Path(self.remote_path) / destination
        cmd = ["rsync", "-art", "--rsync-path", f'"mkdir -p {destination} && rsync"']
        if verbose:
            cmd[1] += "v"
        args = 'ssh -o "StrictHostKeyChecking=no"'
        if self.port is not None:
            args += f" -p {self.port}"
        cmd += ["-e", f'"{args}"']
        cmd += [f"{username}@{host}:{path}/", str(Path(local_path).parent)]
        return " ".join(cmd)

    def rsync_up(
        self,
        local_path: Path,
        destination: str,
        timeout_s: int | None = None,
        run_async=False,
    ):
        cmd = self._make_cmd_up(local_path=local_path, destination=destination)
        p = Process(target=run_cmd_wait, args=(cmd, timeout_s))
        p.start()
        if not run_async:
            p.join()

    def rsync_down(
        self,
        local_path: Path,
        destination: str,
        timeout_s: int | None = None,
        run_async=False,
    ):
        cmd = self._make_cmd_down(local_path=local_path, destination=destination)
        p = Process(target=run_cmd_wait, args=(cmd, timeout_s))
        p.start()
        if not run_async:
            p.join()

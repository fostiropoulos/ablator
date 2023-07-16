"""
File is obsolete to be replaced with rclone. TODO
"""
# pylint: skip-file
import os
import signal
import subprocess
import traceback
from multiprocessing import Process
from pathlib import Path


from ablator.config.main import ConfigBase, configclass
from ablator.config.types import Optional


def run_cmd_wait(cmd, timeout=300, raise_errors=False) -> Optional[str]:
    """
    Run a command and wait for it to finish.
    If the command takes longer than ``timeout`` seconds, kill it.
    If ``raise_errors`` is True, raise a ``TimeoutExpired`` exception.

    Parameters
    ----------
    cmd : str
        The command to run.
    timeout : int
        The timeout in seconds.
    raise_errors : bool
        Whether to raise errors.

    Returns
    -------
    str
        The output of the command.
    """
    # timeout is in seconds
    output = None
    with subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, preexec_fn=os.setsid
    ) as process:
        try:
            output = process.communicate(timeout=timeout)[0].decode("utf-8", errors="ignore")
        except subprocess.TimeoutExpired as e:
            os.killpg(process.pid, signal.SIGINT)  # send signal to the process group
            output = process.communicate()[0].decode("utf-8", errors="ignore")
            traceback.print_exc()
            if raise_errors:
                raise e
    return output


@configclass
class RemoteConfig(ConfigBase):
    """
    Configuration for a remote storage.

    Attributes
    ----------
    remote_path : str
        The path to the remote storage.
    username : str
        The username to use for the remote storage.
    hostname : str
        The hostname of the remote storage.
    port : None | int
        The port to use for the remote storage.
    exclude_glob : None | str
        A glob to exclude from the rsync.
    exclude_chkpts : bool
        Whether to exclude checkpoints from the rsync.
    """
    remote_path: str
    username: str
    hostname: str
    port: Optional[int] = None
    exclude_glob: Optional[str] = None
    exclude_chkpts: bool = False

    def _make_cmd_up(self, local_path: Path, destination: str):
        """
        Make the rsync command to upload files to the remote storage.

        Parameters
        ----------
        local_path : Path
            The local path to upload.
        destination : str
            The destination path in the remote storage.

        Returns
        -------
        str
            The rsync command.
        """
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
        """
        Make the rsync command to download files from the remote storage.

        Parameters
        ----------
        local_path : Path
            The local path to download to.
        destination : str
            The destination path in the remote storage.
        verbose : bool
            Whether to print the output.

        Returns
        -------
        str
            The rsync command.
        """
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
        """
        start a new process and upload files to the remote storage.

        Parameters
        ----------
        local_path : Path
            The local path to upload.
        destination : str
            The destination path in the remote storage.
        timeout_s : int | None
            The timeout in seconds.
        run_async : bool
            Whether to run the command asynchronously.
        """
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
        """
        start a new process and download files from the remote storage.

        Parameters
        ----------
        local_path : Path
            The local path to download to.
        destination : str
            The destination path in the remote storage.
        timeout_s : int | None
            The timeout in seconds.
        run_async : bool
            Whether to run the command asynchronously.
        """
        cmd = self._make_cmd_down(local_path=local_path, destination=destination)
        p = Process(target=run_cmd_wait, args=(cmd, timeout_s))
        p.start()
        if not run_async:
            p.join()

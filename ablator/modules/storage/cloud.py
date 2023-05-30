import json
import os
import socket
import subprocess
import typing as ty
from pathlib import Path
from ablator.config.main import ConfigBase, configclass
from ablator.config.types import Optional
from ablator.modules.loggers.file import FileLogger


@configclass
class GcpConfig(ConfigBase):
    """
    Configuration for Google Cloud Storage.

    Attributes
    ----------
    bucket : str
        The bucket to use.
    exclude_glob : Optional[str]
        A glob to exclude from the rsync.
    exclude_chkpts : bool
        Whether to exclude checkpoints from the rsync.
    """
    bucket: str
    exclude_glob: Optional[str] = None
    exclude_chkpts: bool = False

    def __init__(self, *args, **kwargs):
        """
        Initialize the GcpConfig class for managing Google Cloud Platform configurations.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.

        Raises
        ------
        AssertionError
            If the GCP instance is not found.

        Notes
        -----
        The IP address check avoids overly generic hostnames to match with existing instances.
        """
        super().__init__(*args, **kwargs)
        self.bucket = self.bucket.lstrip("gs:").lstrip("/").rstrip("/")
        self.list_bucket()

        hostname = socket.gethostname()
        nodes = self._find_gcp_nodes(hostname)
        # The IP address check avoids overly generic hostnames to match with existing instances.
        ip_address = socket.gethostbyname(hostname)
        assert (
            len(nodes) == 1
            and sum(
                network_interface["networkIP"] == ip_address
                for network_interface in nodes[0]["networkInterfaces"]
            )
            == 1
        ), "Can only use GcpConfig from Google Cloud Server. Consider switching to RemoteConfig."

    def _make_cmd_up(self, local_path: Path, destination: str):
        """
        Make the command to upload files to the bucket.

        Parameters
        ----------
        local_path : Path
            The local path to upload.
        destination : str
            Bucket path.

        Returns
        -------
        list[str]
            The command to upload the file.
        """

        destination = str(Path(self.bucket) / destination / local_path.name)
        src = local_path
        cmd = ["gsutil", "-m", "rsync", "-r"]
        if self.exclude_glob is not None:
            cmd += ["--exclude", f"{self.exclude_glob}"]
        if self.exclude_chkpts:
            cmd += ["--exclude", "*.pt"]
        cmd += [f"{src}", f"gs://{destination}"]
        return cmd

    def _make_cmd_down(self, src_path: str, local_path: Path):
        """
        Make the command to download files from the bucket.

        Parameters
        ----------
        src_path : str
            The source path in the bucket.
        local_path : Path
            The local path to download to.

        Returns
        -------
        list[str]
            The command to download the file.
        """
        src = Path(self.bucket) / src_path / local_path.name
        destination = local_path
        cmd = ["gsutil", "-m", "rsync", "-r"]
        cmd += [f"gs://{src}", f"{destination}"]
        return cmd

    def list_bucket(self, destination: str | None = None):
        """
        List the contents of a bucket. If destination is None, list the bucket itself.

        Parameters
        ----------
        destination : str | None
            Bucket path.

        Returns
        -------
        list[str]
            List of files in the bucket.
        """
        destination = str(
            Path(self.bucket) / destination
            if destination is not None
            else Path(self.bucket)
        )
        cmd = ["gsutil", "ls", f"gs://{destination}"]

        p = self._make_process(cmd, verbose=False)
        stdout, stderr = p.communicate()
        assert len(stderr) == 0, (
            f"There was an error running `{' '.join(cmd)}`. "
            "Make sure gsutil is installed and that the destination exists. "
            f"`{stderr.decode('utf-8').strip()}`"
        )
        return stdout.decode("utf-8").strip().split("\n")

    def rsync_up(
        self,
        local_path: Path,
        remote_path: str,
        logger: FileLogger | None = None,
    ):
        """
        Rsync files to the bucket.

        Parameters
        ----------
        local_path : Path
            The local path to upload.
        remote_path : str
            The destination path in the bucket.
        logger : FileLogger | None
            The logger to use.

        Raises
        ------
        AssertionError
            If the rsync fails.
        """
        cmd = self._make_cmd_up(local_path, remote_path)
        p = self._make_process(cmd, verbose=logger is not None)
        hostname = socket.gethostname()
        if logger is not None:
            logger.info(f"Rsync {hostname}:{cmd[-2]} to {cmd[-1]}")
        p.wait()

    def _make_process(self, cmd, verbose) -> subprocess.Popen:
        """
        Make a subprocess.Popen object.

        Parameters
        ----------
        cmd : list[str]
            The command to run.
        verbose : bool
            Whether to print the output.

        Returns
        -------
        subprocess.Popen
            The process object.
        """

        if verbose:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL
        else:
            stdout = subprocess.PIPE
            stderr = subprocess.PIPE
        p = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, preexec_fn=os.setsid)
        return p

    def _find_gcp_nodes(self, node_hostname: None | str = None) -> list[dict[str, ty.Any]]:
        """
        Find the GCP instances with the given hostname.

        Parameters
        ----------
        node_hostname : None | str
            The hostname of the node to find. If None, find all nodes.

        Returns
        -------
        list[dict[str, ty.Any]]
            List of GCP instances.

        Raises
        ------
        AssertionError
            no nodes are found.
        """
        cmd = ["gcloud", "compute", "instances", "list"]
        if node_hostname is not None:
            cmd += ["--filter", f'"{node_hostname}"']
        cmd += ["--format", "json"]
        p = self._make_process(cmd, verbose=False)
        stdout, stderr = p.communicate()
        assert len(stderr) == 0 and len(stdout) > 0
        return json.loads(stdout.decode("utf-8"))

    def rsync_down(
        self,
        remote_path: str,
        local_path: Path,
        logger: FileLogger | None = None,
        verbose=True,
    ):
        """
        Rsync files from the bucket.

        Parameters
        ----------
        remote_path : str
            The source path in the bucket.
        local_path : Path
            The local path to download to.
        logger : FileLogger | None
            The logger to use.
        verbose : bool
            Whether to print the output.
        """
        cmd = self._make_cmd_down(remote_path, local_path)
        p = self._make_process(cmd, verbose)
        hostname = socket.gethostname()
        if logger is not None:
            logger.info(f"Rsync {cmd[-2]} to {hostname}:{cmd[-1]}")
        p.wait()

    def rsync_down_node(
        self,
        node_hostname,
        remote_path: str,
        local_path: Path,
        logger: FileLogger | None = None,
        verbose=True,
    ):
        """
        Rsync files from the bucket to all nodes with the given hostname.

        Parameters
        ----------
        node_hostname : str
            The hostname of the nodes to rsync to.
        remote_path : str
            The source path in the bucket.
        local_path : Path
            The local path to download to.
        logger : FileLogger | None
            The logger to use.
        verbose : bool
            Whether to print the output.
        """
        nodes = self._find_gcp_nodes(node_hostname)
        ps: list[subprocess.Popen] = []
        for node in nodes:
            zone = node["zone"].split("/")[-1]
            name = node["name"]
            rsync_cmd = self._make_cmd_down(remote_path, local_path)
            cmd = [
                "gcloud",
                "compute",
                "ssh",
                name,
                "--zone",
                zone,
                "--tunnel-through-iap",
                "--quiet",
                "--",
                "mkdir",
                "-p",
                f"{local_path};",
            ] + rsync_cmd

            p = self._make_process(cmd, verbose)
            ps.append(p)
            if logger is not None:
                logger.info(f"Rsync {cmd[-2]} to {name}:{cmd[-1]}")

        for p in ps:
            p.wait()

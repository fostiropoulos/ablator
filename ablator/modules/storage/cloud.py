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
    bucket: str
    exclude_glob: Optional[str] = None
    exclude_chkpts: bool = False

    def __init__(self, *args, add_attributes=False, **kwargs):
        super().__init__(*args, add_attributes=add_attributes, **kwargs)
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
        destination = Path(self.bucket) / destination / local_path.name
        src = local_path
        cmd = ["gsutil", "-m", "rsync", "-r"]
        if self.exclude_glob is not None:
            cmd += ["--exclude", f"{self.exclude_glob}"]
        if self.exclude_chkpts:
            cmd += ["--exclude", "*.pt"]
        cmd += [f"{src}", f"gs://{destination}"]
        return cmd

    def _make_cmd_down(self, src_path: str, local_path: Path):
        src = Path(self.bucket) / src_path / local_path.name
        destination = local_path
        cmd = ["gsutil", "-m", "rsync", "-r"]
        cmd += [f"gs://{src}", f"{destination}"]
        return cmd

    def list_bucket(self, destination: str | None = None):
        destination = (
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
        cmd = self._make_cmd_up(local_path, remote_path)
        p = self._make_process(cmd, verbose=logger is not None)
        hostname = socket.gethostname()
        if logger is not None:
            logger.info(f"Rsync {hostname}:{cmd[-2]} to {cmd[-1]}")
        p.wait()

    def _make_process(self, cmd, verbose) -> subprocess.Popen:
        if verbose:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL
        else:
            stdout = subprocess.PIPE
            stderr = subprocess.PIPE
        p = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, preexec_fn=os.setsid)
        return p

    def _find_gcp_nodes(self, node_hostname: None | str = None) -> dict[str, ty.Any]:
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
        cmd = self._make_cmd_down(remote_path, local_path)
        p = self._make_process(cmd, verbose)
        hostname = socket.gethostname()
        if logger is not None:
            logger.info(f"Rsync {cmd[-2]} to {hostname}:{cmd[-1]}")
        p.wait()

    def rsync_down_nodes(
        self,
        node_hostname,
        remote_path: str,
        local_path: Path,
        logger: FileLogger | None = None,
        verbose=True,
    ):
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

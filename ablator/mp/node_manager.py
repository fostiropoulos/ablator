import getpass
import logging
import socket
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import paramiko
import ray
from ray.util.state import list_nodes, list_tasks
import numpy as np
import psutil
from ablator.utils.base import get_gpu_mem


@dataclass
class Resource:
    gpu_free_mem: dict[str, int]
    mem: int
    cpu_usage: float
    cpu_count: int
    running_tasks: list[str] = field(default_factory=lambda: [])

    @property
    def gpu_free_mem_arr(self) -> np.ndarray:
        return np.array(list(self.gpu_free_mem.values()))

    @property
    def cpu_mean_util(self) -> float:
        return np.array(self.cpu_usage).mean()

    @property
    def least_used_gpu(self):
        return min(self.gpu_free_mem, key=self.gpu_free_mem.get)


def make_private_key(home_path: Path):
    pkey_path = Path(home_path).joinpath(".ssh", "ablator_id_rsa")
    pkey_path.parent.mkdir(exist_ok=True)

    if not pkey_path.exists():
        pkey = paramiko.RSAKey.generate(bits=2048)
        with pkey_path.open("w", encoding="utf-8") as p:
            pkey.write_private_key(p)
    else:
        pkey = paramiko.RSAKey.from_private_key_file(pkey_path.as_posix())
    name = pkey.get_name()
    public_key = pkey.get_base64()

    hostname = socket.gethostname()
    node_ip = socket.gethostbyname(hostname)
    key = f"{name} {public_key} ablator-{hostname}@{node_ip}"
    return pkey, key


def utilization():
    free_gpu = get_gpu_mem("free")
    mem_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent(interval=2, percpu=True)
    cpu_count = psutil.cpu_count()

    return Resource(
        gpu_free_mem=free_gpu, mem=mem_usage, cpu_usage=cpu_usage, cpu_count=cpu_count
    )


@ray.remote
def update_node(node_ip, key):
    # check if key in authorized keys
    ssh_dir = Path.home().joinpath(".ssh")
    ssh_dir.mkdir(exist_ok=True)

    username = getpass.getuser()
    authorized_keys = ssh_dir.joinpath("authorized_keys")
    if authorized_keys.exists() and key in authorized_keys.read_text(encoding="utf-8"):
        return node_ip, username
    with authorized_keys.open("a", encoding="utf-8") as f:
        f.write(f"{key}\n")
    return node_ip, username


class NodeManager:
    def __init__(self, private_key_home: Path, ray_address: str | None = None):
        self.pkey, self.public_key = make_private_key(private_key_home)
        if not ray.is_initialized():
            ray.init(address=ray_address)
        elif not ray.is_initialized():
            raise RuntimeError("No ray cluster was found running.")

        self.ray_address = ray.get_runtime_context().gcs_address

        self.nodes: dict[str, str] = {}
        self.update()

    def update(self):
        futures = []
        nodes = {}
        for node in ray.nodes():
            node_ip = node["NodeManagerAddress"]
            node_alive = node["Alive"]
            if node_alive and node_ip not in self.nodes:
                futures.append(
                    update_node.options(resources={f"node:{node_ip}": 0.01}).remote(
                        node_ip, self.public_key
                    )
                )
            elif node_alive and node_ip not in nodes:
                nodes[node_ip] = self.nodes[node_ip]
        nodes.update(dict(ray.get(futures)))
        self.nodes = nodes

    def utilization(self, node_ips: list | str | None = None) -> dict[str, Resource]:
        return self.run_lambda(utilization, node_ips)

    def available_resources(
        self, node_ips: list | str | None = None
    ) -> dict[str, Resource]:
        results = self.utilization(node_ips)
        node_id_map: dict[str, str] = {
            n.node_id: n.node_ip for n in list_nodes(address=self.ray_address)
        }
        node_ip_tasks: dict[str, list[str]] = defaultdict(lambda: [])
        running_tasks = list_tasks(
            address=self.ray_address,
            filters=[
                ("state", "=", "RUNNING"),
                # exclude the utilization lambda from above
                ("func_or_class_name", "!=", "utilization"),
            ],
        )
        for task in running_tasks:
            if task.node_id in node_id_map:
                node_ip_tasks[node_id_map[task.node_id]].append(task.name)

        for node_ip in results:
            results[node_ip].running_tasks = node_ip_tasks[node_ip]
        return results

    def _parse_node_ips(self, node_ips: list | str | None = None) -> list[str]:
        _node_ips = []
        if node_ips is None:
            _node_ips = self.node_ips
        if isinstance(node_ips, str):
            _node_ips = [node_ips]
        if any(node_ip not in self.nodes for node_ip in _node_ips):
            raise RuntimeError(
                f"Not all {set(_node_ips)} found in running nodes: {set(self.node_ips)}."
            )
        return _node_ips

    @property
    def node_ips(self) -> list[str]:
        return list(self.nodes.keys())

    def run_lambda(self, fn, node_ips: list | str | None = None):
        self.update()
        futures = {}

        for node_ip in self._parse_node_ips(node_ips):
            futures[node_ip] = (
                ray.remote(fn).options(resources={f"node:{node_ip}": 0.001}).remote()
            )
        return {k: ray.get(v) for k, v in futures.items()}

    def run_cmd(
        self, cmd, node_ips: list | str | None = None, timeout: int = 20
    ) -> dict[str, str]:
        self.update()
        result = {}

        for node_ip in self._parse_node_ips(node_ips):
            node_username = self.nodes[node_ip]
            client = paramiko.SSHClient()
            policy = paramiko.AutoAddPolicy()
            client.set_missing_host_key_policy(policy)
            node_name = f"{node_username}@{node_ip}"
            try:
                client.connect(
                    node_ip, username=node_username, pkey=self.pkey, timeout=timeout
                )
                # _stdin, _stdout, _stderr
                _, _stdout, _ = client.exec_command(cmd)
                result[node_name] = _stdout.read().decode()
            # pylint: disable=broad-exception-caught
            except Exception as e:
                logging.error(
                    (
                        "Could not connect to %s. Make sure ssh is configured "
                        "and accessible from the ray head node. \n %s"
                    ),
                    node_name,
                    str(e),
                )
        return result

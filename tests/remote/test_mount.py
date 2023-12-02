import os
import platform
import random
import time
from pathlib import Path

import paramiko
import pytest
import ray
import torch
from torch import nn

from ablator import (
    ModelConfig,
    ModelWrapper,
    OptimizerConfig,
    ProtoTrainer,
    RemoteConfig,
    RunConfig,
    SearchSpace,
    Stateless,
    TrainConfig,
)
from ablator.main.mp import ParallelTrainer
from ablator.mp.node import MountServer, run_actor_node

IS_LINUX = "linux" in platform.system().lower()

if IS_LINUX:
    # must be imported after checking for OS. Otherwise
    # will throw an error.
    from rmount.server import RemoteServer  # noqa: E402

    pytestmark = pytest.mark.remote()
else:
    pytestmark = pytest.mark.skip(
        reason="RMount tests are only supported for Linux platforms."
    )


class SimpleModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))

    def forward(self, x: torch.Tensor):
        x = self.param
        if self.training:
            x = x + torch.rand_like(self.param) * 0.01
        return {"preds": x}, x.sum().abs()


class SimpleWrapper(ModelWrapper):
    def make_dataloader_train(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl

    def make_dataloader_val(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl


class SimpleTrainConfig(TrainConfig):
    epochs: Stateless[int]


class SimpleConfig(RunConfig):
    train_config: SimpleTrainConfig


def _write_private_key(tmp_path: Path):
    pkey_path = tmp_path.joinpath("id_rsa")
    pkey = paramiko.RSAKey.generate(bits=2048)
    with open(
        os.open(
            pkey_path.as_posix(),
            flags=(os.O_WRONLY | os.O_CREAT | os.O_TRUNC),
            mode=0o600,
        ),
        "w",
        encoding="utf-8",
    ) as p:
        pkey.write_private_key(p)
    public_key = f"ssh-rsa {pkey.get_base64()}"
    public_key_path = tmp_path.joinpath("id_rsa.pub")
    public_key_path.write_text(public_key)

    return public_key


def make_config_kwargs(ip: str, tmp_path: Path):
    cfg_kwargs = dict(
        remote_config=RemoteConfig(
            ssh={
                "host": ip,
                "user": "admin",
                "port": "2222",
                "key_file": tmp_path / "id_rsa",
            }
        ),
        experiment_dir=tmp_path / "ablator-exp",
        train_config=SimpleTrainConfig(
            dataset="test",
            batch_size=128,
            epochs=2,
            optimizer_config=OptimizerConfig(name="sgd", arguments={"lr": 0.1}),
            scheduler_config=None,
        ),
        model_config=ModelConfig(),
        verbose="silent",
        device="cpu",
        amp=False,
    )
    return cfg_kwargs


def folders_equal(folder_a, folder_b):
    # NOTE we might need to retry in case of concurrent writing
    for _ in range(30):
        try:
            files_a = [
                p.read_bytes()
                for p in sorted(folder_a.rglob("*"), key=lambda x: str(x))
                if p.is_file() and p.name != ".rmount"
            ]
            files_b = [
                p.read_bytes()
                for p in sorted(folder_b.rglob("*"), key=lambda x: str(x))
                if p.is_file() and p.name != ".rmount"
            ]

            if all(a == b for a, b in zip(files_a, files_b)) and len(files_a) == len(
                files_b
            ):
                return True

        except Exception:
            ...
        finally:
            time.sleep(1)
    return False


def test_mount(tmp_path: Path, volume_name):
    if volume_name is not None:
        local_path = None
        folder_a = Path("/ablator")
    else:
        local_path = tmp_path / "ablator-remote"
        folder_a = local_path

    pub_key = _write_private_key(tmp_path=tmp_path)
    server = RemoteServer(
        local_path=local_path,
        volume_name=volume_name,
        remote_path="/ablator",
        public_key=pub_key,
    )
    server.start()
    kwargs = make_config_kwargs(server.ip_address, tmp_path)
    config = SimpleConfig(**kwargs)

    wrapper = SimpleWrapper(SimpleModel)
    ablator = ProtoTrainer(wrapper=wrapper, run_config=config)
    with ablator:
        ablator.launch(tmp_path)
        # assert the artifacts are the same
        folder_b = Path(config.experiment_dir)
        folder_a = folder_a / ablator.run_config.experiment_id
        assert folders_equal(folder_a, folder_b)
        t = (folder_b / "results.json").read_text()
        ablator.run_config.train_config.epochs += 2
        ablator.launch(tmp_path, resume=True)
        assert folders_equal(folder_a, folder_b)
        # assert that it was updated
        assert t != (folder_b / "results.json").read_text()
    server.kill()


# we schedule first because ray cluster gets misconfigured afterward


@pytest.mark.order(2)
def test_mp_mount(tmp_path: Path, wrapper, make_config, ray_cluster, volume_name):
    if volume_name is not None:
        local_path = None
        folder_a = Path("/ablator")
    else:
        local_path = tmp_path / "ablator-remote"
        folder_a = local_path
    pub_key = _write_private_key(tmp_path=tmp_path)
    with RemoteServer(
        local_path=local_path,
        volume_name=volume_name,
        remote_path="/ablator",
        public_key=pub_key,
    ) as server:
        config = make_config(tmp_path.joinpath("test_mp_mount"))
        config.remote_config = RemoteConfig(
            ssh={
                "host": server.ip_address,
                "user": "admin",
                "port": "2222",
                "key_file": tmp_path / "id_rsa",
            }
        )
        config.search_space["train_config.optimizer_config.arguments.lr"] = SearchSpace(
            value_range=[1, 2], value_type="int"
        )
        config.optim_metrics = {"val_loss": "min"}
        config.optim_metric_name = "val_loss"
        config.total_trials = 2
        config.concurrent_trials = 1

        ablator = ParallelTrainer(wrapper=wrapper, run_config=config)

        with ablator:
            ablator.launch(working_directory=tmp_path)

            nodes = ablator.cluster_manager.available_resources.keys()
            assert (
                len(nodes) == ray_cluster.nodes
            ), "Check if the experiment run with same number of nodes as the cluster"

            folder_b = Path(config.experiment_dir)

            folder_a = folder_a / ablator.run_config.experiment_id

            assert folders_equal(
                folder_a,
                folder_b,
            )
            logs = (folder_a / "mp.log").read_text()
            assert all(n in logs for n in nodes)


def test_mount_error(tmp_path: Path):
    # Do not start server

    _write_private_key(tmp_path=tmp_path)
    kwargs = make_config_kwargs("172.17.0.1", tmp_path)
    config = RunConfig(**kwargs)

    wrapper = SimpleWrapper(SimpleModel)
    ablator = ProtoTrainer(wrapper=wrapper, run_config=config)
    with pytest.raises(RuntimeError, match="Could not mount to remote directory "):
        ablator._mount(timeout=10)


# TODO fix flaky test
@pytest.mark.skip
def test_mount_actor(tmp_path, volume_name, ray_cluster):
    if volume_name is not None:
        local_path = None
        folder_a = Path("/ablator")
    else:
        local_path = tmp_path / "ablator-remote"
        folder_a = local_path
    pub_key = _write_private_key(tmp_path=tmp_path)
    with RemoteServer(
        local_path=local_path,
        volume_name=volume_name,
        remote_path="/ablator",
        public_key=pub_key,
    ) as server:
        config = make_config_kwargs(server.ip_address, tmp_path)["remote_config"]
        config.remote_path = Path("/ablator")
        config.local_path = folder_a
        for node_ip in ray_cluster.node_ips():
            if node_ip.startswith("68"):
                continue
            mount_server = run_actor_node(
                MountServer,
                cuda=False,
                node=node_ip,
                kwargs={"config": config},
            )
            mount_server.mount.remote()
            file_name = f"{random.randint(0,100)}"
            folder_a.joinpath(file_name).write_text(file_name)
            for _ in range(30):
                file_names = [
                    p.name for p in ray.get(mount_server.remote_files.remote())
                ]
                if file_name in file_names:
                    break
                time.sleep(1)
            assert file_name in file_names


if __name__ == "__main__":
    from tests.conftest import run_tests_local
    from tests.test_plugins.model import (
        TestWrapper,
        MyCustomModel,
        _make_config,
    )

    if not IS_LINUX:
        raise NotImplementedError(
            "Tests in this file are not supported for non-linux platforms"
        )

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]
    kwargs = {
        "wrapper": TestWrapper(MyCustomModel),
        "make_config": _make_config,
        "volume_name": None,
    }
    run_tests_local(test_fns, kwargs=kwargs)

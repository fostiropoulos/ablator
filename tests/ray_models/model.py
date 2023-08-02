import os
import time
from pathlib import Path
from filelock import FileLock

import pytest
import ray
import torch
from torch import nn

from ablator import (
    Derived,
    ModelConfig,
    ModelWrapper,
    OptimizerConfig,
    RunConfig,
    Stateless,
    TrainConfig,
)
from ablator.analysis.results import Results
from ablator.config.main import configclass
from ablator.config.mp import ParallelConfig, SearchSpace
from ablator.main.mp import ParallelTrainer

N_MOCK_NODES = 10

N_BATCHES = 100
DEVICE = "cpu"

WORKING_DIR = Path(__file__).parent.parent.as_posix()


class MyCustomException(Exception):
    pass


class CustomModelConfig(ModelConfig):
    lr: Derived[int]
    lr_error_limit: int = 5
    mock_param: int = 0


class CustomTrainConfig(TrainConfig):
    epochs: Stateless[int]


@configclass
class MyParallelConfig(ParallelConfig):
    model_config: CustomModelConfig
    train_config: CustomTrainConfig


class TestWrapper(ModelWrapper):
    def make_dataloader_train(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(N_BATCHES)]
        return dl

    def make_dataloader_val(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(N_BATCHES)]
        return dl

    def config_parser(self, run_config: MyParallelConfig):
        run_config.model_config.lr = (
            run_config.train_config.optimizer_config.arguments.lr
        )
        return super().config_parser(run_config)


class MyCustomModel(nn.Module):
    def __init__(self, config: CustomModelConfig) -> None:
        super().__init__()
        self.lr = config.lr
        self.lr_error_limit = config.lr_error_limit
        self.param = nn.Parameter(torch.ones(100, 1))
        self.itr = 0

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        self.itr += 1
        return {"preds": x}, x.sum().abs()


class MyErrorCustomModel(MyCustomModel):
    def __init__(self, config: CustomModelConfig) -> None:
        super().__init__(config)
        self.exception_class = MyCustomException

    def forward(self, x: torch.Tensor):
        out, loss = super().forward(x)
        if self.itr > 10 and self.lr >= self.lr_error_limit:
            raise MyCustomException("large lr.")
        return out, loss


class MyDivCustomModel(MyCustomModel):
    def forward(self, x: torch.Tensor):
        out, loss = super().forward(x)
        if self.itr > 10 and self.lr >= self.lr_error_limit:
            return out, loss + torch.nan
        return out, loss


@pytest.fixture()
def divergent_wrapper() -> TestWrapper:
    return TestWrapper(MyDivCustomModel)


@pytest.fixture()
def error_wrapper():
    return TestWrapper(MyErrorCustomModel)


@pytest.fixture()
def wrapper():
    return TestWrapper(MyCustomModel)


@pytest.fixture()
def make_config():
    return _make_config


@pytest.fixture(scope="session")
def working_dir():
    return WORKING_DIR


def _remote_fn(gpu_id: int, gpu_manager=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
    t = torch.randn(300, 100, 300).to(f"cuda")
    time.sleep(2)
    if gpu_manager is not None:
        ray.get(gpu_manager.unlock.remote(gpu_id))
    return gpu_id


@pytest.fixture(scope="session")
def ablator(
    tmpdir_factory,
    working_dir,
    main_ray_cluster,
):
    tmp_path = Path(tmpdir_factory.getbasetemp())
    with FileLock(tmp_path.joinpath(".ray_cluster")):
        main_ray_cluster.setUp()
        assert len(main_ray_cluster.node_ips()) == main_ray_cluster.nodes + 1
        n_trials = 9
        error_wrapper = TestWrapper(MyErrorCustomModel)
        config = _make_config(tmp_path, search_space_limit=n_trials)
        config.experiment_dir = tmp_path
        config.total_trials = n_trials
        ablator = ParallelTrainer(wrapper=error_wrapper, run_config=config)
        ablator.launch(working_dir)
        return ablator


@pytest.fixture(scope="session")
def ablator_results(ablator):
    config = ablator.run_config
    return Results(config, ablator.experiment_dir)


@pytest.fixture()
def remote_fn():
    return _remote_fn


def _make_config(
    tmp_path: Path,
    search_space_limit: int | None = None,
    gpu_util: int = 100,
    search_space: dict | None = None,
):
    optimizer_config = OptimizerConfig(name="sgd", arguments={"lr": 0.1})
    train_config = CustomTrainConfig(
        dataset="test",
        batch_size=128,
        epochs=2,
        optimizer_config=optimizer_config,
        scheduler_config=None,
    )
    if search_space is None:
        n_bins = None
        if search_space_limit is not None:
            n_vals = int((search_space_limit) ** 0.5)
            n_bins = n_vals
        else:
            n_vals = 10
        search_space = {
            "train_config.optimizer_config.arguments.lr": SearchSpace(
                value_range=[0, 19],
                value_type="float",
                n_bins=n_bins,
            ),
            "model_config.mock_param": SearchSpace(
                categorical_values=list(range(n_vals)),
            ),
        }

    config = MyParallelConfig(
        experiment_dir=tmp_path,
        train_config=train_config,
        model_config=CustomModelConfig(),
        verbose="silent",
        device="cuda" if torch.cuda.is_available() else "cpu",
        amp=False,
        search_space=search_space,
        optim_metrics={"val_loss": "min"} if search_space_limit is None else None,
        total_trials=10,
        search_algo="tpe" if search_space_limit is None else "grid",
        concurrent_trials=10,
        gpu_mb_per_experiment=gpu_util,
    )
    return config

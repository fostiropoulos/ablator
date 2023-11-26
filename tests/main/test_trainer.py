import copy
import os
from pathlib import Path
import git
import torch
import numpy as np
from torch import nn
from ablator import (
    ModelConfig,
    OptimizerConfig,
    RunConfig,
    TrainConfig,
    ProtoTrainer,
    ModelWrapper,
    SchedulerConfig,
)
import pytest
import random
from ablator.config.hpo import SearchSpace
from ablator.config.mp import ParallelConfig
from ablator.main.model.main import EvaluationError
from ablator.main.mp import ParallelTrainer

from ablator.modules.scheduler import SCHEDULER_CONFIG_MAP, PlateauConfig

_optimizer_config = OptimizerConfig(name="sgd", arguments={"lr": 0.1})


class MyPlateauConfig(PlateauConfig):
    def __init__(self, *args, debug: bool = False, **kwargs):
        super().__init__(*args, debug=debug, **kwargs)

    def make_scheduler(self, model: nn.Module, optimizer):
        scheduler_cls = self.init_scheduler(model, optimizer)
        scheduler_cls.step_when = "val"
        return scheduler_cls


class MyTrainConfig(TrainConfig):
    scheduler_config: MyPlateauConfig


class MyParallelConfig(ParallelConfig):
    train_config: MyTrainConfig


_train_config = TrainConfig(
    dataset="test",
    batch_size=128,
    epochs=2,
    optimizer_config=_optimizer_config,
    scheduler_config=None,
)

_config = ParallelConfig(
    train_config=_train_config,
    model_config=ModelConfig(),
    verbose="silent",
    device="cpu",
    amp=False,
    search_space={
        "train_config.optimizer_config.arguments.lr": SearchSpace(
            value_range=[0.01, 0.1], value_type="float"
        )
    },
    optim_metrics={"val_loss": "min"},
    optim_metric_name="val_loss",
    total_trials=2,
)

scheduler_args = {"cycle": {"max_lr": 0.1, "total_steps": 1000}}


@pytest.fixture
def config():
    return copy.deepcopy(_config)


class MyCustomModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))

    def forward(self, x: torch.Tensor):
        x = self.param
        if self.training:
            x = x + torch.rand_like(self.param) * 0.01
        return {"preds": x}, x.sum().abs()


class TestWrapper(ModelWrapper):
    def make_dataloader_train(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl

    def make_dataloader_val(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl


class MyCustomModel3(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        return {"preds": x}, None


class TestWrapper2(ModelWrapper):
    def make_dataloader_train(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl


def my_accuracy(preds):
    return float(np.mean(preds))


class TestWrapperCustomEval(TestWrapper):
    def evaluation_functions(self):
        return {"mean": my_accuracy}


class TestWrapperCustomEvalUnderscore(TestWrapper):
    def evaluation_functions(self):
        return {"mean_score": my_accuracy}


def test_proto(tmp_path: Path, config, working_dir):
    wrapper = TestWrapper(MyCustomModel)

    config.experiment_dir = tmp_path.joinpath(f"{random.random()}")
    ablator = ProtoTrainer(wrapper=wrapper, run_config=config)
    metrics = ablator.launch(working_directory=working_dir)
    val_metrics = ablator.evaluate()
    assert np.isclose(metrics["val_loss"], val_metrics["val"]["loss"])


def test_proto_with_scheduler(tmp_path: Path, config, working_dir):
    wrapper = TestWrapper(MyCustomModel)
    config.experiment_dir = tmp_path.joinpath(f"{random.random()}")
    config.train_config.scheduler_config = SchedulerConfig(
        "step", arguments={"step_when": "val"}
    )
    ablator = ProtoTrainer(wrapper=wrapper, run_config=config)
    metrics = ablator.launch(working_directory=working_dir)
    val_metrics = ablator.evaluate()
    assert np.isclose(metrics["val_loss"], val_metrics["val"]["loss"])


def test_missing_val_dataloader(config: ParallelConfig, assert_error_msg):
    msg = assert_error_msg(lambda: TestWrapper2(MyCustomModel3).train(config))
    assert (
        msg
        == "optim_metric_name=`val_loss` not found in metrics ['best_iteration',"
        " 'best_val_loss', 'current_epoch', 'current_iteration', 'epochs',"
        " 'learning_rate', 'loss', 'total_steps']. Make sure your validation loader"
        " and validation loop are configured correctly."
    )
    metrics = TestWrapper(MyCustomModel3).train(config)
    assert metrics["current_iteration"] == 200


def test_custom_scheduler(config: ParallelConfig):
    # bypasses the `mode` check during initialization.

    kwargs = config.to_dict()
    kwargs["train_config"]["scheduler_config"] = MyPlateauConfig()
    kwargs["optim_metric_name"] = None
    kwargs["optim_metrics"] = None

    _config = MyParallelConfig(**kwargs)
    with pytest.raises(
        EvaluationError,
        match=(
            "A validation optimization argument is required with ReduceLROnPlateau"
            " scheduler. Try setting a `optim_metric_name`"
        ),
    ):
        metrics = TestWrapper(MyCustomModel3).train(_config)

    kwargs = config.to_dict()
    kwargs["train_config"]["scheduler_config"] = MyPlateauConfig()
    _config = MyParallelConfig(**kwargs)

    metrics = TestWrapper(MyCustomModel3).train(_config)
    assert metrics["current_iteration"] == 200


def test_invalid_optim_metrics(config: ParallelConfig):
    _config = copy.deepcopy(config)
    _config.optim_metrics = None
    with pytest.raises(
        ValueError,
        match=(
            "Invalid configuration. Must specify both `optim_metrics` and"
            " `optim_metric_name` or neither."
        ),
    ):
        metrics = TestWrapper(MyCustomModel3).train(_config)
    _config = copy.deepcopy(config)
    _config.optim_metric_name = None
    with pytest.raises(
        ValueError,
        match=(
            "Invalid configuration. Must specify both `optim_metrics` and"
            " `optim_metric_name` or neither."
        ),
    ):
        metrics = TestWrapper(MyCustomModel3).train(_config)


@pytest.mark.parametrize("scheduler_name", list(SCHEDULER_CONFIG_MAP.keys()) + [None])
def test_val_scheduler(config: ParallelConfig, scheduler_name, assert_error_msg):
    arguments = {"step_when": "val"}
    if scheduler_name in scheduler_args:
        arguments.update(scheduler_args[scheduler_name])
    config.train_config.scheduler_config = (
        SchedulerConfig(scheduler_name, arguments=arguments)
        if scheduler_name is not None
        else None
    )
    if scheduler_name == "plateau":
        _config = copy.deepcopy(config)
        _config.optim_metric_name = None
        _config.optim_metrics = None
        msg = assert_error_msg(lambda: TestWrapper2(MyCustomModel3).train(_config))
        assert msg == "Must provide `optim_metrics` when using Scheduler = `plateau`."
        metrics = TestWrapper(MyCustomModel3).train(config)
        assert metrics["current_iteration"] == 200

    else:
        _config = copy.deepcopy(config)
        _config.optim_metric_name = None
        _config.optim_metrics = None
        metrics = TestWrapper2(MyCustomModel3).train(_config)
        assert metrics["current_iteration"] == 200


@pytest.mark.parametrize("trainer_class", [ProtoTrainer])
def test_experiment_dir(
    tmp_path: Path,
    config,
    assert_error_msg,
    trainer_class,
    working_dir,
):
    config.experiment_dir = tmp_path.joinpath(f"{random.random()}")
    ablator = trainer_class(wrapper=TestWrapper(MyCustomModel3), run_config=config)
    ablator.launch(working_directory=working_dir)

    experiminet_dir = tmp_path.joinpath(f"{random.random()}")
    assert not experiminet_dir.exists()
    os.chdir(tmp_path)
    config.experiment_dir = f"{random.random()}"

    ablator = trainer_class(wrapper=TestWrapper(MyCustomModel3), run_config=config)
    ablator.launch(working_directory=working_dir)
    assert ablator.experiment_dir == tmp_path.joinpath(config.experiment_dir)
    assert ablator.experiment_dir.exists()
    config.experiment_dir = None
    assert_error_msg(
        lambda: trainer_class(wrapper=TestWrapper(MyCustomModel3), run_config=config),
        "Must specify an experiment directory.",
    )


def test_git_diffs(
    tmp_path: Path,
    config,
):
    config.experiment_dir = tmp_path.joinpath(f"{random.random()}")
    ablator = ProtoTrainer(wrapper=TestWrapper(MyCustomModel3), run_config=config)
    repo_path = tmp_path.joinpath("repo_path")
    remote_path = tmp_path.joinpath("remote_path.git")
    with pytest.raises(
        FileNotFoundError, match=f"Directory {repo_path} was not found. "
    ):
        msg = ablator._get_diffs(repo_path)
    os.mkdir(repo_path)

    msg = ablator._get_diffs(repo_path)
    assert (
        msg
        == f"No git repository was detected at {repo_path}. We recommend setting the"
        " working directory to a git repository to keep track of changes."
    )
    remote_repo = git.Repo.init(
        remote_path,
        bare=True,
    )

    repo = git.Repo.init(
        repo_path,
    )
    with pytest.raises(RuntimeError, match=".*Reference at .* does not exist"):
        msg = ablator._get_diffs(repo.working_dir)

    repo.create_remote("origin", url=remote_repo.working_dir)
    file_name = os.path.join(repo.working_dir, "mock.txt")
    open(file_name, "wb").close()
    repo.index.add([file_name])
    repo.index.commit("initial commit")
    repo.create_head("master")
    repo.remote("origin").push("master")

    msg = ablator._get_diffs(repo.working_dir)
    diffs = msg.split("\n")
    assert diffs[0] == f"Git Diffs for {repo.head.ref} @ {repo.head.commit}: "
    assert diffs[1] == ""
    Path(file_name).write_text("\n".join(str(i) for i in range(100)) + "\n")
    msg = ablator._get_diffs(repo.working_dir)
    diffs = msg.split("\n")
    assert all(diffs[-(i + 1)] == f"+{99-i}" for i in range(100))

    repo.index.add([file_name])
    repo.index.commit("second commit")
    repo.remote("origin").push("master")
    Path(file_name).write_text("\n".join(str(i) for i in range(100)) + "\n")
    msg = ablator._get_diffs(repo.working_dir)
    diffs = msg.split("\n")
    assert diffs[1] == ""

    Path(file_name).write_text("\n".join(str(i) for i in range(50)) + "\n")
    msg = ablator._get_diffs(repo_path)
    diffs = msg.split("\n")
    assert all(diffs[-(i + 1)] == f"-{99-i}" for i in range(50))


def test_smoke_tests(tmp_path: Path):
    optimizer_config = OptimizerConfig(name="sgd", arguments={"lr": 0.1})
    train_config = TrainConfig(
        dataset="test",
        batch_size=128,
        epochs=2,
        optimizer_config=optimizer_config,
        scheduler_config=None,
    )
    config = RunConfig(
        train_config=train_config,
        model_config=ModelConfig(),
        verbose="silent",
        device="cpu",
        amp=False,
    )
    config.experiment_dir = tmp_path.joinpath("i")

    # Runs successfully.
    wrapper = TestWrapper(MyCustomModel)

    ablator = ProtoTrainer(wrapper=wrapper, run_config=config)
    assert not ablator.wrapper._is_init
    ablator.launch(".")
    assert ablator.wrapper._is_init
    config.train_config.batch_size = 512
    assert ablator.smoke_test(config)
    # does not corrupt the wrapper config.
    assert ablator.wrapper.run_config != config


def test_proto_custom_eval(tmp_path: Path, config):
    wrapper = TestWrapperCustomEval(MyCustomModel)
    config.experiment_dir = tmp_path.joinpath(f"{random.random()}")
    ablator = ProtoTrainer(wrapper=wrapper, run_config=config)
    train_metrics = ablator.launch(tmp_path)
    eval_metrics = ablator.evaluate()
    assert np.isclose(train_metrics["val_mean"], eval_metrics["val"]["mean"])
    wrapper = TestWrapperCustomEvalUnderscore(MyCustomModel)
    config.experiment_dir = tmp_path.joinpath(f"{random.random()}")
    ablator = ProtoTrainer(wrapper=wrapper, run_config=config)
    train_metrics = ablator.launch(tmp_path)
    eval_metrics = ablator.evaluate()
    assert np.isclose(
        train_metrics["val_mean_score"], eval_metrics["val"]["mean_score"]
    )


if __name__ == "__main__":
    from tests.conftest import run_tests_local

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]
    kwargs = {
        "scheduler_name": list(SCHEDULER_CONFIG_MAP.keys()) + [None],
        "trainer_class": [ProtoTrainer, ParallelTrainer],
        "config": _config,
    }
    run_tests_local(test_fns, kwargs)

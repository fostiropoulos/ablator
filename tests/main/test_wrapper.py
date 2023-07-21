import copy
import io
import typing
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import mock
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler

from ablator import (
    Derived,
    ModelConfig,
    ModelWrapper,
    OptimizerConfig,
    RunConfig,
    TrainConfig,
)
from ablator.utils.base import Dummy

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


amp_config = RunConfig(
    train_config=train_config,
    model_config=ModelConfig(),
    verbose="silent",
    device="cuda",
    amp=True,
)


class BadMyModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        return x.sum().abs()


class MyModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        return {"preds": x}, x.sum().abs()


class MyUnstableModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))
        self.iteration = 0

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        self.iteration += 1
        if self.iteration > 10:
            return {"preds": x}, x.sum().abs() + torch.tensor(float("inf"))

        return {"preds": x}, x.sum().abs()


class MyWrongCustomModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))
        self.iteration = 0

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        self.iteration += 1
        if self.iteration > 10:
            return {"preds": x}, None
        return {"preds": x}, x.sum().abs() * 1e-7


class MyCustomModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))
        self.iteration = 0

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        self.iteration += 1
        if self.iteration > 10:
            if self.training:
                x.sum().abs().backward()
            return {"preds": x}, None

        return {"preds": x}, x.sum().abs() * 1e-7


class MyReturnNoneModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100))
        self.iteration = 0

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        self.iteration += 1
        if self.iteration > 10:
            if self.training:
                x.sum().abs().backward()
            return {"preds": None}, None

        return {"preds": None}, x.sum().abs() * 1e-7


class MyBadModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))
        self.iteration = 0

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01

        return None, x.sum().abs() * 1e-7


class TestWrapper(ModelWrapper):
    def make_dataloader_train(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl

    def make_dataloader_val(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl


class DisambigiousTestWrapper(ModelWrapper):
    def make_dataloader_train(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl

    def make_dataloader_val(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl

    def config_parser(self, run_config: RunConfig):
        run_config.model_config.ambigious_var = 10
        return run_config


class AuxWrapper(ModelWrapper):
    def make_dataloader_train(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl

    def make_dataloader_val(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl

    def aux_metrics(
        self, output_dict: dict[str, torch.Tensor] | None
    ) -> dict[str, typing.Any] | None:
        return {"learning_rate": 0.1}


def test_error_models(assert_error_msg):
    assert_error_msg(
        lambda: TestWrapper(BadMyModel).train(config),
        "Model should return outputs: dict[str, torch.Tensor] | None, loss: torch.Tensor | None.",
    )
    assert_error_msg(
        lambda: TestWrapper(MyUnstableModel).train(config),
        "Loss Diverged. Terminating. loss: inf",
    )
    # TODO find how to address the model not doing backward
    # assert_error_msg(
    #     lambda: TestWrapper(MyWrongCustomModel).train(amp_config),
    #     "No inf checks were recorded for this optimizer.",
    # )


def assert_console_output(fn, assert_fn):
    f = io.StringIO()
    with redirect_stdout(f):
        fn()
    s = f.getvalue()
    assert assert_fn(s)


def capture_output(fn):
    out = io.StringIO()

    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        fn()

    return out.getvalue(), err.getvalue()


class DummyScreen(Dummy):
    def addstr(self, *args, **kwargs):
        print(args[2])

    def getmaxyx(self):
        return 100, 100


def test_verbosity():
    verbose_config = RunConfig(
        train_config=train_config,
        model_config=ModelConfig(),
        verbose="progress",
        metrics_n_batches=100,
        device="cpu",
        amp=False,
    )

    with mock.patch("curses.initscr", DummyScreen), mock.patch(
        "ablator.utils.progress_bar.Display.close", lambda self: None
    ):
        out, err = capture_output(
            lambda: TestWrapper(MyCustomModel).train(verbose_config, debug=True)
        )
        assert (
            any(
                [
                    out.strip().split("\n")[i].endswith("?it/s, Remaining: ??]")
                    for i in range(5)
                ]
            )
            and len(err) == 0
        )
        verbose_config = RunConfig(
            train_config=train_config,
            model_config=ModelConfig(),
            verbose="progress",
            metrics_n_batches=32,
            device="cpu",
            amp=False,
        )
        out, err = capture_output(
            lambda: TestWrapper(MyCustomModel).train(verbose_config, debug=True)
        )
        assert (
            "Metrics batch-limit 32 is larger than 20% of the train dataloader length 100. You might experience slow-down during training. Consider decreasing `metrics_n_batches`."
            in out
        )
        console_config = RunConfig(
            train_config=train_config,
            model_config=ModelConfig(),
            verbose="console",
            device="cpu",
            amp=False,
        )
        out, err = capture_output(
            lambda: TestWrapper(MyCustomModel).train(console_config, debug=True)
        )
        assert len(err) == 0 and out.endswith(
            "learning_rate: 0.100000 total_steps: 00000200\n"
        )


def test_train_stats():
    res = TestWrapper(MyCustomModel).train(config)
    assert res["train_loss"] < 2e-05
    del res["train_loss"]

    assert res == {
        "val_loss": np.nan,
        "best_iteration": 0,
        "best_loss": float("inf"),
        "current_epoch": 2,
        "current_iteration": 200,
        "epochs": 2,
        "learning_rate": 0.1,
        "total_steps": 200,
    }


def test_state(assert_error_msg):
    wrapper = TestWrapper(MyCustomModel)
    assert_error_msg(
        lambda: wrapper.train_stats,
        "Undefined train_dataloader.",
    )
    assert wrapper.current_state == {}

    class AmbigiousModelConfig(ModelConfig):
        ambigious_var: Derived[int]

    _config = RunConfig(
        train_config=train_config,
        model_config=AmbigiousModelConfig(),
        verbose="silent",
        device="cpu",
        amp=False,
        random_seed=100,
    )

    assert_error_msg(
        lambda: wrapper._init_state(run_config=_config),
        "Ambigious configuration. Must provide value for ambigious_var",
    )
    disambigious_wrapper = DisambigiousTestWrapper(MyCustomModel)
    disambigious_wrapper._init_state(run_config=_config)

    _config = copy.deepcopy(config)
    _config.random_seed = 100
    wrapper = TestWrapper(MyCustomModel)
    wrapper._init_state(run_config=_config)

    assert len(wrapper.train_dataloader) == 100 and len(wrapper.val_dataloader) == 100
    train_stats = {
        "learning_rate": float("inf"),
        "total_steps": 200,
        "epochs": 2,
        "current_epoch": 0,
        "current_iteration": 0,
        "best_iteration": 0,
        "best_loss": float("inf"),
    }
    assert dict(wrapper.train_stats) == train_stats
    assert (
        wrapper.current_state["run_config"] == _config.to_dict()
        and wrapper.current_state["train_metrics"]
        == {**train_stats, **{"loss": np.nan}}
        and wrapper.current_state["eval_metrics"] == {"loss": np.nan}
    )
    assert str(wrapper.model.param.device) == "cpu"
    assert wrapper.model.param.requires_grad == True
    assert wrapper.current_checkpoint is None
    assert wrapper.best_loss == float("inf")
    assert isinstance(wrapper.model, MyCustomModel)
    assert isinstance(wrapper.scaler, GradScaler)
    assert wrapper.scheduler is None
    assert wrapper.logger is not None
    assert wrapper.device == "cpu"
    assert wrapper.amp == False
    assert wrapper.random_seed == 100


def test_load_save_errors(tmp_path: Path, assert_error_msg):
    tmp_path = tmp_path.joinpath("test_exp")
    wrapper = TestWrapper(MyCustomModel)

    _config = copy.deepcopy(config)
    _config.verbose = "console"
    _config.experiment_dir = tmp_path

    msg = assert_error_msg(
        lambda: [
            wrapper._init_state(run_config=_config),
            wrapper._init_state(run_config=_config),
        ],
    )
    assert msg == f"SummaryLogger: Resume is set to False but {tmp_path} exists."

    assert wrapper._init_state(run_config=_config, debug=True) is None
    assert_error_msg(
        lambda: [wrapper._init_state(run_config=_config, resume=True)],
        f"Could not find a valid checkpoint in {tmp_path.joinpath('checkpoints')}",
    )

    pass


def test_load_save(tmp_path: Path, assert_error_msg):
    tmp_path = tmp_path.joinpath("test_exp")
    _config = copy.deepcopy(config)
    _config.verbose = "console"
    _config.experiment_dir = tmp_path
    wrapper = TestWrapper(MyCustomModel)

    wrapper.train(_config)
    old_stats = copy.deepcopy(wrapper.train_stats)
    old_model = copy.deepcopy(wrapper.model)
    wrapper = TestWrapper(MyCustomModel)

    wrapper._init_state(run_config=_config, resume=True)
    assert old_stats == wrapper.train_stats
    with mock.patch("ablator.ModelWrapper.epochs", return_value=3):
        wrapper._init_state(run_config=_config, resume=True)
        wrapper.epochs = 3
        assert_error_msg(
            lambda: wrapper.checkpoint(),
            f"Checkpoint iteration {wrapper.current_iteration} >= training iteration {wrapper.current_iteration}. Can not overwrite checkpoint.",
        )
        wrapper._inc_iter()
        wrapper.checkpoint()
        assert (
            wrapper.current_state["model"]["param"] == old_model.state_dict()["param"]
        ).all()


def test_train_loop(assert_error_msg):
    _config = copy.deepcopy(config)

    wrapper = TestWrapper(MyReturnNoneModel)
    wrapper._init_state(run_config=_config)
    assert_error_msg(
        lambda: wrapper.train_loop(),
        "Model should return outputs: dict[str, torch.Tensor] | None, loss: torch.Tensor | None.",
    )


def test_validation_loop():
    wrapper = AuxWrapper(MyBadModel)
    _config = copy.deepcopy(config)
    wrapper._init_state(_config)
    val_dataloder = wrapper.make_dataloader_val(_config)
    metrics_dict = wrapper.validation_loop(
        MyBadModel(_config),
        val_dataloder,
        wrapper.eval_metrics,
    )
    assert len(metrics_dict) == 1 and "loss" in metrics_dict.keys()


def test_train_resume(tmp_path: Path, assert_error_msg):
    tmp_path = tmp_path.joinpath("test_exp")
    _config = copy.deepcopy(config)
    _config.verbose = "console"
    _config.experiment_dir = tmp_path
    wrapper = TestWrapper(MyCustomModel)

    msg = assert_error_msg(
        lambda: [wrapper.train(_config, resume=True)],
    )
    assert (
        msg
        == f"Could not find a valid checkpoint in {wrapper.experiment_dir.joinpath('checkpoints')}"
    )


if __name__ == "__main__":
    import shutil
    from tests.conftest import _assert_error_msg

    tmp_path = Path("/tmp/")

    shutil.rmtree(tmp_path.joinpath("test_exp"), ignore_errors=True)
    test_load_save(tmp_path, _assert_error_msg)
    shutil.rmtree(tmp_path.joinpath("test_exp"), ignore_errors=True)
    test_load_save_errors(tmp_path, _assert_error_msg)
    test_error_models(_assert_error_msg)
    test_train_stats()
    test_state(_assert_error_msg)

    test_verbosity()
    test_train_resume(tmp_path, _assert_error_msg)
    test_train_loop(_assert_error_msg)
    test_validation_loop()

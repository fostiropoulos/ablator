import copy
from functools import cached_property
import re
import typing
from pathlib import Path
import pytest
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
from ablator.main.model.main import TrainPlateauError
from ablator.utils.base import Dummy

_train_config = TrainConfig(
    dataset="test",
    batch_size=128,
    epochs=2,
    optimizer_config=OptimizerConfig(name="sgd", arguments={"lr": 0.1}),
    scheduler_config=None,
)

_config = RunConfig(
    train_config=_train_config,
    model_config=ModelConfig(),
    verbose="silent",
    device="cpu",
    amp=False,
)


@pytest.fixture()
def config():
    return copy.deepcopy(_config)


@pytest.fixture()
def train_config():
    return copy.deepcopy(_train_config)


amp_config = RunConfig(
    train_config=_train_config,
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


class MyModelInitWeights(MyModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.randn(100, 1))

    def init_weights(self):
        self.param.data = torch.ones_like(self.param)


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


class DummyScreen(Dummy):
    def addstr(self, *args, **kwargs):
        print(args[2])

    def getmaxyx(self):
        return 100, 100


def test_error_models(assert_error_msg, config: RunConfig):
    assert_error_msg(
        lambda: TestWrapper(BadMyModel).train(config),
        (
            "Model should return outputs: dict[str, torch.Tensor] | None, loss:"
            " torch.Tensor | None."
        ),
    )
    assert_error_msg(
        lambda: TestWrapper(MyUnstableModel).train(config),
        "Loss Diverged. Terminating. loss: inf",
    )


def test_verbosity(capture_output, train_config):
    verbose_config = RunConfig(
        train_config=train_config,
        model_config=ModelConfig(),
        verbose="progress",
        metrics_n_batches=100,
        device="cpu",
        amp=False,
    )

    with (
        mock.patch("curses.initscr", DummyScreen),
        mock.patch("ablator.utils.progress_bar.Display.close", lambda self: None),
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
            "Metrics batch-limit 32 is larger than 20% of the train dataloader length"
            " 100. You might experience slow-down during training. Consider decreasing"
            " `metrics_n_batches`."
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


def test_state(
    assert_error_msg,
    train_config,
):
    wrapper = TestWrapper(MyCustomModel)
    msg = assert_error_msg(lambda: wrapper.train_stats)

    assert (
        msg
        == "Can not read property train_stats of unitialized TestWrapper. It must be"
        " initialized with `init_state` before using."
    )

    class AmbigiousModelConfig(ModelConfig):
        ambigious_var: Derived[int]

    config = RunConfig(
        train_config=train_config,
        model_config=AmbigiousModelConfig(),
        verbose="silent",
        device="cpu",
        amp=False,
        random_seed=100,
    )

    assert_error_msg(
        lambda: wrapper.init_state(run_config=config),
        (
            "Ambiguous configuration `AmbigiousModelConfig`. Must provide value for"
            " ambigious_var"
        ),
    )
    disambigious_wrapper = DisambigiousTestWrapper(MyCustomModel)
    disambigious_wrapper.init_state(run_config=config)


def test_train_stats(config: RunConfig):
    config.random_seed = 100
    config.optim_metric_name = "val_loss"
    config.optim_metrics = {"val_loss": "max"}

    wrapper = TestWrapper(MyCustomModel)
    wrapper.init_state(run_config=config)

    assert len(wrapper.train_dataloader) == 100 and len(wrapper.val_dataloader) == 100
    train_stats = {
        "learning_rate": float("inf"),
        "total_steps": 200,
        "epochs": 2,
        "current_epoch": 0,
        "current_iteration": 0,
        "best_iteration": None,
        "best_val_loss": float("-inf"),
    }
    assert dict(wrapper.train_stats) == train_stats

    config.optim_metrics = {"val_loss": "min"}
    train_stats["best_val_loss"] = float("inf")
    wrapper = TestWrapper(MyCustomModel)
    wrapper.init_state(run_config=config)

    assert dict(wrapper.train_stats) == train_stats

    assert (
        wrapper.current_state["run_config"] == config.to_dict()
        and wrapper.current_state["train_metrics"]
        == {**train_stats, **{"loss": np.nan}}
        and wrapper.current_state["eval_metrics"] == {"loss": np.nan}
    )
    assert str(wrapper.model.param.device) == "cpu"
    assert wrapper.model.param.requires_grad
    assert wrapper.current_checkpoint is None
    assert wrapper.best_metrics["val_loss"] == float("inf")
    assert isinstance(wrapper.model, MyCustomModel)
    assert isinstance(wrapper.scaler, GradScaler)
    assert wrapper.scheduler is None
    assert wrapper.logger is not None
    assert wrapper.device == "cpu"
    assert not wrapper.amp
    assert wrapper.random_seed == 100

    res = wrapper.train()
    assert res["train_loss"] < 2e-05
    del res["train_loss"]
    assert np.isnan(res["val_loss"])
    del res["val_loss"]

    assert res == {
        "best_iteration": None,
        "best_val_loss": float("inf"),
        "current_epoch": 2,
        "current_iteration": 200,
        "epochs": 2,
        "learning_rate": 0.1,
        "total_steps": 200,
    }


def test_load_save_errors(tmp_path: Path, assert_error_msg, config: RunConfig):
    tmp_path = tmp_path.joinpath("test_exp")
    wrapper = TestWrapper(MyCustomModel)

    config.verbose = "console"
    config.experiment_dir = tmp_path

    def _run_two_wrappers():
        wrapper = TestWrapper(MyCustomModel)
        wrapper.init_state(run_config=config)
        wrapper = TestWrapper(MyCustomModel)
        wrapper.init_state(run_config=config)

    msg = assert_error_msg(_run_two_wrappers)
    assert msg == f"SummaryLogger: Resume is set to False but {tmp_path} is not empty."

    assert wrapper.init_state(run_config=config, debug=True) is None
    assert_error_msg(
        lambda: [wrapper.init_state(run_config=config, resume=True)],
        f"Could not find a valid checkpoint in {tmp_path.joinpath('checkpoints')}",
    )


def test_load_save(tmp_path: Path, assert_error_msg, config: RunConfig):
    tmp_path = tmp_path.joinpath("test_exp")
    config.verbose = "console"
    config.experiment_dir = tmp_path
    wrapper = TestWrapper(MyCustomModel)

    wrapper.train(config)
    old_stats = copy.deepcopy(wrapper.train_stats)
    old_model = copy.deepcopy(wrapper.model)
    wrapper = TestWrapper(MyCustomModel)

    wrapper.init_state(run_config=config, resume=True)
    assert old_stats == wrapper.train_stats

    ModelWrapper.epochs = 0
    with mock.patch("ablator.ModelWrapper.epochs", return_value=3):
        wrapper.init_state(run_config=config, resume=True)
        assert_error_msg(
            lambda: wrapper.checkpoint(),
            (
                f"Checkpoint iteration {wrapper.current_iteration} >= training"
                f" iteration {wrapper.current_iteration}. Can not overwrite checkpoint."
            ),
        )
        wrapper._inc_iter()
        wrapper.checkpoint()
        assert (
            wrapper.current_state["model"]["param"] == old_model.state_dict()["param"]
        ).all()


def test_train_loop(assert_error_msg, config):
    wrapper = TestWrapper(MyReturnNoneModel)
    wrapper.init_state(run_config=config)
    assert_error_msg(
        lambda: wrapper.train_loop(),
        (
            "Model should return outputs: dict[str, torch.Tensor] | None, loss:"
            " torch.Tensor | None."
        ),
    )


def test_validation_loop(config: RunConfig):
    wrapper = AuxWrapper(MyBadModel)

    wrapper.init_state(config)
    val_dataloder = wrapper.make_dataloader_val(config)
    metrics_dict = wrapper.validation_loop(
        MyBadModel(config),
        val_dataloder,
        wrapper.eval_metrics,
    )
    assert len(metrics_dict) == 1 and "loss" in metrics_dict.keys()


def test_train_resume(tmp_path: Path, assert_error_msg, config: RunConfig):
    tmp_path = tmp_path.joinpath("test_exp")
    config.verbose = "console"
    config.experiment_dir = tmp_path
    wrapper = TestWrapper(MyCustomModel)

    msg = assert_error_msg(
        lambda: [wrapper.train(config, resume=True)],
    )
    assert (
        msg
        == "Could not find a valid checkpoint in"
        f" {wrapper.experiment_dir.joinpath('checkpoints')}"
    )


def test_create_model(tmp_path: Path, config: RunConfig):
    tmp_path = tmp_path.joinpath("test_exp")
    config.verbose = "console"
    config.experiment_dir = tmp_path
    wrapper = TestWrapper(MyModelInitWeights)
    wrapper.init_state(copy.deepcopy(config))
    wrapper.create_model()
    assert (wrapper.model.param == 1).all()
    rand_weights = torch.randn_like(wrapper.model.param)
    wrapper.model.param.data = rand_weights
    save_dict = wrapper.save_dict()
    del wrapper

    config.experiment_dir = tmp_path.joinpath("test_exp_2")
    wrapper = TestWrapper(MyModelInitWeights)
    wrapper.init_state(copy.deepcopy(config))
    wrapper.create_model(save_dict)
    assert (wrapper.model.param == rand_weights).all().item()
    wrapper.create_model()
    assert (wrapper.model.param == 1).all()
    del MyModelInitWeights.init_weights
    wrapper.create_model()
    new_param = wrapper.model.param
    assert not (new_param == 1).all()
    MyModelInitWeights.init_weights = ""
    wrapper.create_model()
    assert (wrapper.model.param != new_param).all()


def test_early_stopping(tmp_path: Path, config: RunConfig):
    exp_dir = tmp_path.joinpath("test_exp")
    config.verbose = "console"
    config.experiment_dir = exp_dir

    config.early_stopping_iter = 1
    with pytest.raises(
        ValueError, match="Must provide `optim_metrics` when using early_stopping_iter"
    ):
        wrapper = TestWrapper(MyCustomModel)
        wrapper.train(config)

    exp_dir = tmp_path.joinpath("test_exp_2")
    config.optim_metric_name = "train_loss"
    config.experiment_dir = exp_dir
    config.train_config.epochs = 4
    config.optim_metrics = {"train_loss": "min"}
    with pytest.raises(
        TrainPlateauError,
        match=re.escape(
            "Early stopping. No improvement for 100 > early_stopping_iter = `1`"
            " iterations."
        ),
    ):
        wrapper = TestWrapper(MyCustomModel)
        wrapper.train(config)


def test_cached_properties(tmp_path: Path, config: RunConfig):
    exp_dir = tmp_path.joinpath("test_exp")
    config.verbose = "console"
    config.experiment_dir = exp_dir

    wrapper = TestWrapper(MyCustomModel)
    wrapper.init_state(config)
    wrapper.train()
    attrs = sorted(
        [
            a
            for a in dir(wrapper)
            if isinstance(getattr(type(wrapper), a, type(wrapper)), cached_property)
        ]
    )
    assert attrs == sorted(wrapper._cached_properties)
    config.eval_epoch = 0.1
    config.log_epoch = 0.5
    assert "eval_itr" in wrapper._cached_properties
    assert "log_itr" in wrapper._cached_properties
    eval_itr = wrapper.eval_itr
    log_itr = wrapper.log_itr
    with mock.patch(
        "ablator.main.model.main.ModelBase._reset_cached_attributes",
        lambda *args, **kwargs: True,
    ):
        wrapper.init_state(config, resume=True)

    assert wrapper.eval_itr == eval_itr
    assert wrapper.log_itr == log_itr

    wrapper.init_state(config, resume=True)
    assert wrapper.eval_itr == eval_itr * 0.1
    assert wrapper.log_itr == log_itr * 0.5


def test_derived_stats_names(tmp_path: Path, config: RunConfig):
    exp_dir = tmp_path.joinpath("test_exp")
    config.verbose = "console"
    config.experiment_dir = exp_dir

    wrapper = TestWrapper(MyCustomModel)
    wrapper.init_state(config)
    assert all(
        attr in wrapper._derived_stats_names
        for attr in wrapper._overridable_stats_names
    )
    for attr in wrapper._derived_stats_names:
        if attr in wrapper._overridable_stats_names:
            setattr(wrapper, attr, None)
            continue

        with pytest.raises(
            RuntimeError, match=f"Can not set derived attribute {attr}."
        ):
            setattr(wrapper, attr, None)


if __name__ == "__main__":
    from tests.conftest import run_tests_local

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]

    kwargs = {
        "config": copy.deepcopy(_config),
        "train_config": copy.deepcopy(_train_config),
    }
    run_tests_local(test_fns, kwargs)

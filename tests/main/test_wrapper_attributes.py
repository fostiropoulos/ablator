import copy
import re
import shutil
from functools import partial
from pathlib import Path

import mock
import pytest
import torch
from sklearn.metrics import accuracy_score
from torch import nn

from ablator import ModelConfig, ModelWrapper, OptimizerConfig, RunConfig, TrainConfig
from ablator.modules.metrics.main import LossDivergedError, Metrics
from ablator.modules.scheduler import SchedulerConfig
from ablator.utils.base import Lock

_train_config = TrainConfig(
    dataset="test",
    batch_size=10,
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
def config(tmp_path: Path):
    _config.experiment_dir = tmp_path.joinpath("test_exp")
    return copy.deepcopy(_config)


@pytest.fixture()
def wrapper():
    wrapper = DeterminiticWrapper(MyDeterministicModel)
    return copy.deepcopy(wrapper)


# Custom accuracy function
def my_accuracy(y_true, y_pred):
    return accuracy_score(y_true.flatten(), y_pred.flatten())


class DeterminiticWrapper(ModelWrapper):
    def make_dataloader_train(self, run_config: RunConfig):
        g = torch.Generator()
        g.manual_seed(0)

        dl = [torch.rand(100, generator=g) for i in range(100)]
        return dl

    def make_dataloader_val(self, run_config: RunConfig):
        g = torch.Generator()
        g.manual_seed(1)
        dl = [torch.rand(100, generator=g) for i in range(100)]
        return dl

    def evaluation_functions(self):
        return {"accuracy_score": my_accuracy}


class MyDeterministicModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))

    def forward(self, x: torch.Tensor):
        if self.training:
            x = self.param + torch.rand_like(self.param) * 0.01
        else:
            x = x[:, None]
        return {
            "y_pred": torch.randint(0, 2, x.shape, generator=torch.Generator()),
            "y_true": torch.zeros_like(x),
        }, x.sum().abs()


def test_scheduler(tmp_path: Path, config: RunConfig):
    tmp_path = tmp_path.joinpath("test_exp")
    config.verbose = "console"
    config.experiment_dir = tmp_path
    wrapper = DeterminiticWrapper(MyDeterministicModel)
    wrapper.init_state(config, debug=True)
    assert wrapper._scheduler_step_when is None

    # EPOCH SCHEDULER
    config.train_config.scheduler_config = SchedulerConfig("step", arguments={})
    wrapper.init_state(config, debug=True)
    assert wrapper._scheduler_step_when == "epoch"
    step_count = wrapper.scheduler._step_count
    wrapper.scheduler_step()
    assert wrapper.scheduler._step_count == step_count
    for i in range(wrapper.epoch_len):
        wrapper._inc_iter()
    wrapper.scheduler_step()
    assert wrapper.scheduler._step_count == step_count + 1

    # VAL SCHEDULER
    config.train_config.scheduler_config = SchedulerConfig("plateau", arguments={})
    config.optim_metric_name = "val_loss"
    config.optim_metrics = {"val_loss": "min"}
    wrapper.init_state(config, debug=True)
    assert wrapper._scheduler_step_when == "val"

    step_count = wrapper.scheduler.last_epoch
    wrapper.scheduler_step()
    assert wrapper.scheduler.last_epoch == step_count
    wrapper.scheduler_step()
    assert wrapper.scheduler.last_epoch == step_count
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ReduceLROnPlateau.step() missing 1 required positional argument: 'metrics'"
        ),
    ):
        wrapper.scheduler_step(is_val_step=True)
    wrapper.scheduler_step(0.01, is_val_step=True)
    assert wrapper.scheduler.last_epoch == step_count + 1
    wrapper.scheduler_step(0.01, is_val_step=True)
    assert wrapper.scheduler.last_epoch == step_count + 2

    # TRAIN SCHEDULER
    config.train_config.scheduler_config = SchedulerConfig(
        "cycle", arguments={"max_lr": 0.01, "total_steps": 100}
    )
    wrapper.init_state(config, debug=True)
    assert wrapper._scheduler_step_when == "train"
    step_count = wrapper.scheduler._step_count
    wrapper.scheduler_step()
    assert wrapper.scheduler._step_count == step_count + 1


def test_apply_loss(tmp_path: Path, config: RunConfig):
    tmp_path = tmp_path.joinpath("test_exp")
    config.experiment_dir = tmp_path
    wrapper = DeterminiticWrapper(MyDeterministicModel)
    wrapper.init_state(copy.deepcopy(config))
    model: MyDeterministicModel = wrapper.model
    optimizer = wrapper.optimizer
    scaler = wrapper.scaler
    # Ensure no left-over grads are in the model's parameters from custom evaluation or what-not
    optimizer.zero_grad(set_to_none=True)
    preds, loss = model(torch.zeros(100, 100))
    og_param = model.param.clone()
    assert model.param.grad is None
    loss_value = wrapper.apply_loss(model, loss, optimizer, scaler)
    assert model.param.grad is None
    assert loss_value == loss.item()
    new_param = model.param.clone()
    assert torch.isclose(new_param, og_param * 0.9).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The test is meant to evaluate amp which is available only with cuda.",
)
def test_optim_step(tmp_path: Path, config: RunConfig, assert_error_msg):
    tmp_path = tmp_path.joinpath("test_exp")
    config.experiment_dir = tmp_path
    wrapper = DeterminiticWrapper(MyDeterministicModel)
    config.device = "cuda"
    config.amp = True
    wrapper.init_state(copy.deepcopy(config))
    assert wrapper.amp
    model: MyDeterministicModel = wrapper.model
    optimizer = wrapper.optimizer
    scaler = wrapper.scaler
    optimizer.zero_grad(set_to_none=True)

    og_param = model.param.clone()
    preds, loss = model(torch.zeros(100, 100))

    msg = assert_error_msg(lambda: wrapper.optim_step(optimizer, scaler, model, loss))
    assert (
        msg
        == "Attempted unscale_ but _scale is None.  This may indicate your script did"
        " not use scaler.scale(loss or outputs) earlier in the iteration."
    )
    wrapper.backward(loss, scaler)
    wrapper.optim_step(optimizer, scaler, model, loss)
    new_param = model.param.clone()
    assert torch.isclose(new_param, og_param * 0.98).all()


def test_evaluate(tmp_path: Path, config: RunConfig):
    tmp_path = tmp_path.joinpath("test_exp")
    config.experiment_dir = tmp_path
    config.train_config.batch_size = 10
    wrapper = DeterminiticWrapper(MyDeterministicModel)
    metrics = wrapper.train(config)
    evaluate_metrics = wrapper.evaluate(config)
    assert evaluate_metrics["val"]["loss"] == metrics["val_loss"]
    config.eval_subsample = 0.1
    shutil.rmtree(tmp_path, ignore_errors=True)
    wrapper = DeterminiticWrapper(MyDeterministicModel)
    metrics = wrapper.train(config)
    evaluate_metrics = wrapper.evaluate(config)
    assert evaluate_metrics["val"]["loss"] != metrics["val_loss"]


def test_metrics(tmp_path: Path, wrapper: DeterminiticWrapper, config: RunConfig):
    tmp_path = tmp_path.joinpath("test_metrics")
    config.experiment_dir = tmp_path
    metrics = wrapper.train(config)
    assert metrics == wrapper.metrics
    train_metrics = wrapper.train_metrics.to_dict()
    eval_metrics = wrapper.eval_metrics.to_dict()
    for k, v in eval_metrics.items():
        k = f"val_{k}"
        assert k in metrics
        assert metrics[k] == v
        del metrics[k]
    for k, v in train_metrics.items():
        k = f"train_{k}" if k in eval_metrics else k
        assert k in metrics
        assert metrics[k] == v
        del metrics[k]
    assert len(metrics) == 0
    wrapper.train_metrics.update_ma_metrics({"loss": float("-inf")})
    assert wrapper.metrics["train_loss"] == float("-inf")


def test_validation_loop(
    tmp_path: Path, wrapper: DeterminiticWrapper, config: RunConfig, capture_output
):
    tmp_path = tmp_path.joinpath("test_metrics")
    config.experiment_dir = tmp_path
    wrapper.init_state(config)

    stdout, stderr = capture_output(
        lambda: wrapper.validation_loop(
            wrapper.model, wrapper.val_dataloader, wrapper.eval_metrics
        )
    )
    assert (
        "Called `validation_loop` without setting the model to evaluation mode. i.e."
        " `model.eval()`"
        in stdout
    )
    wrapper.model.eval()
    stdout, stderr = capture_output(
        lambda: wrapper.validation_loop(
            wrapper.model, wrapper.val_dataloader, wrapper.eval_metrics
        )
    )
    assert len(stdout) == 0 and len(stderr) == 0
    metrics = wrapper.validation_loop(
        wrapper.model, wrapper.val_dataloader, wrapper.eval_metrics
    )
    assert metrics == wrapper.eval_metrics.to_dict()


def test_config_parser(
    tmp_path: Path,
    wrapper: DeterminiticWrapper,
    config: RunConfig,
    assert_error_msg,
    capture_output,
):
    tmp_path = tmp_path.joinpath("test_config_parser")
    config.experiment_dir = tmp_path
    config.optim_metrics = None
    config.train_config.scheduler_config = SchedulerConfig("plateau", arguments={})
    msg = assert_error_msg(lambda: wrapper.init_state(config))
    assert msg == "Must provide `optim_metrics` when using Scheduler = `plateau`."

    config.optim_metrics = {"loss": "max"}
    config.optim_metric_name = "loss"
    stdout, stderr = capture_output(lambda: wrapper.init_state(config, debug=True))
    assert (
        "Different optim_metric_direction max than scheduler.arguments.mode min."
        " Overwriting scheduler.arguments.mode."
        in stdout
    )


def test_config_parser_plateau(
    tmp_path: Path, wrapper: DeterminiticWrapper, config: RunConfig, capture_output
):
    tmp_path = tmp_path.joinpath("test_config_parser_plateau")
    config.experiment_dir = tmp_path
    config.train_config.scheduler_config = SchedulerConfig("plateau", arguments={})
    config.optim_metrics = {"loss": "min"}
    config.optim_metric_name = "loss"
    stdout, stderr = capture_output(lambda: wrapper.init_state(config))

    assert (
        "Different optim_metric_direction max than scheduler.arguments.mode min."
        " Overwriting scheduler.arguments.mode."
        not in stdout
    )


def test_wrapper_is_init(
    tmp_path: Path,
    wrapper: DeterminiticWrapper,
    config: RunConfig,
    assert_error_msg,
    capture_output,
):
    tmp_path = tmp_path.joinpath("test_wrapper_is_init")
    config.experiment_dir = tmp_path
    assert not wrapper._is_init
    msg = assert_error_msg(lambda: wrapper.apply_loss)
    error_msg = (
        "Can not read property %s of unitialized DeterminiticWrapper. It must be"
        " initialized with `init_state` before using."
    )
    assert msg == (error_msg % "apply_loss")

    msg = assert_error_msg(lambda: wrapper.current_iteration)
    assert msg == (error_msg % "current_iteration")

    msg = assert_error_msg(lambda: wrapper.scheduler_step())
    assert msg == (error_msg % "scheduler_step")
    wrapper.init_state(config)
    assert wrapper.current_iteration == 0

    msg = assert_error_msg(lambda: wrapper.train(config))
    assert (
        msg
        == "Can not provide `run_config` to already initialized `DeterminiticWrapper`"
    )
    wrapper.train()
    msg = assert_error_msg(lambda: wrapper.init_state(config))
    assert msg == "DeterminiticWrapper is already initialized. "
    wrapper._is_init = False

    msg = assert_error_msg(lambda: wrapper.scheduler_step())
    assert msg == (error_msg % "scheduler_step")

    metrics_a = wrapper.evaluate(config)
    metrics_b = wrapper.evaluate(config)
    assert metrics_a == metrics_b

    stdout, stderr = capture_output(lambda: wrapper.train())
    assert (
        "Training is already complete: 200 / 200. Returning current metrics." in stdout
    )

    def _fn():
        wrapper.total_steps = None

    msg = assert_error_msg(_fn)
    assert msg == "Can not set derived attribute total_steps."
    metrics = wrapper.train()
    assert metrics["current_epoch"] == 2
    wrapper.epochs = 10
    metrics = wrapper.train()
    assert metrics["current_epoch"] == 10


def test_init_state(
    tmp_path: Path,
    wrapper: DeterminiticWrapper,
    config: RunConfig,
    assert_error_msg,
    capture_output,
):
    tmp_path = tmp_path.joinpath("test_init_state")
    config.experiment_dir = tmp_path
    stdout, stderr = capture_output(lambda: wrapper.init_state(config, smoke_test=True))
    assert len(stdout) == 0 and len(stderr) == 0
    stdout, stderr = capture_output(lambda: wrapper.init_state(config, debug=True))
    assert len(stderr) == 0
    assert (
        "If saving artifacts is unnecessary you can disable the file system by setting"
        " `run_config.experiment_dir=None`"
        in stdout
    )
    msg = assert_error_msg(lambda: wrapper.init_state(config, resume=True))
    assert "Could not find a valid checkpoint in " in msg
    wrapper.init_state(config, smoke_test=True)
    wrapper.train()
    assert (
        "If saving artifacts is unnecessary you can disable the file system by setting"
        " `run_config.experiment_dir=None`"
        in stdout
    )
    msg = assert_error_msg(lambda: wrapper.init_state(config, resume=True))

    wrapper.init_state(config, debug=True)
    wrapper.train()
    assert (
        "If saving artifacts is unnecessary you can disable the file system by setting"
        " `run_config.experiment_dir=None`"
        in stdout
    )
    msg = assert_error_msg(lambda: wrapper.init_state(config, resume=True))
    del wrapper.logger

    wrapper.init_state(config, debug=True)

    stdout, stderr = capture_output(lambda: wrapper.train(debug=True))
    assert (
        "Training is already complete: 200 / 200. Returning current metrics."
        not in stdout
    )
    msg = assert_error_msg(lambda: wrapper.init_state(config))
    assert msg == "DeterminiticWrapper is already initialized. "
    shutil.rmtree(config.experiment_dir)
    wrapper.init_state(config, smoke_test=True)
    wrapper.init_state(config)
    wrapper.train()
    wrapper.init_state(config, resume=True)
    stdout, stderr = capture_output(lambda: wrapper.train())
    assert (
        "Training is already complete: 200 / 200. Returning current metrics." in stdout
    )
    wrapper.init_state(config, smoke_test=True)
    stdout, stderr = capture_output(lambda: wrapper.train())
    assert (
        "Training is already complete: 200 / 200. Returning current metrics."
        not in stdout
    )
    wrapper.init_state(config, debug=True)
    stdout, stderr = capture_output(lambda: wrapper.train())
    assert (
        "Training is already complete: 200 / 200. Returning current metrics."
        not in stdout
    )
    stdout, stderr = capture_output(lambda: wrapper.train())
    assert (
        "Training is already complete: 200 / 200. Returning current metrics." in stdout
    )
    stdout, stderr = capture_output(lambda: wrapper.train(debug=True))
    assert (
        "Training is already complete: 200 / 200. Returning current metrics."
        not in stdout
    )


def test_dataloader_data_lock(
    tmp_path: Path,
    wrapper: DeterminiticWrapper,
    config: RunConfig,
):
    tmp_path = tmp_path.joinpath("test_dataloader_data_lock")
    config.experiment_dir = tmp_path
    f = Lock(timeout=1)

    wrapper.init_state(config, smoke_test=True, data_lock=f)

    f.acquire()
    with pytest.raises(TimeoutError):
        wrapper.init_state(config, smoke_test=True, data_lock=f)


@pytest.mark.parametrize("direction", ["min", "max"])
def test_optim_metric_names(
    tmp_path: Path, wrapper: DeterminiticWrapper, config: RunConfig, direction
):
    tmp_path = tmp_path.joinpath("test_init_state")
    config.experiment_dir = tmp_path
    sign = -1 if direction == "min" else 1
    _metric = "loss"
    metric_name = f"val_{_metric}"
    config.optim_metrics = {metric_name: direction}
    config.optim_metric_name = metric_name
    config.warm_up_epochs = -1
    wrapper.init_state(config, smoke_test=True)

    assert wrapper.best_metrics[metric_name] == float("-inf") * sign
    wrapper._train_evaluation_step()
    assert wrapper.best_metrics[metric_name] > float("-inf")
    wrapper.best_metrics[metric_name] += 100 * sign
    current_loss = wrapper.best_metrics[metric_name]
    wrapper._train_evaluation_step()
    assert wrapper.best_metrics[metric_name] == current_loss
    wrapper.best_metrics[metric_name] += 1000000 * sign
    with pytest.raises(LossDivergedError):
        wrapper._train_evaluation_step()

    wrapper.best_metrics[metric_name] -= 1000000 * sign

    wrapper._train_evaluation_step()

    current_loss = wrapper.best_metrics[metric_name]

    wrapper.best_metrics[metric_name] = 0.001

    def _mock_validation_loop(
        *args,
        metrics: Metrics,
        _update_metrics={},
        **kwargs,
    ):
        metrics.update_ma_metrics(_update_metrics)
        metrics.reset(reset_ma=True)

    with mock.patch(
        "ablator.ModelWrapper._validation_loop",
        partial(_mock_validation_loop, _update_metrics={_metric: 0.01}),
    ):
        wrapper._train_evaluation_step()

    with mock.patch(
        "ablator.ModelWrapper._validation_loop",
        partial(_mock_validation_loop, _update_metrics={_metric: 1.0 * sign}),
    ):
        wrapper.best_metrics[metric_name] = 100 * sign

        with pytest.raises(LossDivergedError):
            wrapper._train_evaluation_step()

        wrapper.best_metrics[metric_name] = 10 * sign
        wrapper._train_evaluation_step()

    with mock.patch(
        "ablator.ModelWrapper._validation_loop",
        partial(_mock_validation_loop, _update_metrics={_metric: -1.0 * sign}),
    ):
        wrapper.best_metrics[metric_name] = 10 * sign
        wrapper._train_evaluation_step()


def test_warmup_epochs(wrapper: DeterminiticWrapper, config: RunConfig):
    config.optim_metrics = {"val_loss": "min"}
    config.optim_metric_name = "val_loss"
    config.warm_up_epochs = 1
    wrapper.init_state(config, smoke_test=True)

    wrapper._train_evaluation_step()

    wrapper.best_metrics["val_loss"] -= 1000000
    wrapper._train_evaluation_step()

    for i in range(wrapper.epoch_len + 1):
        wrapper._inc_iter()
    assert wrapper.current_iteration == wrapper.epoch_len + 1
    with pytest.raises(LossDivergedError):
        wrapper._train_evaluation_step()
    assert True


def test_save_dict(wrapper: DeterminiticWrapper, config: RunConfig):
    config.train_config.scheduler_config = SchedulerConfig("step", arguments={})
    wrapper.init_state(config, smoke_test=True)

    save_dict = wrapper.save_dict()
    assert wrapper.optimizer.state_dict() == save_dict["optimizer"]
    assert wrapper.scheduler.state_dict() == save_dict["scheduler"]
    assert (wrapper.model.state_dict()["param"] == save_dict["model"]["param"]).all()
    assert wrapper.scaler.state_dict() == save_dict["scaler"]
    del wrapper.optimizer
    save_dict = wrapper.save_dict()
    assert save_dict["optimizer"] is None
    del wrapper.scheduler
    save_dict = wrapper.save_dict()
    assert save_dict["scheduler"] is None
    del wrapper.model
    with pytest.raises(AttributeError, match="object has no attribute 'model'"):
        save_dict = wrapper.save_dict()


if __name__ == "__main__":
    from tests.conftest import run_tests_local

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]

    _config.experiment_dir = Path("/tmp/test_exp")
    kwargs = {
        "config": copy.deepcopy(_config),
        "wrapper": copy.deepcopy(DeterminiticWrapper(MyDeterministicModel)),
        "train_config": copy.deepcopy(_train_config),
        "direction": "min",
    }
    run_tests_local(test_fns, kwargs)

import copy
import inspect
from pathlib import Path

import pytest
import torch
from torch import nn

from ablator import (
    ModelConfig,
    ModelWrapper,
    OptimizerConfig,
    RunConfig,
    TrainConfig,
)

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


class TestWrapper(ModelWrapper):
    def make_dataloader_train(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl

    def make_dataloader_val(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl


class MyWrongCustomModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        return {"preds": x}, None


class MyInternalCustomModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        loss = x.sum().abs() * 1e-7
        if self.training:
            loss.backward()
        return {"preds": x}, None


class MyWrongPolarCustomModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))
        self.current_iteration = 0

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        loss = x.sum().abs() * 1e-7

        self.current_iteration += 1
        if self.current_iteration > 10:
            return {"preds": x}, None

        return {"preds": x}, loss


class MyPolarCustomModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))
        self.second_param = nn.Parameter(torch.ones(100, 1))
        self.current_iteration = 0

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        loss = x.sum().abs() * 1e-7

        self.current_iteration += 1
        if self.current_iteration > 20:
            return {"preds": x}, None
        if self.current_iteration > 5 and self.training:
            loss.backward()
            return {"preds": x}, None

        return {"preds": x}, loss


class MyCustomInitModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))

    def init_weights(self):
        self.param.data = torch.randn_like(self.param)


class MyCustomUnInitModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))
        self.init_weights = True


def test_no_grads(config: RunConfig, capture_output):
    wrapper = TestWrapper(MyWrongCustomModel)
    stdout, stderr = capture_output(lambda: wrapper.train(config))
    assert (
        "The loss returned by the model is `None` and no optimization parameter"
        " contains gradients. "
        in stdout
    )
    assert wrapper._is_partially_optimized
    assert wrapper._is_self_optim
    wrapper = TestWrapper(MyInternalCustomModel)
    stdout, stderr = capture_output(lambda: wrapper.train(config))
    assert (
        "The loss returned by the model is `None` and no optimization parameter"
        " contains gradients. "
        not in stdout
    )
    assert not wrapper._is_partially_optimized
    assert not wrapper._is_self_optim
    wrapper = TestWrapper(MyWrongPolarCustomModel)
    stdout, stderr = capture_output(lambda: wrapper.train(config))
    assert (
        "The loss returned by the model is `None` and no optimization parameter"
        " contains gradients. "
        in stdout
    )
    assert not wrapper._is_partially_optimized
    assert wrapper._is_self_optim
    wrapper = TestWrapper(MyPolarCustomModel)
    stdout, stderr = capture_output(lambda: wrapper.train(config))
    assert (
        "The loss returned by the model is `None` and no optimization parameter"
        " contains gradients. "
        not in stdout
    )
    assert "Not all optimization parameters contain gradients. " in stdout
    assert wrapper._is_partially_optimized
    assert not wrapper._is_self_optim


def test_init_weights(config: RunConfig):
    custom_wrapper = TestWrapper(MyCustomInitModel)
    custom_wrapper.init_state(config)
    custom_wrapper.create_model()
    custom_model = custom_wrapper.model
    vanilla_wrapper = TestWrapper(MyCustomUnInitModel)
    vanilla_wrapper.init_state(config)
    vanilla_wrapper.create_model()
    vanilla_model = vanilla_wrapper.model
    assert (vanilla_model.param == 1).all()
    assert (custom_model.param != vanilla_model.param).all()

    custom_wrapper.create_model(vanilla_wrapper.save_dict())
    custom_model = custom_wrapper.model
    assert (custom_model.param == vanilla_model.param).all()


if __name__ == "__main__":
    import shutil

    from tests.conftest import _assert_error_msg, _capture_output

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]

    for fn in test_fns:
        parameters = inspect.signature(fn).parameters

        tmp_path = Path("/tmp/test_exp")
        shutil.rmtree(tmp_path, ignore_errors=True)
        kwargs = {
            "tmp_path": tmp_path,
            "assert_error_msg": _assert_error_msg,
            "config": copy.deepcopy(_config),
            "capture_output": _capture_output,
            "train_config": copy.deepcopy(_train_config),
        }
        _kwargs = {k: kwargs[k] for k in parameters}
        fn(**_kwargs)

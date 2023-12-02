import typing

import pytest
import torch
from torch import nn

from ablator import (
    OPTIMIZER_CONFIG_MAP,
    SCHEDULER_CONFIG_MAP,
    Derived,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from ablator.modules.optimizer import OptimizerArgs, get_optim_parameters
from ablator.modules.scheduler import SchedulerArgs


class MyModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.param = nn.Parameter(torch.ones(100))

    def forward(self):
        x = self.param + torch.rand_like(self.param) * 0.01
        return x.sum().abs()


class MySecondCustomModel(MyModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.param = nn.Parameter(torch.ones(100))
        self.second_param = nn.Parameter(torch.ones(100))


class MyPartialCustomModel(MySecondCustomModel):
    def get_optim_param(self):
        return [self.param]


@pytest.mark.parametrize("optim_name", OPTIMIZER_CONFIG_MAP)
@pytest.mark.parametrize("scheduler_name", SCHEDULER_CONFIG_MAP)
def test_optimizers(optim_name, scheduler_name):
    initial_lr = 0.1
    optim_config = OptimizerConfig(optim_name, {"lr": initial_lr})
    args = {}
    if scheduler_name == "cycle":
        args["max_lr"] = 1
    scheduler_config = SchedulerConfig(scheduler_name, arguments=args)
    if hasattr(scheduler_config.arguments, "total_steps"):
        scheduler_config.arguments.total_steps = 100
    if hasattr(scheduler_config.arguments, "threshold"):
        scheduler_config.arguments.threshold = 1
    model = MyModel()
    optim = optim_config.make_optimizer(model)
    scheduler = scheduler_config.make_scheduler(model, optim)
    for i in range(100):
        optim.zero_grad(set_to_none=True)
        loss = model()
        loss.backward()
        optim.step()
        if scheduler_config.arguments.step_when == "val":
            scheduler.step(100)
        else:
            scheduler.step()
    assert model.param.mean().abs().detach().item() < 0.1
    if scheduler_name == "none":
        assert optim.param_groups[0]["lr"] == optim.param_groups[0]["initial_lr"]
    else:
        assert optim.param_groups[0]["lr"] < initial_lr


@pytest.mark.parametrize(
    "optim_name",
    OPTIMIZER_CONFIG_MAP,
)
@pytest.mark.parametrize(
    "model_class",
    [MySecondCustomModel, MyModel, MyPartialCustomModel],
)
def test_optimizer_param_groups(model_class: typing.Type, optim_name):
    model = model_class()
    fn = getattr(model, "get_optim_param", None)
    if fn is not None:
        model_optim_params = [id(p) for p in fn()]
    else:
        model_optim_params = [id(p) for p in model.parameters()]
    optim_config = OptimizerConfig(optim_name, {"lr": 0.1})
    optimizer = optim_config.make_optimizer(model)
    optim_params = [
        id(p) for param_group in optimizer.param_groups for p in param_group["params"]
    ]
    assert len(optim_params) == len(model_optim_params)
    assert all(p in model_optim_params for p in optim_params)
    assert all(p in optim_params for p in model_optim_params)


# This will test whether the function correctly extracts all the parameters from the model
# and whether these are the correct parameters.
# This test case is written for the scenario where `weight_decay`` is None.
def test_get_optim_parameters_without_decay():
    model = torch.nn.Linear(10, 1)
    params = get_optim_parameters(model)
    assert len(list(model.parameters())) == len(list(params))
    for p1, p2 in zip(model.parameters(), params):
        assert torch.all(p1 == p2)


class CustomModelConfig(ModelConfig):
    lr: Derived[int]
    lr_error_limit: int = 5
    mock_param: int = 0


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


# This will test whether the function correctly extracts all the parameters from the model
# and whether these are the correct parameters.
# This test case is written for the scenario where `only_requires_grad`` is False.
def test_get_optim_parameters_without_decay_and_with_all_parameters():
    model = MyCustomModel(CustomModelConfig(lr=0.1))
    params = get_optim_parameters(model)
    assert len(list(model.parameters())) == len(list(params))
    for p1, p2 in zip(model.parameters(), params):
        assert torch.all(p1 == p2)


def test_init_optimizer_not_implemented():
    class DummyModel(nn.Module):
        def forward(self, x):
            return x

    optimizer_args = OptimizerArgs(lr=0.01)
    dummy_model = DummyModel()
    with pytest.raises(NotImplementedError) as e_info:
        optimizer_args.init_optimizer(dummy_model)
    # Test situation where `init_optimizer` is not implemented.
    assert str(e_info.value) == "init_optimizer method not implemented."


def test_init_scheduler_not_implemented():
    class DummyModel(nn.Module):
        def forward(self, x):
            return x

    class DummyOptimizer:
        pass

    scheduler_args = SchedulerArgs(step_when="train")
    dummy_model = DummyModel()
    dummy_optimizer = DummyOptimizer()
    with pytest.raises(NotImplementedError) as e_info:
        scheduler_args.init_scheduler(dummy_model, dummy_optimizer)
    # Test situation where `init_scheduler` is not implemented.
    assert str(e_info.value) == "init_scheduler method not implemented."


if __name__ == "__main__":
    from tests.conftest import run_tests_local

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]

    kwargs = {
        "model_class": [MySecondCustomModel, MyModel, MyPartialCustomModel],
        "optim_name": OPTIMIZER_CONFIG_MAP,
        "scheduler_name": SCHEDULER_CONFIG_MAP,
    }
    run_tests_local(test_fns, kwargs)

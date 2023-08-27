import typing
import pytest
import torch
from torch import nn

from ablator import (
    OPTIMIZER_CONFIG_MAP,
    OptimizerConfig,
    SCHEDULER_CONFIG_MAP,
    SchedulerConfig,
)


class MyModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.param = nn.Parameter(torch.ones(100))

    def forward(self):
        x = self.param + torch.rand_like(self.param) * 0.01
        return x.sum().abs()


class MyCustomModel(MyModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.param = nn.Parameter(torch.ones(100))
        self.second_param = nn.Parameter(torch.ones(100))


class MyPartialCustomModel(MyCustomModel):
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
    [MyCustomModel, MyModel, MyPartialCustomModel],
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


if __name__ == "__main__":
    from tests.conftest import run_tests_local

    l = locals()
    fn_names = [fn for fn in l if fn.startswith("test_")]
    test_fns = [l[fn] for fn in fn_names]

    kwargs = {
        "model_class": [MyCustomModel, MyModel, MyPartialCustomModel],
        "optim_name": OPTIMIZER_CONFIG_MAP,
        "scheduler_name": SCHEDULER_CONFIG_MAP,
    }
    run_tests_local(test_fns, kwargs)

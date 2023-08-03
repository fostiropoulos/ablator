import torch
from torch import nn
import pytest
from ablator.modules.optimizer import get_optim_parameters, get_parameter_names, OptimizerArgs
from ablator.modules.scheduler import SchedulerArgs
from ablator import (
    OPTIMIZER_CONFIG_MAP,
    OptimizerConfig,
    SCHEDULER_CONFIG_MAP,
    SchedulerConfig,
    Derived,
    ModelConfig,
)


class MyModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.param = nn.Parameter(torch.ones(100))

    def forward(self):
        x = self.param + torch.rand_like(self.param) * 0.01
        return x.sum().abs()


def run_optimizer(optim_name, scheduler_name):
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
    # if hasattr(scheduler_config.arguments, "gamma") and scheduler_name!="none":
    #     scheduler_config.arguments.gamma = 0.1
    model = MyModel()
    optim = optim_config.make_optimizer(model)
    scheduler = scheduler_config.make_scheduler(model, optim)
    for i in range(100):
        optim.zero_grad()
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


def test_optimizers():
    for optim_name in OPTIMIZER_CONFIG_MAP:
        for scheduler_name in SCHEDULER_CONFIG_MAP:
            run_optimizer(optim_name, scheduler_name)

# This will test whether the function correctly extracts all the parameters from the model
# and whether these are the correct parameters.
# This test case is written for the scenario where `weight_decay`` is None.
def test_get_optim_parameters_without_decay():
    model = torch.nn.Linear(10, 1)
    params = get_optim_parameters(model)
    assert len(list(model.parameters())) == len(params)
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
    params = get_optim_parameters(model, only_requires_grad=False)
    assert len(list(model.parameters())) == len(params)
    for p1, p2 in zip(model.parameters(), params):
        assert torch.all(p1 == p2)


# Test the recursive line in get_parameter_names function.
# To check if return parameter names of all submodules.
def test_get_parameter_names_with_submodules():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    param_names = get_parameter_names(model, forbidden_layer_types=[])
    expected_param_names = ['0.weight', '0.bias', '2.weight', '2.bias']
    assert set(param_names) == set(expected_param_names)


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
    scheduler_args = SchedulerArgs(step_when='train')
    dummy_model = DummyModel()
    dummy_optimizer = DummyOptimizer()
    with pytest.raises(NotImplementedError) as e_info:
        scheduler_args.init_scheduler(dummy_model, dummy_optimizer)
    # Test situation where `init_scheduler` is not implemented.
    assert str(e_info.value) == "init_scheduler method not implemented."


if __name__ == "__main__":
    test_optimizers()
    test_get_optim_parameters_without_decay()
    test_get_optim_parameters_without_decay_and_with_all_parameters()
    test_get_parameter_names_with_submodules()
    test_init_optimizer_not_implemented()
    test_init_scheduler_not_implemented()


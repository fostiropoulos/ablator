from trainer.modules.optimizer import (
    OptimizerConfig,
    OPTIMIZER_CONFIG_MAP,
)

import torch
from torch import nn

from trainer.modules.scheduler import SCHEDULER_CONFIG_MAP, SchedulerConfig


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


if __name__ == "__main__":
    test_optimizers()

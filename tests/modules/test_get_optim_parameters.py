import torch
from torch import nn

from ablator import (
    OPTIMIZER_CONFIG_MAP,
    OptimizerConfig,
    SCHEDULER_CONFIG_MAP,
    SchedulerConfig,

)
from ablator.modules.optimizer import get_optim_parameters


class MyModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.param = nn.Parameter(torch.ones(100))

    def forward(self):
        x = self.param + torch.rand_like(self.param) * 0.01
        return x.sum().abs()


def run_get_optim_parameters(model, weight_decay, only_requires_grad):

    model_named_params = [name for name, _ in model.named_parameters()]
    parameters = get_optim_parameters(model, weight_decay, only_requires_grad)

    assert isinstance(parameters, list)
    if weight_decay is not None:
        assert isinstance(parameters[0], dict)
        assert isinstance(parameters[1], dict)
        assert len(parameters[0]['params']) <= len(model_named_params)
        assert len(parameters[1]['params']) <= len(model_named_params)

    if not only_requires_grad and weight_decay is None:
        assert len(parameters) == len(model_named_params)

    if only_requires_grad and weight_decay is None:
        # import pdb;pdb.set_trace()
        assert len(parameters) <= len(model_named_params)
    print('All tests passed for weight_decay:{} and only_requires_grad:{}'.format(weight_decay, only_requires_grad))


def test_get_optim_parameters():
    model = MyModel()
    weight_decay_vals = [0.0, None]
    only_requires_grad_vals = [True, False]
    for weight_decay in weight_decay_vals:
        for only_requires_grad in only_requires_grad_vals:
            run_get_optim_parameters(model, weight_decay, only_requires_grad)

if __name__ == '__main__':
    pass

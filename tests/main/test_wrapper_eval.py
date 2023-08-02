import shutil
import copy
from pathlib import Path
import pandas as pd

import torch
from torch import nn

from ablator import (
    ModelConfig,
    ModelWrapper,
    OptimizerConfig,
    RunConfig,
    TrainConfig,
)
from ablator.analysis.results import Results
from ablator.utils.base import get_latest_chkpts

# Learning rate of 1 will result in decreasing the model parameter by 1 value each iteration
optimizer_config = OptimizerConfig(name="sgd", arguments={"lr": 1})
train_config = TrainConfig(
    dataset="test",
    batch_size=128,
    epochs=4,
    optimizer_config=optimizer_config,
    scheduler_config=None,
)

config = RunConfig(
    train_config=train_config,
    model_config=ModelConfig(),
    divergence_factor=100000000,
    verbose="silent",
    device="cpu",
    amp=False,
)
BEST_EPOCH = 3

ITER_PER_EPOCH = 100
BEST_ITER = BEST_EPOCH * ITER_PER_EPOCH


class MyModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.tensor([[BEST_ITER]], dtype=float))

    def forward(self, x: torch.Tensor):
        x = self.param
        # when we reach 0 we continue to decrement to make the test more challenging
        if (x <= 0).all() and self.training:
            # we decrement by 2 since the optimizer will do a +1
            self.param.data -= 2
        return {"preds": x}, x.sum().abs()


class TestWrapper(ModelWrapper):
    def make_dataloader_train(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(ITER_PER_EPOCH)]
        return dl

    def make_dataloader_val(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(ITER_PER_EPOCH)]
        return dl


def assert_results_equal(res: pd.DataFrame, new_res: pd.DataFrame):
    assert (
        (
            res.reset_index()[
                [
                    "best_loss",
                    "best_iteration",
                    "current_iteration",
                    "val_loss",
                    "current_epoch",
                    "step",
                ]
            ]
            == new_res.reset_index()[
                [
                    "best_loss",
                    "best_iteration",
                    "current_iteration",
                    "val_loss",
                    "current_epoch",
                    "step",
                ]
            ]
        )
        .all()
        .all()
    )


def test_wrapper_eval(tmp_path: Path, assert_error_msg):
    config.experiment_dir = tmp_path.joinpath("test_exp")
    TestWrapper(MyModel).train(config)
    res = Results.read_results(config, tmp_path)
    # The loss should decrease by 100 at every iteration
    assert ((BEST_ITER - res["current_iteration"]).abs() == res["val_loss"]).all()
    assert res["best_iteration"].max() == BEST_ITER
    assert res["best_loss"].min() == 0
    assert res.iloc[res["best_loss"].argmin()]["current_iteration"] == BEST_ITER
    assert (res["timestamp"].astype(int).diff(1).dropna() > 0).all()
    new_config = copy.deepcopy(config)
    new_config.experiment_dir = tmp_path.joinpath("test_exp_2")
    new_config.divergence_factor = int(
        (100 - 1) / 1e-5
    )  # the eps in the check for divergence
    msg = assert_error_msg(lambda: TestWrapper(MyModel).train(new_config))
    assert (
        msg
        == "Val loss 1.00e+02 has diverged by a factor larger than 9900000 to best loss 0.00e+00"
    )
    msg = assert_error_msg(lambda: TestWrapper(MyModel).train(new_config, resume=True))
    assert (
        msg
        == "Val loss 1.00e+02 has diverged by a factor larger than 9900000 to best loss 0.00e+00"
    )
    new_config.divergence_factor = 10
    msg = assert_error_msg(lambda: TestWrapper(MyModel).train(new_config, resume=True))
    assert (
        msg
        == "Val loss 1.00e+02 has diverged by a factor larger than 10 to best loss 0.00e+00"
    )
    new_config.divergence_factor = int((100) / 1e-5)
    TestWrapper(MyModel).train(new_config, resume=True)
    # NOTE even if the goal of the test is not to produce an error, we can check the results
    # to see if they are identical to training without interuptions.
    new_res = Results.read_results(new_config, new_config.experiment_dir)
    assert_results_equal(res, new_res)

    new_config = copy.deepcopy(config)
    new_config.experiment_dir = tmp_path.joinpath("test_exp_3")
    new_config.divergence_factor = None
    TestWrapper(MyModel).train(new_config)
    new_res = Results.read_results(new_config, new_config.experiment_dir)
    assert_results_equal(res, new_res)

    # NOTE we test resuming from a checkpoint
    new_config = copy.deepcopy(config)
    new_config.experiment_dir = tmp_path.joinpath("test_exp_4")
    new_config.train_config.epochs = 5

    assert new_config.uid != config.uid
    new_config.init_chkpt = new_config.experiment_dir.joinpath("checkpoints")
    msg = assert_error_msg(lambda: TestWrapper(MyModel).train(new_config))
    assert "test_exp_4/checkpoints is not a valid checkpoint e.g. a `.pt` file. " in msg
    chkpt = get_latest_chkpts(Path(config.experiment_dir).joinpath("checkpoints"))[0]
    assert chkpt.name.endswith("400.pt")
    new_config.init_chkpt = chkpt
    shutil.rmtree(new_config.experiment_dir)
    TestWrapper(MyModel).train(new_config)

    new_res = Results.read_results(new_config, new_config.experiment_dir)
    # we virtually continue training from when validation loss was 100 on the previous checkpoint.
    # the validation loss will keep decreasing
    assert (new_res["current_iteration"] + 100 == new_res["val_loss"]).all()
    bad_config = copy.deepcopy(new_config)
    bad_config.train_config.epochs = 6

    assert new_config.uid != bad_config.uid

    msg = assert_error_msg(lambda: TestWrapper(MyModel).train(bad_config, resume=True))
    assert msg == "Differences between configurations:\n\tepochs:(int)6->(int)5"


if __name__ == "__main__":
    from tests.conftest import _assert_error_msg
    import shutil

    tmp_path = Path("/tmp/test_exp")

    shutil.rmtree(tmp_path, ignore_errors=True)
    test_wrapper_eval(tmp_path, _assert_error_msg)

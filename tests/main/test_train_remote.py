from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile

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
from ablator.config.main import configclass
from ablator.main.configs import ParallelConfig, SearchSpace
from ablator.main.mp import ParallelTrainer, train_main_remote
from ablator.modules.loggers.file import FileLogger
from ablator.main.state import TrialState

from ablator import Derived


class CustomModelConfig(ModelConfig):
    lr: Derived[int]


@configclass
class MyParallelConfig(ParallelConfig):
    model_config: CustomModelConfig


optimizer_config = OptimizerConfig(name="sgd", arguments={"lr": 0.1})
train_config = TrainConfig(
    dataset="test",
    batch_size=128,
    epochs=2,
    optimizer_config=optimizer_config,
    scheduler_config=None,
)
search_space = {
    "train_config.optimizer_config.arguments.lr": SearchSpace(
        value_range=[0, 10], value_type="int"
    ),
}

config = MyParallelConfig(
    train_config=train_config,
    model_config=CustomModelConfig(),
    verbose="silent",
    device="cpu",
    amp=False,
    search_space=search_space,
    optim_metrics={"val_loss": "min"},
    total_trials=10,
    concurrent_trials=10,
    gpu_mb_per_experiment=100,
    cpus_per_experiment=0.5,
)




class TestWrapper(ModelWrapper):
    def make_dataloader_train(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl

    def make_dataloader_val(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl

    def config_parser(self, run_config: MyParallelConfig):
        run_config.model_config.lr = (
            run_config.train_config.optimizer_config.arguments.lr
        )
        return super().config_parser(run_config)


class MyCustomModel(nn.Module):
    def __init__(self, config: CustomModelConfig) -> None:
        super().__init__()
        self.lr = config.lr
        self.param = nn.Parameter(torch.ones(100))
        self.itr = 0

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        self.itr += 1
        if self.itr > 10 and self.lr > 5:
            raise Exception("large lr.")
        return {"preds": x}, x.sum().abs()
    
   
    
# Two tests for the train_main_remote funciton in mp.py (does not run on ray node)
def test_train_remote(tmp_path:Path):
    wrapper = TestWrapper(MyCustomModel)

    my_run_config=MyParallelConfig(
        train_config=train_config,
        model_config=CustomModelConfig(),
        metrics_n_batches = 100,
        verbose="silent",
        device="cpu",
        amp=False,
        search_space=search_space,
        optim_metrics={"val_loss": "min"},
        total_trials=5,
        concurrent_trials=5,
        gpu_mb_per_experiment=0.001,
        cpus_per_experiment=0.001,
        
    )
    my_run_config.experiment_dir = tmp_path
    mp_logger = FileLogger(path=my_run_config.experiment_dir / "mp.log")

    parallel_config,metrics,trial_state = train_main_remote(model=wrapper,run_config=my_run_config,mp_logger=mp_logger,root_dir=my_run_config.experiment_dir)

    # Test on running a successful experiment (very similar to above)
    assert trial_state == TrialState.COMPLETE 
    assert not metrics is None
    

    # Large LR test to trigger a runtime error in model wrapper
    wrapper = TestWrapper(MyCustomModel)
    lr_optimizer_config = OptimizerConfig(name="sgd", arguments={"lr": 10})
    lr_train_config = TrainConfig(
        dataset="test",
        batch_size=128,
        epochs=2,
        optimizer_config=lr_optimizer_config,
        scheduler_config=None,
    )
    lr_run_config=MyParallelConfig(
        train_config=lr_train_config,
        model_config=CustomModelConfig(),
        metrics_n_batches = 100,
        verbose="silent",
        device="cpu",
        amp=False,
        search_space=search_space,
        optim_metrics={"val_loss": "min"},
        total_trials=5,
        concurrent_trials=5,
        gpu_mb_per_experiment=0.001,
        cpus_per_experiment=0.001,
    )
    parallel_config,metrics,trial_state = train_main_remote(model=wrapper,run_config=lr_run_config,mp_logger=mp_logger,root_dir=lr_run_config.experiment_dir)

    # Test on running a experiment with runtime exception
    assert trial_state == TrialState.FAIL
    assert metrics is None
    
    return    

def test_poor_performance(tmp_path:Path):
    # Set early stopping iteration to 1 to trigger TrialState.PRUNED_POOR_PERFORMANCE due to training plateau
    wrapper = TestWrapper(MyCustomModel)

    my_config=MyParallelConfig(
        train_config=train_config,
        model_config=CustomModelConfig(),
        metrics_n_batches = 100,
        verbose="silent",
        device="cpu",
        amp=False,
        search_space=search_space,
        optim_metrics={"val_loss": "min"},
        total_trials=5,
        concurrent_trials=5,
        gpu_mb_per_experiment=0.001,
        cpus_per_experiment=0.001,
        early_stopping_iter=1 # This should cause a TrainPlateauError
    )
    my_config.experiment_dir = tmp_path
    
    ablator=ParallelTrainer(wrapper=wrapper,run_config=my_config)
    ablator.launch(Path(__file__).parent.as_posix(), ray_head_address=None, resume=True)

    return


if __name__ == "__main__":
    import shutil

    tmp_path = Path("/tmp/experiment_dir")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir()

    test_poor_performance(tmp_path)
    test_train_remote(tmp_path)
 


from pathlib import Path

import ray
import torch
from torch import nn

import shutil
from ablator import (
    ModelConfig,
    ModelWrapper,
    OptimizerConfig,
    RunConfig,
    TrainConfig,
)
from ablator.analysis.results import Results
from ablator.main.configs import ParallelConfig, SearchSpace
from ablator.main.mp import ParallelTrainer


class MyCustomModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100))

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(x) * 0.01
        return {"preds": x}, x.sum().abs()
    
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
        value_range=[0.01, 0.1], value_type="float"
    ),
    "train_config.batch_size": SearchSpace(
        categorical_values=[32]
    ),
}

config = ParallelConfig(
    train_config=train_config,
    model_config=ModelConfig(),
    verbose="silent",
    device="cpu",
    amp=False,
    search_space=search_space,
    optim_metrics={"val_loss": "min"},
    total_trials=2,
    concurrent_trials=2,
    gpu_mb_per_experiment=100,
    cpus_per_experiment=0.5,
)

class TestWrapperEval(ModelWrapper):
    def make_dataloader_train(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl

    def make_dataloader_val(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl

def test_results(tmp_path: Path):
    wrapper = TestWrapperEval(MyCustomModel)
    config.experiment_dir = tmp_path

    shutil.rmtree(tmp_path, ignore_errors=True)
    ablator = ParallelTrainer(wrapper=wrapper, run_config = config)

    ablator.launch(working_directory = Path(__file__).parent.as_posix(), ray_head_address=None)

    # To cover the case that if ray is shutdown and use_ray = True. It will initialize ray.
    if ray.is_initialized():
        ray.shutdown()
    results = Results(config = ParallelConfig, experiment_dir=ablator.experiment_dir, use_ray=True)

    assert ray.is_initialized()

    df = results.read_results(config_type = ParallelConfig, experiment_dir=ablator.experiment_dir)

    # since, we are added "path" column separately, check whether it is in the df.
    assert "path" in df.columns

    # check whether all the metrics are loaded.
    metric_names = results.metric_names
    for metrics in metric_names:
        assert isinstance(metrics, str)
        optim, obj_fn = metrics.split('.')
        assert optim == "Optim" and obj_fn in ["min", "max"]


    # check whether the exception handles the case of empty/incorrect json files.
    try:
        json_paths = ablator.experiment_dir.rglob("results.json")
        for json_file in json_paths:
            with open(json_file, "w") as file:
                file.truncate(0)
    
        if ray.is_initialized():
            ray.shutdown()
        results.read_results(config_type = ParallelConfig, experiment_dir=ablator.experiment_dir)
    except Exception as e:
            assert "No objects to concatenate" in str(e) or "All objects passed were None" in str(e)
    
    # to use read_results when provided empty directory (no json-paths).
    try:
        shutil.rmtree(ablator.experiment_dir, ignore_errors=True)
        results.read_results(config_type= ParallelConfig, experiment_dir=ablator.experiment_dir)
    except Exception as e:
        if not str(e).startswith("No results found in"):
            raise e
    
    # Checking Results.__init__() when provided incorrect directory.
    try:
        Results(config = ParallelConfig, experiment_dir=tmp_path, use_ray=False)
    except Exception as e:
        if str(e) != str(tmp_path.joinpath("default_config.yaml")):
            raise e

if __name__ == "__main__":
    test_results(Path("/tmp/results/"))
    pass

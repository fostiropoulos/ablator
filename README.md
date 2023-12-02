

<p align="center">
  <img alt="Ablator Image" src="assets/ablator-logo.png">
</p>
<p align="center">
  <a href="https://dashboard.ablator.online/landing"><img src="https://img.shields.io/badge/docs-ABLATOR-blue" alt="Documentation"></a>
  <a href="https://twitter.com/ABLATOR_ORG"><img src="https://img.shields.io/twitter/follow/username?label=Follow&style=social" alt="Twitter Follow"></a>
  <a href="https://discord.com/invite/9dqThvGnUW"><img src="https://img.shields.io/discord/1153550638324650035?label=Discord&logo=discord" alt="Discord"></a>
  <a href="https://join.slack.com/t/ablator/shared_invite/zt-28009w2fk-fXG5Io38gjM0pnK4f4nZ9w"><img src="https://img.shields.io/badge/Slack-Join%20Us-blue?logo=slack" alt="Slack"></a>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.1-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python 3.10">
  <a href="https://codecov.io/gh/fostiropoulos/ablator"><img src="https://codecov.io/gh/fostiropoulos/ablator/graph/badge.svg?token=LUGKC1R8CG" alt="codecov"></a>
  <!-- <a href="https://github.com/fostiropoulos/ablator/actions/workflows/_linux_test.yml"><img src="https://github.com/fostiropoulos/ablator/actions/workflows/_linux_test.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/fostiropoulos/ablator/actions/workflows/_mac_test.yml"><img src="https://github.com/fostiropoulos/ablator/actions/workflows/_mac_test.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/fostiropoulos/ablator/actions/workflows/_wsl_test.yml"><img src="https://github.com/fostiropoulos/ablator/actions/workflows/_wsl_test.yml/badge.svg" alt="CI"></a> -->
</p>

🚀 ABLATOR is a <i>DISTRIBUTED EXECUTION FRAMEWORK</i> designed to enhance ablation studies in complex machine learning models. It automates the process of configuration and conducts multiple experiments in parallel.

### What are Ablation Studies?

It involves removing specific parts of a neural network architecture or changing different aspects of the training process to examine their contributions to the model's performance.

## Why ABLATOR?

As machine learning models grow in complexity, the number of components that need to be ablated also increases. This consequently expands the search space of possible configurations, requiring an efficient approach to horizontally scale multiple parallel experimental trials. ABLATOR is a tool that aids in the horizontal scaling of experimental trials.

Instead of manually configuring and conducting multiple experiments with various hyperparameter settings, ABLATOR automates this process. It initializes experiments based on different hyperparameter configurations, tracks the state of each experiment, and provides experiment persistence on the cloud.

## Key Features

- It is a tool that simplifies the process of prototyping of models.
- It streamlines model experimentation and evaluation.
- It offers a flexible configuration system.
- It facilitates result interpretation through visualization.
- "Auto-Trainer" feature reduces redundant coding tasks.

## ABLATOR vs. Without ABLATOR

<p align='center'><b>Left</b>, ABLATOR efficiently conducts multiple trials in parallel based and log the experiment results. </p>
<p align='center'><b>Right</b>, <b>manually</b>, one would need to run trials sequentially, demanding more effort and independent analysis.
</p>

![Comparison of ABLATOR and Manual Proces](assets/ablator.png)

## Install
For MacOS and Linux systems directly install via pip.

```pip install ablator```

<p>If you are using Windows, you will need to install WSL using the official from Microsoft. </p>
<p> WSL is a Linux subsystem and for ABLATOR purposes is identical to using Linux. </p>

## Basic Concepts

### 1. Create your Configuration

```python
from torch import nn
import torch
from ablator import (
    ModelConfig,
    ModelWrapper,
    OptimizerConfig,
    TrainConfig,
    configclass,
    Literal,
    ParallelTrainer,
    SearchSpace,
)
from ablator.config.mp import ParallelConfig


@configclass
class TrainConfig(TrainConfig):
    dataset: str = "random"
    dataset_size: int


@configclass
class ModelConfig(ModelConfig):
    layer: Literal["layer_a", "layer_b"] = "layer_a"


@configclass
class ParallelConfig(ParallelConfig):
    model_config: ModelConfig
    train_config: TrainConfig


config = ParallelConfig(
    experiment_dir="ablator-exp",
    train_config=TrainConfig(
        batch_size=128,
        epochs=2,
        dataset_size=100,
        optimizer_config=OptimizerConfig(name="sgd", arguments={"lr": 0.1}),
        scheduler_config=None,
    ),
    model_config=ModelConfig(),
    device="cpu",
    search_space={
        "model_config.layer": SearchSpace(categorical_values=["layer_a", "layer_b"])
    },
    total_trials=2,
)

```

### 2. Define your Model

```python

class SimpleModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.layer == "layer_a":
            self.param = nn.Parameter(torch.ones(100, 1))
        else:
            self.param = nn.Parameter(torch.randn(200, 1))

    def forward(self, x: torch.Tensor):
        x = self.param
        return {"preds": x}, x.sum().abs()


class SimpleWrapper(ModelWrapper):
    def make_dataloader_train(self, run_config: ParallelConfig):
        dl = [torch.rand(100) for i in range(run_config.train_config.dataset_size)]
        return dl

    def make_dataloader_val(self, run_config: ParallelConfig):
        dl = [torch.rand(100) for i in range(run_config.train_config.dataset_size)]
        return dl
```

### 3. Launch 🚀

```python
mywrapper = SimpleWrapper(SimpleModel)
with ParallelTrainer(mywrapper, config) as ablator:
    ablator.launch(".")
```

## Learn More about ABLATOR Modules

<table border="0">
    <tr>
        <td align="center">
            <a target="_blank" href="https://docs.ablator.org/en/latest/config.html">
                <img src="https://www.svgrepo.com/show/399136/wrench.svg" alt="Configuration Icon" width="30%">
            </a>
        </td>
        <td align="center">
            <a target="_blank" href="https://docs.ablator.org/en/latest/training.html">
                <img src="https://www.svgrepo.com/show/407038/person-lifting-weights.svg" alt="Process Icon" width="40%">
            </a>
        </td>
        <td align="center">
            <a target="_blank" href="https://docs.ablator.org/en/latest/results.html">
                <img src="https://www.svgrepo.com/show/398472/test-tube.svg" alt="Results Icon" width="30%">
            </a>
        </td>
        <td align="center">
            <a target="_blank" href="https://docs.ablator.org/en/latest/analysis.html">
                <img src="https://www.svgrepo.com/show/395855/bar-chart.svg" alt="Analysis Icon" width="30%">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">Configuration Module</td>
        <td align="center">Training Module</td>
        <td align="center">Experiment and Metrics Module</td>
        <td align="center">Analysis Module</td>
    </tr>
</table>

## Tutorials

Explore a variety of tutorials and examples on how to utilize ABLATOR.
Ready to dive in? 👉 [Ablation Tutorials](https://github.com/fostiropoulos/ablator-tutorials)

## Contribution Guidelines

ABLATOR is open source, and we value contributions from our community!
Check out our [Development Guide](https://github.com/fostiropoulos/ablator/blob/v0.0.1-mp-mount/DEVELOPER.md) for details on our development process and insights into the internals of the ABLATOR library.

<p>For any bugs or feature requests related to ABLATOR, please visit our GitHub Issues or reach out to slack </p>

## Ablator Community
| Platform       | Purpose                                                             | Support Level  |
|----------------|---------------------------------------------------------------------|----------------|
| [GitHub Issues](https://github.com/fostiropoulos/ablator/issues) | To report issues or suggest new features. | ABLATOR Team   |
| [Slack](https://join.slack.com/t/ablator/shared_invite/zt-28009w2fk-fXG5Io38gjM0pnK4f4nZ9w)        | To collaborate with fellow ABLATOR users.  | Community      |
| [Discord](https://discord.com/invite/9dqThvGnUW)       | To inquire about ABLATOR usage and collaborate with other ABLATOR enthusiasts. | Community      |
| [Twitter](https://twitter.com/ABLATOR_ORG)       | For staying up-to-date on new features of Ablator.               | ABLATOR Team   |

## References

```tex
@inproceedings{fostiropoulos2023ablator,
  title={ABLATOR: Robust Horizontal-Scaling of Machine Learning Ablation Experiments},
  author={Fostiropoulos, Iordanis and Itti, Laurent},
  booktitle={AutoML Conference 2023 (ABCD Track)},
  year={2023}
}
```

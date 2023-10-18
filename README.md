# 🚀 ABLATOR

|<img src="docs/source/_static/logo.png" alt="logo" width="200"/>|  [Website](https://ablator.org) \| [Docs](https://docs.ablator.org) </br></br><a href="https://join.slack.com/t/ablator/shared_invite/zt-23ak9ispz-HObgZSEZhyNcTTSGM_EERw" target="_blank"> <img class="banner-icon" src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Slack_Technologies_Logo.svg/1200px-Slack_Technologies_Logo.svg.png" height="20px" alt="Slack"> </a> <a href="https://twitter.com/ablator_org" target="_blank"> <img class="banner-icon" src="https://img.shields.io/twitter/url/https/twitter.com/ablator_org.svg?style=social&label=Follow%20%40ablator_org" height="20px" alt="Twitter"> </a> <a href="https://discord.gg/9dqThvGnUW" target="_blank"> <img class="banner-icon" src="https://dcbadge.vercel.app/api/server/9dqThvGnUW" height="20px" alt="Twitter"> </a> <br></br>[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![codecov](https://codecov.io/gh/fostiropoulos/ablator/graph/badge.svg?token=LUGKC1R8CG)](https://codecov.io/gh/fostiropoulos/ablator) </br>[![CI](https://github.com/fostiropoulos/ablator/actions/workflows/_linux_test.yml/badge.svg)](https://github.com/fostiropoulos/ablator/actions/workflows/_linux_test.yml) [![CI](https://github.com/fostiropoulos/ablator/actions/workflows/_mac_test.yml/badge.svg)](https://github.com/fostiropoulos/ablator/actions/workflows/_mac_test.yml)[![CI](https://github.com/fostiropoulos/ablator/actions/workflows/_wsl_test.yml/badge.svg)](https://github.com/fostiropoulos/ablator/actions/workflows/_wsl_test.yml) |
|--|--|


A distributed experiment execution framework for deep learning models.

ABLATOR provides an *auto-trainer* (or bring your own) for your deep learning model to help you prototype.

Once you are confident there are no bugs in your code, you can launch 🚀 many experiments across many machines and evaluate thousands of model variants with 1 code change.



```python
ProtoTrainer -> ParallelTrainer
```


ABLATOR is designed with ablation experiments first. Ablation experiments can be used to design, improve and learn on how each component of a neural network affects performance. For example, `does X layer improve performance?`

<!-- [figure] in progress  -->

## Why ABLATOR?

ABLATOR is meant to be used as:

1. **Pedagogical tool**, for learning and understanding what helps improve a model's performance. A lot of the design choices of a Neural Network are ad-hoc. For example, the Transformer architecture was introduced in 2017 ([Attention Is All You Need](https://arxiv.org/abs/1706.03762)) but it wasn't until 2020 that the influence of the [layer-norm was thoroughly examined](https://arxiv.org/pdf/2002.04745.pdf).

2. **Research**, for rapid development and prototyping of a new idea or novel component

3. **Deployment**, for A / B testing of the ML model and whether the code changes improve the performance of an ML model during deployment i.e. as part of CI / CD pipeline.

Doing all of the above **efficiently** and at **scale** requires cumbersome set-up and know-how. We remove boiler-plate code and details to help users achieve what they want.

### Technical Details
 1. Strictly typed configuration system prevents errors.
 2. Seamless prototyping to production.
 3. Stateful experiment design. Stop, Resume, Share your experiments
 4. Automated analysis artifacts
 5. Auto-Trainer: Remove boiler-plate code


## Install
For **MacOS** and **Linux** you can directly install via pip. If you are using **Windows**, you will need to install WSL. [Using the official guide](https://learn.microsoft.com/en-us/windows/wsl/install). WSL is a Linux subsystem and for ABLATOR purposes is identical to using Linux.


Use a python virtual environment to avoid version conflicts.

```bash
pip install ablator
```


### Multi-Node Cluster

ABLATOR uses a distributed framework [Ray](https://ray.io) to launch experiments in Parallel. It is possible to connect several servers (nodes) in a single network to distribute the experimental trials among them. This is currently only supported for Linux servers (or Linux containers). Installing and setting up a ray cluster is an endeavor of its own and we recommend the [official guide](https://docs.ray.io/en/latest/cluster/getting-started.html) for detailed instructions.

## Usage

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

## Tutorials

There are several tutorials and examples on how to use ABLATOR.

We have created a [dedicated repository with them](https://github.com/fostiropoulos/ablator-tutorials)

Or simply get started

<a target="_blank" href="https://colab.research.google.com/github/fostiropoulos/ablator-tutorials/blob/6d79f47703b05f99655a717662f717d238f5dbfc/notebooks/HPO.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


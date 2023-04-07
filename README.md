# Model Trainer

An experimental verification analysis tool-kit. This is a wrapper for your model to experiment on different training settings to verify the validity of your hypothesis, or model. The trainer works by creating stateful and stateless experimental attributes through complex strictly typed configuration files. The configuration recipes allows for the verification of the experimental hypothesis.

1. Why not use pytorch lighting etc...
   1. You can wrap your model with pytorch lighting or any other library inside the model wrapper.
   2. You can train with more complex settings such as custom gradient updates that is much more challenging to integrate with other libraries.
   3. You can set up your experiment, containerize everything and distributely run multiple parallel experiments at once.
   4. You can obtain fast analysis and results without having to manually check multiple settings.


This tool-kit can help reduce the time it takes to go from hypothesis -> experiment design -> analysis -> results. Allow you to spend more time in the creative process of ML research and less time on dev-ops.

### Alpha - Phase

The library is under active development and a lot of the API endpoints will be removed / renamed or their functionality changed without notice.

## Semi-Minimal working example on CIFAR10

[HERE](tests/resnet.py)


## High-Level Overview

### 1. Install

Highly suggest to use a python virtual enviroment to avoid version conflicts.

`cd model-trainer`
`pip install .`

### 2. Configs


### 3. Data

You have access to the config inside the dataloader creator in order to adjust the dataset specifically to the model configuration or train configuration.

```python
class CifarWrapper(torchvision.datasets.CIFAR10):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        x, y = super().__getitem__(index)
        return {
            "x": x,
            "labels": y,
            "custom_input": torch.zeros(x.shape, dtype=torch.bool),
        }

def load_cifar10(config, flag="train"):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    # You can access config["train"]["train_hparam"] here
    dataset = CifarWrapper(
        root=config["save_dir"],
        train=flag == "train",
        transform=transform,
        target_transform=None,
        download=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config["train"]["batch_size"], shuffle=True
    )
    return dataloader
```

### 4. Model

You need to define your model that it returns predictions, and a loss. You can also not return a loss i.e. set it to `None` and in that case you would have to make the updates on the parameters `manually`. The forward input keys must match the ones returned by your Dataset.

```python
class MyModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

    def forward(self, x, labels=None, custom_input=None):
        out = some_fn()
        out = out.argmax(axis=-1)
        return out, loss

```

### 5. ModelWrapper

You can over-write functions here such as `custom_evalatuion` and custom `train_step` or training loop all-together.

```python
class MyModelWrapper(ModelWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

```


### 6. Profit!


TODO: Explain types of trainers
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import torch
import torchvision
import torchvision.models as models
from sklearn.metrics import accuracy_score
from torch import nn
from torchvision import transforms
from trainer import ModelWrapper
from trainer.config.main import configclass
from trainer.config.run import ModelConfigBase, RunConfig
from torch.utils.data import DataLoader
from PIL import Image


class CifarWrapper(torchvision.datasets.CIFAR10):
    def __getitem__(self, index: int) -> Dict[Any, Any]:
        x, y = super().__getitem__(index)
        return {
            "x": x,
            "labels": y,
            "custom_input": transforms.ToTensor()(Image.fromarray(self.data[index])),
        }


@configclass
class ModelConfig(ModelConfigBase):
    # Configurable attributes
    name: str
    weights: str = "IMAGENET1K_V1"
    progress: bool = False


@configclass
class ResRunConfig(RunConfig):
    model_config: ModelConfig


def load_cifar10(config: RunConfig, flag="train") -> DataLoader:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    root = "/tmp/cifar10"
    dataset = CifarWrapper(
        root=root,
        train=flag == "train",
        transform=transform,
        target_transform=None,
        download=not Path(root).exists(),
    )
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=config.train_config.batch_size,
        shuffle=True,
    )
    return dataloader


class MyModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

        model_dict = {
            "resnet18": models.resnet18,
            "alexnet": models.alexnet,
            "vgg16": models.vgg16,
            "squeezenet": models.squeezenet1_0,
            "densenet": models.densenet161,
            "inception": models.inception_v3,
            "googlenet": models.googlenet,
        }

        self.model = model_dict[config.name](weights=config.weights)

    def forward(self, x, labels=None, custom_input=None):
        # custom_input is for demo purposes only, defined in the dataset wrapper
        out: torch.Tensor = self.model(x)
        loss = None
        if labels is not None:
            loss = self.loss(out, labels)

        out = out.argmax(dim=-1)
        return out, loss


class MyModelWrapper(ModelWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_dataloader_train(self, run_config: ResRunConfig):  # type: ignore
        return load_cifar10(run_config, flag="val")
        # return load_cifar10(run_config, flag="train")

    def make_dataloader_val(self, run_config: ResRunConfig):  # type: ignore
        return load_cifar10(run_config, flag="val")

    @property
    def evaluation_functions(self) -> Dict[str, Callable]:
        return {"accuracy_score": accuracy_score}

    def custom_evaluation(
        self, model: nn.Module, dataloader: Iterable
    ) -> Optional[Dict[str, Any]]:
        b = next(iter(dataloader))
        img = torchvision.utils.make_grid(b["custom_input"])
        self.logger.add_image("train", img, self.iteration)
        return super().custom_evaluation(model, dataloader)

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import torch
import torchvision
import torchvision.models as models
from PIL import Image
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from ablator import ModelConfig, ModelWrapper, RunConfig, configclass, Literal


class CifarWrapper(torchvision.datasets.CIFAR10):
    def __getitem__(self, index: int) -> Dict[Any, Any]:
        x, y = super().__getitem__(index)
        return {
            "x": x,
            "labels": y,
            "custom_input": transforms.ToTensor()(Image.fromarray(self.data[index])),
        }


@configclass
class ResConfig(ModelConfig):
    # Configurable attributes
    name: Literal[
        "resnet18",
        "resnet34",
        "resnet50",
    ]
    weights: str = "IMAGENET1K_V1"
    progress: bool = False


@configclass
class ResRunConfig(RunConfig):
    model_config: ResConfig


def load_cifar10(config: ResRunConfig, flag: str = "train") -> DataLoader:
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
    def __init__(self, config: ResConfig) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

        model_dict = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
        }

        self.model = model_dict[config.name](weights=config.weights)

    def forward(self, x, labels, custom_input=None):
        # custom_input is for demo purposes only, defined in the dataset wrapper
        out: torch.Tensor = self.model(x)
        loss = None
        if labels is not None:
            loss = self.loss(out, labels)

        out = out.argmax(dim=-1)
        return {"y_pred": out, "y_true": labels}, loss


def my_accuracy(y_true, y_pred):
    return accuracy_score(y_true.flatten(), y_pred.flatten())


class MyModelWrapper(ModelWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_dataloader_train(self, run_config: ResRunConfig):  # type: ignore
        # return load_cifar10(run_config, flag="val")
        return load_cifar10(run_config, flag="train")

    def make_dataloader_val(self, run_config: ResRunConfig):  # type: ignore
        return load_cifar10(run_config, flag="val")

    def evaluation_functions(self) -> Dict[str, Callable]:
        return {"accuracy_score": my_accuracy}

    def custom_evaluation(
        self, model: nn.Module, dataloader: Iterable
    ) -> Optional[Dict[str, Any]]:
        b = next(iter(dataloader))
        img = torchvision.utils.make_grid(b["custom_input"])
        self.logger.update({"train_image": img})
        return super().custom_evaluation(model, dataloader)

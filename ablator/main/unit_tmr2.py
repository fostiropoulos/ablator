import shutil
import argparse
import unittest
from unittest.mock import MagicMock
from typing import Any, Callable, Dict
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score

from ablator import ModelConfig, TrainConfig, OptimizerConfig, RunConfig, configclass, Literal,ModelWrapper,ProtoTrainer,ParallelConfig
# from ablator.ablator.modules.metrics.loggers.file import FileLogger
from ablator.main.mp import train_main_remote
from ablator.main.configs import SearchSpace
# from modules.metrics.loggers.file import FileLogger

@configclass
class SimpleConfig(ModelConfig):
    name: Literal["simplenet"]

# @configclass
# class SimpleRunConfig(RunConfig):
#     model_config: SimpleConfig

@configclass
class ParallelRunConfig(ParallelConfig):
    model_config: SimpleConfig


run_config = ParallelRunConfig(
    model_config=SimpleConfig(name="simplenet"),
    train_config=TrainConfig(
        dataset="mnist",
        batch_size=64,
        epochs=10,
        optimizer_config = OptimizerConfig(

                name = "sgd",
                arguments = {
                    "lr": 0.001,
                    "momentum": 0.1
                }
            ),
        scheduler_config=None,
        rand_weights_init=True,

    ),
    experiment_dir="/tmp/exp2",
    total_trials=20,
    concurrent_trials=20,
    optim_metrics={"val_loss": "max"},
    metrics_n_batches = 800,
    device="cuda",
    amp=True,
    search_algo="random",
    cpus_per_experiment=1,
    gpu_mb_per_experiment=4096,
    ignore_invalid_params=True,
    search_space={
        "train_config.optimizer_config.arguments.lr": SearchSpace(value_range=[0, 10], value_type="int"),
    },
    
)




# Define a simple CNN model using components from PyTorch packages
# And then we wrap up the CNN model in a wrapper class, which defines the loss function,
# forward pass and indicated output formats

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 200, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(200, 250, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x


class MyModel(nn.Module):
    def __init__(self, config: SimpleConfig) -> None:
        super().__init__()
        self.model = SimpleCNN()
        self.loss = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x, labels, custom_input=None):
        # custom_input is for demo purposes only, defined in the dataset wrapper
        out = self.model(x)
        loss = self.loss(out, labels)
        if labels is not None:
            loss = self.loss(out, labels)

        out = out.argmax(dim=-1)
        return {"y_pred": out, "y_true": labels}, loss


# Create the training & validation dataloaders from the MNIST dataset.
# Also, data preprocessing is defined here, including normalization and other transformations

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./datasets', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./datasets', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)


# A evaluation function is definded here for Ablator to evaluate the model and training process.

def my_accuracy(y_true, y_pred):
    return accuracy_score(y_true.flatten(), y_pred.flatten())



class MyModelWrapper(ModelWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_dataloader_train(self, run_config: ParallelRunConfig):  # type: ignore
        return trainloader

    def make_dataloader_val(self, run_config: ParallelRunConfig):  # type: ignore
        return testloader

    def evaluation_functions(self) -> Dict[str, Callable]:
        return {"accuracy_score": my_accuracy}
    




if __name__ == "__main__":
    
    wrapper = MyModelWrapper(model_class=MyModel)

    '''
    Creates a unit test that returns a Trial State status of TrialState.FAIL: 3>.

    This was achieved by changing the structure of the SimpleCNN model to be a
    3 layer CNN with a 2 layer fully connected network. This was done by changing
    the number of input channels to the first convolutional layer to 3 and the
    number of output channels to 3 for the second convolutional layer. The number
    of input features for the first fully connected layer was also changed to 3.
    

    
    Similar code the the get started feature of ablator is used here however
    the run_config is changed to leverage the parallel run config allowing for 
    concurrent trials to be run, gpu and cpu resources to be leveraged and more.
    '''




    test = train_main_remote(
        model=wrapper,
        run_config=run_config,
        root_dir="/tmp/exp2",
        mp_logger=MagicMock(),
        clean_reset=False

    )
    print((test))

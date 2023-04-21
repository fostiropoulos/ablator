from ablator.config.main import ConfigBase, configclass
from ablator.main.configs import ParallelConfig,TrainConfig
from ablator.main.configs import SearchSpace, OptimizerConfig,ModelConfig
from ablator.main.state import ExperimentState
from pathlib import Path
from ablator.config.types import (
    Optional,
    Stateless,
    Literal,
    Tuple,
    List,
    Enum,
    Dict,
)
# optimizer_config = OptimizerConfig(name="sgd", arguments={"lr": 0.01})
# train_config = TrainConfig(
#     dataset="test",
#     batch_size=128,
#     epochs=2,
#     optimizer_config=optimizer_config,
#     scheduler_config=None,
# )
# search_space = {"train_config.optimizer_config.arguments.lr": SearchSpace(value_range=[0, 0.1], value_type="float")}

# @configclass
# class MyConfig(ParallelConfig):
#     pass
# config = MyConfig(
#     train_config=train_config,
#     model_config=ModelConfig(),
#     verbose="silent",
#     device="cpu",
#     amp=False,
#     search_space=search_space,
#     optim_metrics={"acc": "max"},
#     total_trials=100,
#     concurrent_trials=100
# )

# ExperimentState(Path("/tmp"),config=config)

import numpy as np
import matplotlib.pyplot as plt
from ablator.analysis.plot import LinearPlot
attributes = np.array([1, 2, 3, 4, 5])
metric = np.array([1, 3, 4, 5, 7])
lp = LinearPlot(attributes, metric)
fig, ax = lp._make()

plt.show()

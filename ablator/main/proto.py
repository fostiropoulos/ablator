import copy
from copy import deepcopy

import torch

from ablator.main.configs import RunConfig
from ablator.main.model.wrapper import ModelWrapper


class ProtoTrainer:
    """
    Manages resources for Prototyping
    """

    def __init__(
        self,
        wrapper: ModelWrapper,
        run_config: RunConfig,
    ):
        super().__init__()

        self.wrapper = copy.deepcopy(wrapper)
        self.run_config: RunConfig = copy.deepcopy(run_config)
        if self.run_config.experiment_dir is None:
            raise RuntimeError("Must specify an experiment directory.")

    def pre_train_setup(self):
        """
        Used to prepare resources to avoid stalling during training or when resources are
        shared between trainers.
        """

    def _init_state(self):
        """
        initialize the data state of the wrapper to force downloading and processing any data artifacts
        in the main train process as opposed to inside the wrapper.
        """
        self.pre_train_setup()
        mock_wrapper = copy.deepcopy(self.wrapper)
        mock_wrapper._init_state(run_config=copy.deepcopy(self.run_config), debug=True)

    def launch(self, debug: bool = False):
        self._init_state()
        metrics = self.wrapper.train(run_config=self.run_config, debug=debug)
        self.sync()
        return metrics

    def sync(self):
        """
        Syncs training artifacts with external logging services.
        """


    def evaluate(self):
        metrics = self.wrapper.evaluate(self.run_config)
        self.sync()
        return metrics

    def smoke_test(self, config=None):
        if config is None:
            config = self.run_config
        run_config = deepcopy(config)
        wrapper = deepcopy(self.wrapper)
        wrapper.train(run_config=run_config, smoke_test=True)
        del wrapper
        torch.cuda.empty_cache()

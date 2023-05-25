import copy
from copy import deepcopy

import torch

from ablator.main.configs import RunConfig
from ablator.main.model.wrapper import ModelWrapper


class ProtoTrainer:
    """
    Manages resources for Prototyping.

    Attributes
    ----------
    wrapper : ModelWrapper
        The main model wrapper.
    run_config : RunConfig
        Running configuration for the model.

    Raises
    ------
    RuntimeError
        If experiment directory is not defined in the running configuration.
    """

    def __init__(
        self,
        wrapper: ModelWrapper,
        run_config: RunConfig,
    ):
        """
        Initialize model wrapper and running configuration for the model.

        Parameters
        ----------
        wrapper : ModelWrapper
            The main model wrapper.
        run_config : RunConfig
            Running configuration for the model.
        """
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
        Initialize the data state of the wrapper to force downloading and processing any data artifacts
        in the main train process as opposed to inside the wrapper.
        """
        self.pre_train_setup()
        mock_wrapper = copy.deepcopy(self.wrapper)
        mock_wrapper._init_state(run_config=copy.deepcopy(self.run_config), debug=True)

    def launch(self, debug: bool = False):
        """
        Initialize the data state of the wrapper and train the model inside the wrapper, then sync training
        results (logged to experiment directory while training) with external logging services (e.g Google
        cloud storage, other remote servers).

        Parameters
        ----------
        debug : bool, default=False
            Whether to train model in debug mode.

        Returns
        -------
        metrics : TrainMetrics
            Metrics returned after training.
        """
        self._init_state()
        metrics = self.wrapper.train(run_config=self.run_config, debug=debug)
        self.sync()
        return metrics

    def sync(self):
        """
        Syncs training artifacts with external logging services.
        """

    def evaluate(self):
        """
        Run model evaluation on the training results, sync evaluation results to external logging services
        (e.g Google cloud storage, other remote servers).

        Returns
        -------
        metrics : TrainMetrics
            Metrics returned after evaluation.
        """
        metrics = self.wrapper.evaluate(self.run_config)
        self.sync()
        return metrics

    def smoke_test(self, config=None):
        """
        Run a smoke test training process on the model.

        Parameters
        ----------
        config : RunConfig
            Running configuration for the model.
        """
        if config is None:
            config = self.run_config
        run_config = deepcopy(config)
        wrapper = deepcopy(self.wrapper)
        wrapper.train(run_config=run_config, smoke_test=True)
        del wrapper
        torch.cuda.empty_cache()

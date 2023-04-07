from abc import ABC
import copy
from copy import deepcopy
from os.path import join
from typing import Optional

import torch
from trainer.config.run import RunConfig
from trainer.modules.logging.file import FileLogger
from trainer.modules.model.wrapper import ModelWrapper
from trainer.utils.mp import get_memory_use


class BaseTrainer(ABC):
    def __init__(
        self,
        model: ModelWrapper,
        run_config: RunConfig,
        description="",
    ):
        super().__init__()

        self.model = copy.deepcopy(model)
        self.description = description
        self.run_config: RunConfig = copy.deepcopy(run_config)

        self.experiment_dir: str = self.run_config.experiment_dir

        self.logger = FileLogger(path=join(self.experiment_dir, "trainer.log"))
        self.logger.info(f"Saving progress at {self.experiment_dir}")
        self.run_config.write(join(self.experiment_dir, "run_config.yaml"))

        loaded_config = type(self.run_config).load(
            join(self.experiment_dir, "run_config.yaml")
        )

        diff = self.run_config.diff(loaded_config, ignore_stateless=True)

        assert len(diff) == 0, f"Can't write and load provided config. Diff: {diff}"

    def pre_train_setup(self):
        pass

    def init_state(self):
        """
        initialize the data state of the model to force downloading and processing any data artifacts
        in the main train process as opposed to inside the model.
        """
        self.pre_train_setup()
        mock_model = copy.deepcopy(self.model)
        mock_model.run_config = copy.deepcopy(self.run_config)
        mock_model.init_data_state()

    def launch(self):

        self.init_state()
        self.model.train(run_config=self.run_config)
        self.model.sync()

    def evaluate(self, model_dir: Optional[str] = None, chkpt: Optional[str] = None):

        self.model.evaluate(self.run_config, model_dir=model_dir, chkpt=chkpt)
        self.model.sync()

    def smoke_test(self, config=None):
        if config is None:
            config = self.run_config
        run_config = deepcopy(config)
        model = deepcopy(self.model)
        model.train(run_config=run_config, smoke_test=True)
        del model
        torch.cuda.empty_cache()



    def get_memory_use(self, config: Optional[RunConfig] = None):
        if config is None:
            config = self.run_config

        return get_memory_use(self.model, config=config)

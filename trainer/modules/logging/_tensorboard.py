import subprocess
from pathlib import Path
from typing import Optional, Union
import numpy as np

import pandas as pd
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from trainer.config.main import ConfigBase
from trainer.modules.logging.base import LoggerBase
from trainer.utils.config import flatten_nested_dict


class TensorboardLogger(LoggerBase):
    def __init__(self, summary_dir: Union[str, Path]):
        self.summary_dir = summary_dir
        self.backend_logger = SummaryWriter(log_dir=summary_dir)

    def _add_image(self, k, v, itr, dataformats="CHW"):
        self.backend_logger.add_image(k, v, itr, dataformats=dataformats)

    def _add_table(self, k, v: pd.DataFrame, itr):
        self.backend_logger.add_text(k, v.to_markdown(), itr)

    def _add_text(self, k, v, itr):
        self.backend_logger.add_text(k, v, itr)

    def _add_scalars(self, k, v_dict, itr):
        self.backend_logger.add_scalars(k, v_dict, itr)

    def _add_scalar(self, k, v, itr):
        if v is None:
            self.backend_logger.add_scalar(k, np.nan, itr)
        else:
            self.backend_logger.add_scalar(k, v, itr)

    def _write_config(self, config: ConfigBase):
        hparams = flatten_nested_dict(config.to_dict())
        run_config = OmegaConf.to_yaml(OmegaConf.create(hparams)).replace("\n", "\n\n")
        self.backend_logger.add_text("config", run_config,0)

    def _sync(self):
        pass

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from ablator.config.main import ConfigBase
from ablator.config.utils import flatten_nested_dict
from ablator.modules.loggers import LoggerBase


class TensorboardLogger(LoggerBase):
    def __init__(self, summary_dir: Union[str, Path]):
        self.summary_dir = Path(summary_dir).as_posix()
        self.backend_logger = SummaryWriter(log_dir=summary_dir)

    def add_image(self, k, v, itr, dataformats="CHW"):
        self.backend_logger.add_image(k, v, itr, dataformats=dataformats)

    def add_table(self, k, v: pd.DataFrame, itr):
        self.backend_logger.add_text(k, v.to_markdown(), itr)

    def add_text(self, k, v, itr):
        self.backend_logger.add_text(k, v, itr)

    def add_scalars(self, k, v: dict[str, float | int], itr):
        for _k, _v in v.items():
            self.backend_logger.add_scalar(f"{k}_{_k}", _v, itr)
        # NOTE this is buggy:
        # self.backend_logger.add_scalars(k, v_dict, itr)

    def add_scalar(self, k, v, itr):
        if v is None:
            self.backend_logger.add_scalar(k, np.nan, itr)
        else:
            self.backend_logger.add_scalar(k, v, itr)

    def write_config(self, config: ConfigBase):
        hparams = flatten_nested_dict(config.to_dict())
        run_config = OmegaConf.to_yaml(OmegaConf.create(hparams)).replace("\n", "\n\n")
        self.backend_logger.add_text("config", run_config, 0)

    def _sync(self):
        pass

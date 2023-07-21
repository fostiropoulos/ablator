from pathlib import Path
from typing import Union
import logging
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter

from ablator.config.main import ConfigBase
from ablator.config.utils import flatten_nested_dict
from ablator.modules.loggers import LoggerBase

logging.getLogger("tensorboardX").setLevel(logging.ERROR)


class TensorboardLogger(LoggerBase):
    """
    A logger class for Tensorboard visualization.

    Attributes
    ----------
    summary_dir : Union[str, Path]
        The directory to store the Tensorboard summary files.
    backend_logger : SummaryWriter
        The PyTorch Tensorboard SummaryWriter object used to log data.
    """

    def __init__(self, summary_dir: Union[str, Path]):
        """
        Initialize the TensorboardLogger with a summary directory.

        Parameters
        ----------
        summary_dir : Union[str, Path]
            The directory to store the Tensorboard summary files.
        """
        self.summary_dir = Path(summary_dir).as_posix()
        self.backend_logger = SummaryWriter(log_dir=summary_dir)

    def add_image(self, k, v, itr, dataformats="CHW"):
        """
        Add an image to the TensorBoard dashboard.

        Parameters
        ----------
        k : str
            The tag associated with the image.
        v : np.ndarray
            The image data.
        itr : int
            The iteration number.
        dataformats : str, optional
            The format of the image data, by default ``"CHW"``.
        """
        self.backend_logger.add_image(k, v, itr, dataformats=dataformats)

    def add_table(self, k, v: pd.DataFrame, itr):
        """
        Add a table to the TensorBoard dashboard.

        Parameters
        ----------
        k : str
            The tag associated with the table.
        v : pd.DataFrame
            The table data.
        itr : int
            The iteration number.
        """
        self.backend_logger.add_text(k, v.to_markdown(), itr)

    def add_text(self, k, v, itr):
        """
        Add a text to the TensorBoard dashboard.

        Parameters
        ----------
        k : str
            The tag associated with the text.
        v : str
            The text data.
        itr : int
            The iteration number.
        """
        self.backend_logger.add_text(k, v, itr)

    def add_scalars(self, k, v: dict[str, float | int], itr):
        """
        Add multiple scalars to the TensorBoard dashboard.

        Parameters
        ----------
        k : str
            The main tag associated with the scalars.
        v : dict[str, float | int]
            A dictionary of scalar tags and values.
        itr : int
            The iteration number.
        """
        for _k, _v in v.items():
            self.backend_logger.add_scalar(f"{k}_{_k}", _v, itr)
        # NOTE this is buggy:
        # self.backend_logger.add_scalars(k, v_dict, itr)

    def add_scalar(self, k, v, itr):
        """
        Add a scalar to the TensorBoard dashboard.

        Parameters
        ----------
        k : str
            The tag associated with the scalar.
        v : float | int
            The scalar value.
        itr : int
            The iteration number.
        """
        if v is None:
            self.backend_logger.add_scalar(k, np.nan, itr)
        else:
            self.backend_logger.add_scalar(k, v, itr)

    def write_config(self, config: ConfigBase):
        """
        Write the configuration to the TensorBoard dashboard.

        Parameters
        ----------
        config : ConfigBase
            The configuration object.
        """
        hparams = flatten_nested_dict(config.to_dict())
        run_config = OmegaConf.to_yaml(OmegaConf.create(hparams)).replace("\n", "\n\n")
        self.backend_logger.add_text("config", run_config, 0)

    def _sync(self):
        pass

import logging
import multiprocessing
import struct
import threading
import time
import typing as ty
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from tensorboardX.event_file_writer import EventsWriter
from tensorboardX.proto import event_pb2
from tensorboardX.record_writer import masked_crc32c

from ablator.config.main import ConfigBase
from ablator.config.utils import flatten_nested_dict
from ablator.modules.loggers import LoggerBase

logging.getLogger("tensorboardX").setLevel(logging.ERROR)


class RecordWriter:
    """
    an extension to `tensorboardX.record_writer.RecordWriter` but removes
    support for remote writing.

    Parameters
    ----------
    path : Path
        The path of where to write the records.
    """

    def __init__(self, path: Path):
        self._name_to_tf_name: dict[str, str] = {}
        self._tf_names: set[str] = set()
        self.path = path
        self._writer = None

    def write(self, data):
        with open(self.path, "ab") as writer:
            w = writer.write
            header = struct.pack("Q", len(data))
            w(header)
            w(struct.pack("I", masked_crc32c(header)))
            w(data)
            w(struct.pack("I", masked_crc32c(data)))

    def flush(self):
        ...

    def close(self):
        ...


# Monkey-patching for faster writes to work with mount
# pylint: disable=super-init-not-called,no-member
class MyEventsWriter(EventsWriter):
    """
    Events files have a name of the form
    '/some/file/path/events.out.tfevents.[timestamp].[hostname]'
    """

    def __init__(self, filename):
        self._file_name = filename
        self._num_outstanding_events = 0
        self._py_recordio_writer = RecordWriter(self._file_name)
        # Initialize an event instance.
        self._event = event_pb2.Event()
        self._event.wall_time = time.time()
        self._event.file_version = "brain.Event:2"
        self._lock = threading.Lock()
        self.write_event(self._event)


class TensorboardLogger(LoggerBase):
    """
    A logger class for Tensorboard visualization.

    Parameters
    ----------
    summary_dir : Union[str, Path]
        The directory to store the Tensorboard summary files.

    Attributes
    ----------
    summary_dir : Union[str, Path]
        The directory to store the Tensorboard summary files.
    backend_logger : SummaryWriter
        The PyTorch Tensorboard SummaryWriter object used to log data.
    """

    def __init__(self, summary_dir: Union[str, Path]):
        # Initialize the TensorboardLogger with a summary directory.
        self.thread_lock = threading.Lock()
        self.summary_dir = Path(summary_dir).as_posix()
        self.backend_logger = SummaryWriter(
            log_dir=summary_dir, max_queue=2, flush_secs=2
        )
        fw = self.backend_logger.file_writer.event_writer
        fw.close()
        filename = fw._ev_writer._file_name
        fw._ev_writer = MyEventsWriter(filename)
        fw._event_queue = multiprocessing.Queue(2)
        fw.reopen()

        super().__init__(heartbeat_interval=10)

    def add_image(
        self, k: str, v: np.ndarray, itr: int, dataformats: ty.Optional[str] = "CHW"
    ):
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
        dataformats : ty.Optional[str]
            The format of the image data, by default ``"CHW"``.
        """
        with self.thread_lock:
            self.backend_logger.add_image(k, v, itr, dataformats=dataformats)
            self.backend_logger.flush()

    def add_table(self, k: str, v: pd.DataFrame, itr: int):
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
        with self.thread_lock:
            self.backend_logger.add_text(k, v.to_markdown(), itr)
            self.backend_logger.flush()

    def add_text(self, k: str, v: str, itr: int):
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
        with self.thread_lock:
            self.backend_logger.add_text(k, v, itr)
            self.backend_logger.flush()

    def add_scalars(self, k: str, v: dict[str, float | int], itr: int):
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
        with self.thread_lock:
            for _k, _v in v.items():
                self.backend_logger.add_scalar(f"{k}_{_k}", _v, itr)
            self.backend_logger.flush()
        # NOTE this is buggy:
        # self.backend_logger.add_scalars(k, v_dict, itr)

    def add_scalar(self, k: str, v: float | int, itr: int):
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
        with self.thread_lock:
            if v is None:
                self.backend_logger.add_scalar(k, np.nan, itr)
            else:
                self.backend_logger.add_scalar(k, v, itr)
            self.backend_logger.flush()

    def write_config(self, config: ConfigBase):
        """
        Write the configuration to the TensorBoard dashboard.

        Parameters
        ----------
        config : ConfigBase
            The configuration object.
        """
        with self.thread_lock:
            hparams = flatten_nested_dict(config.to_dict())
            run_config = OmegaConf.to_yaml(OmegaConf.create(hparams)).replace(
                "\n", "\n\n"
            )
            self.backend_logger.add_text("config", run_config, 0)
            self.backend_logger.flush()

    def heartbeat(self, timeout: int | None = None):
        assert timeout is None
        self._sync()

    def _sync(self):
        with self.thread_lock:
            self.backend_logger.flush()

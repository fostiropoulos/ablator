import copy
import json
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

import ablator.utils.file as futils
from ablator.config.proto import RunConfig
from ablator.modules.loggers import LoggerBase
from ablator.modules.loggers.file import FileLogger
from ablator.modules.loggers.tensor import TensorboardLogger
from ablator.modules.metrics.main import Metrics
from ablator.modules.metrics.stores import MovingAverage


class SummaryLogger:
    """
    A logger for training and evaluation summary.

    Attributes
    ----------
    SUMMARY_DIR_NAME : str
        Name of the summary directory.
    RESULTS_JSON_NAME : str
        Name of the results JSON file.
    LOG_FILE_NAME : str
        Name of the log file.
    CONFIG_FILE_NAME : str
        Name of the configuration file.
    METADATA_JSON : str
        Name of the metadata JSON file.
    CHKPT_DIR_NAMES : list[str]
        List of checkpoint directory names.
    CHKPT_DIR_VALUES : list[str]
        List of checkpoint directory values.
    CHKPT_DIRS : dict[str, Path]
        Dictionary containing checkpoint directories.
    keep_n_checkpoints : int
        Number of checkpoints to keep.
    log_iteration : int
        Current log iteration.
    checkpoint_iteration : dict[str, dict[str, int]]
        ``checkpoint_iteration`` is a dictionary that keeps track of the checkpoint iterations for each directory.
        It is used in the ``checkpoint()`` method
        to determine the appropriate iteration number for the saved checkpoint.
    log_file_path : Path | None
        Path to the log file.
    dashboard : LoggerBase | None
        Dashboard logger.
    experiment_dir : Path | None
        the trial directory.
    result_json_path : Path | None
        Path to the results JSON file.
    """

    SUMMARY_DIR_NAME = "dashboard"
    RESULTS_JSON_NAME = "results.json"
    LOG_FILE_NAME = "train.log"
    CONFIG_FILE_NAME = "config.yaml"
    METADATA_JSON = "metadata.json"
    CHKPT_DIR_NAMES = ["best", "recent"]
    CHKPT_DIR_VALUES = ["best_checkpoints", "checkpoints"]
    CHKPT_DIRS: dict[str, Path]

    def __init__(
        self,
        run_config: RunConfig,
        experiment_dir: str | None | Path = None,
        resume: bool = False,
        keep_n_checkpoints: int | None = None,
        verbose: bool = True,
    ):
        """
        Initialize a SummaryLogger.

        Parameters
        ----------
        run_config : RunConfig
            The run configuration.
        experiment_dir : str | None | Path, optional
            Path to the trial directory, by default ``None``.
        resume : bool, optional
            Whether to resume from an existing model directory, by default ``False``.
        keep_n_checkpoints : int | None, optional
            Number of checkpoints to keep, by default ``None``.
        verbose : bool, optional
            Whether to print messages to the console, by default ``True``.
        """

        run_config = copy.deepcopy(run_config)
        self.uid = run_config.uid
        self.keep_n_checkpoints: int = (
            keep_n_checkpoints if keep_n_checkpoints is not None else int(1e6)
        )
        self.log_iteration: int = 0
        self.checkpoint_iteration: dict[str, dict[str, int]] = {}
        self.log_file_path: Path | None = None
        self.dashboard: LoggerBase | None = None
        self.experiment_dir: Path | None = None
        self.result_json_path: Path | None = None
        self.CHKPT_DIRS = {}
        if experiment_dir is not None:
            self.experiment_dir = Path(experiment_dir)
            if not resume and self.experiment_dir.exists():
                raise FileExistsError(
                    f"SummaryLogger: Resume is set to {resume} but {self.experiment_dir} exists."
                )
            if resume and self.experiment_dir.exists():
                _run_config = type(run_config).load(
                    self.experiment_dir.joinpath(self.CONFIG_FILE_NAME)
                )
                run_config.train_config.assert_state(_run_config.train_config)
                run_config.model_config.assert_state(_run_config.model_config)
                metadata = json.loads(
                    self.experiment_dir.joinpath(self.METADATA_JSON).read_text(
                        encoding="utf-8"
                    )
                )
                self.checkpoint_iteration = metadata["checkpoint_iteration"]
                self.log_iteration = metadata["log_iteration"]

            (self.summary_dir, *chkpt_dirs) = futils.make_sub_dirs(
                experiment_dir, self.SUMMARY_DIR_NAME, *self.CHKPT_DIR_VALUES
            )
            for name, path in zip(self.CHKPT_DIR_NAMES, chkpt_dirs):
                self.CHKPT_DIRS[name] = path

            self.result_json_path = self.experiment_dir / self.RESULTS_JSON_NAME
            self.log_file_path = self.experiment_dir.joinpath(self.LOG_FILE_NAME)
            self.dashboard = self._make_dashboard(self.summary_dir, run_config)
            self._write_config(run_config)
            self._update_metadata()
        self.logger = FileLogger(path=self.log_file_path, verbose=verbose)

    def _update_metadata(self):
        """
        Update the metadata file.
        """
        if self.experiment_dir is None:
            return
        metadata_path = self.experiment_dir.joinpath(self.METADATA_JSON)
        metadata_path.write_text(
            json.dumps(
                {
                    "log_iteration": self.log_iteration,
                    "checkpoint_iteration": self.checkpoint_iteration,
                }
            ),
            encoding="utf-8",
        )

    def _make_dashboard(
        self, summary_dir: Path, run_config: RunConfig | None = None
    ) -> LoggerBase | None:
        """
        Make a dashboard logger.

        Parameters
        ----------
        summary_dir : Path
            Path to the summary directory.
        run_config : RunConfig | None, optional
            The run configuration, by default None.

        Returns
        -------
        LoggerBase | None
            A TensorboardLogger.
        """
        if run_config is None or not run_config.tensorboard:
            return None
        return TensorboardLogger(summary_dir.joinpath("tensorboard"))

    def _write_config(self, run_config: RunConfig):
        """
        Write the run configuration to the model directory and to the dashboard.

        Parameters
        ----------
        run_config : RunConfig
            The run configuration.
        """
        if self.experiment_dir is None:
            return
        self.experiment_dir.joinpath(self.CONFIG_FILE_NAME).write_text(
            str(run_config), encoding="utf-8"
        )

        if self.dashboard is not None:
            self.dashboard.write_config(run_config)

    # pylint: disable=too-complex
    def _add_metric(self, k, v, itr):
        """
        Add a metric to the dashboard.

        Parameters
        ----------
        k : str
            The metric name.
        v : Any
            The metric value.
        itr : int
            The iteration.
        """
        if self.dashboard is None:
            return
        if isinstance(v, (list, np.ndarray)):
            v = np.array(v)
            if v.dtype.kind in {"b", "i", "u", "f", "c"}:
                v_dict = {str(i): _v for i, _v in enumerate(v)}
                self.dashboard.add_scalars(k, v_dict, itr)

            else:
                self.dashboard.add_text(k, " ".join(v), itr)
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                self.dashboard.add_scalar(f"{k}_{sub_k}", sub_v, itr)
        elif isinstance(v, MovingAverage):
            self.dashboard.add_scalar(k, v.get(), itr)
        elif isinstance(v, str):
            self.dashboard.add_text(k, v, itr)

        elif isinstance(v, Image.Image):
            self.dashboard.add_image(
                k, np.array(v).transpose(2, 0, 1), itr, dataformats="CHW"
            )
        elif isinstance(v, pd.DataFrame):
            self.dashboard.add_table(k, v, itr)
        elif isinstance(v, (int, float)):
            self.dashboard.add_scalar(k, v, itr)
        else:
            raise ValueError(
                f"Unsupported dashboard value {v}. Must be "
                "[int,float, pd.DataFrame, Image.Image, str, "
                "MovingAverage, dict[str,float|int], list[float,int], np.ndarray] "
            )

    def _append_metrics(self, metrics: dict[str, float]):
        """Append metrics to the result json file.

        Parameters
        ----------
        metrics : dict[str, float]
            The metrics to append.
        """
        if self.result_json_path is not None:
            _metrics = copy.deepcopy(metrics)
            _metrics["timestamp"] = float(time.time())
            _metrics_str = futils.dict_to_json(_metrics)
            if self.result_json_path.exists():
                futils.truncate_utf8_chars(self.result_json_path, "]")
                with open(self.result_json_path, "a", encoding="utf-8") as fp:
                    fp.write(",\n" + _metrics_str + "]")
            else:
                self.result_json_path.write_text(f"[{_metrics_str}]", encoding="utf-8")

    def update(
        self,
        metrics: Union[Metrics, dict],
        itr: Optional[int] = None,
    ):
        """Update the dashboard with the given metrics.
        write some metrics to json files and update the current metadata (``log_iteration``)

        Parameters
        ----------
        metrics : Union[Metrics, dict]
            The metrics to update.
        itr : Optional[int], optional
            The iteration, by default ``None``.

        Raises
        ------
        AssertionError
            If the iteration is not greater than the current iteration.

        Notes
        -----
        Attribute ``log_iteration`` is increased by 1 every time ``update()`` is called while training models.
        """

        if itr is None:
            itr = self.log_iteration
            self.log_iteration += 1
        else:
            assert (
                itr > self.log_iteration
            ), f"Current iteration > {itr}. Can not add metrics."
            self.log_iteration = itr
        if isinstance(metrics, Metrics):
            dict_metrics = metrics.to_dict()
        else:
            dict_metrics = metrics

        for k, v in dict_metrics.items():
            self._add_metric(k, v, itr)
        self._append_metrics(dict_metrics)
        self._update_metadata()

    def checkpoint(
        self,
        save_dict: object,
        file_name: str,
        itr: int | None = None,
        is_best: bool = False,
    ):
        """Save a checkpoint and update the checkpoint iteration

        Saves the model checkpoint in the appropriate directory based on the ``is_best`` parameter.
        If ``is_best==True``, the checkpoint is saved in the ``"best"`` directory, indicating the best
        performing model so far. Otherwise, the checkpoint is saved in the ``"recent"`` directory,
        representing the most recent checkpoint.

        The file path for the checkpoint is constructed using the selected directory name (``"best"`` or
        ``"recent"``), and the file name with the format ``"{file_name}_{itr:010}.pt"``, where ``itr`` is the
        iteration number.

        The ``checkpoint_iteration`` dictionary is updated with the current iteration number for each
        directory. If ``itr`` is not provided, the iteration number is increased by 1 each time a
        checkpoint is saved. Otherwise, the iteration number is set to the provided ``itr``.

        Parameters
        ----------
        save_dict : object
            The object to save.

        file_name : str
            The file name.

        itr : int | None, optional
            The iteration, by default None. If not provided, the current iteration is incremented by 1.

        is_best : bool, optional
            Whether this is the best checkpoint, by default False.

        Raises
        ------
        AssertionError
            If the provided ``itr`` is not larger than the current iteration associated with the checkpoint.
        """
        if self.experiment_dir is None:
            return
        dir_name = "best" if is_best else "recent"
        if self.keep_n_checkpoints > 0:
            if dir_name not in self.checkpoint_iteration:
                self.checkpoint_iteration[dir_name] = {}
            if file_name not in self.checkpoint_iteration[dir_name]:
                self.checkpoint_iteration[dir_name][file_name] = -1
            if itr is None:
                self.checkpoint_iteration[dir_name][file_name] += 1
                itr = self.checkpoint_iteration[dir_name][file_name]
            else:
                cur_iter = self.checkpoint_iteration[dir_name][file_name]
                assert (
                    itr > cur_iter
                ), f"Checkpoint iteration {cur_iter} >= training iteration {itr}. Can not overwrite checkpoint."
                self.checkpoint_iteration[dir_name][file_name] = itr

            dir_path = self.experiment_dir.joinpath(self.CHKPT_DIRS[dir_name])

            file_path = dir_path.joinpath(f"{file_name}_{itr:010}.pt")

            assert not file_path.exists(), f"Checkpoint exists: {file_path}"
            futils.save_checkpoint(save_dict, file_path)
            futils.clean_checkpoints(dir_path, self.keep_n_checkpoints)
            self._update_metadata()

    def clean_checkpoints(self, keep_n_checkpoints: int):
        """
        Clean up checkpoints and keep only the specified number of checkpoints.

        Parameters
        ----------
        keep_n_checkpoints : int
            Number of checkpoints to keep.
        """
        if self.experiment_dir is None:
            return
        for chkpt_dir in self.CHKPT_DIR_VALUES:
            dir_path = self.experiment_dir.joinpath(chkpt_dir)
            futils.clean_checkpoints(dir_path, keep_n_checkpoints)

    def info(self, *args, **kwargs):
        """
        Log an info to files and to console message using the logger.

        Parameters
        ----------
        *args
            Positional arguments passed to the logger's info method.
        **kwargs
            Keyword arguments passed to the logger's info method.
        """
        self.logger.info(*args, **kwargs)

    def warn(self, *args, **kwargs):
        """
        Log a warning message to files and to console using the logger.

        Parameters
        ----------
        *args
            Positional arguments passed to the logger's warn method.
        **kwargs
            Keyword arguments passed to the logger's warn method.
        """
        self.logger.warn(*args, **kwargs)

    def error(self, *args, **kwargs):
        """
        Log an error message to files and to console using the logger.

        Parameters
        ----------
        *args
            Positional arguments passed to the logger's error method.
        **kwargs
            Keyword arguments passed to the logger's error method.
        """
        self.logger.error(*args, **kwargs)

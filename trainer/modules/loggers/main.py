from collections import defaultdict
import copy
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
from trainer.main.configs import RunConfig
from trainer.modules.loggers import LoggerBase
from trainer.modules.loggers.file import FileLogger
from trainer.modules.loggers.tensor import TensorboardLogger
from trainer.modules.metrics.main import TrainMetrics
from trainer.modules.metrics.stores import MovingAverage
import trainer.utils.file as futils
from trainer.config.main import ConfigBase
import json


class DuplicateRunError(Exception):
    pass


class SummaryLogger:
    SUMMARY_DIR_NAME = "dashboard"
    RESULTS_JSON_NAME = "results.json"
    LOG_FILE_NAME = "train.log"
    CONFIG_FILE_NAME = "config.yaml"
    METADATA_JSON = "metadata.json"
    CHKPT_DIR_NAMES = ["best", "recent"]
    CHKPT_DIR_VALUES = ["best_checkpoints", "checkpoints"]
    CHKPT_DIRS: Dict[str, Path]

    def __init__(
        self,
        run_config: RunConfig | None = None,
        model_dir: str | None | Path = None,
        resume: bool = False,
        keep_n_checkpoints: int | None = None,
        verbose: bool = True,
    ):
        run_config = copy.deepcopy(run_config)
        self.uid = run_config.uid
        self.keep_n_checkpoints: int = (
            keep_n_checkpoints if keep_n_checkpoints is not None else float("inf")
        )
        self.log_iteration: int = 0
        self.checkpoint_iteration: dict[str, dict[str, int]] = {}

        if model_dir is not None:
            self.model_dir: Path = Path(model_dir)
            if not resume and self.model_dir.exists():
                raise DuplicateRunError(
                    f"SummaryLogger: Resume is set to {resume} but {self.model_dir} exists."
                )
            elif resume and self.model_dir.exists():
                _run_config = type(run_config).load(
                    self.model_dir.joinpath(self.CONFIG_FILE_NAME)
                )
                assert (
                    run_config.train_config == _run_config.train_config
                    and run_config.model_config == _run_config.model_config
                ), f"Different supplied run_config than existing run_config {self.model_dir}. \n{run_config}----\n{_run_config}"

                metadata = json.loads(
                    self.model_dir.joinpath(self.METADATA_JSON).read_text()
                )
                self.checkpoint_iteration = metadata["checkpoint_iteration"]
                self.log_iteration = metadata["log_iteration"]

            (self.summary_dir, *chkpt_dirs) = futils.make_sub_dirs(
                model_dir, self.SUMMARY_DIR_NAME, *self.CHKPT_DIR_VALUES
            )
            self.CHKPT_DIRS = {}
            for name, path in zip(self.CHKPT_DIR_NAMES, chkpt_dirs):
                self.CHKPT_DIRS[name] = path

            self.result_json_path = self.model_dir / self.RESULTS_JSON_NAME
            self.log_file_path = self.model_dir.joinpath(self.LOG_FILE_NAME)
        self.dashboard: LoggerBase | None = self._make_dashboard(
            self.summary_dir, run_config
        )
        self.logger = FileLogger(path=self.log_file_path.as_posix(), verbose=verbose)
        self._write_config(run_config)
        self._update_metadata()

    def _update_metadata(self):
        metadata_path = self.model_dir.joinpath(self.METADATA_JSON)
        metadata_path.write_text(
            json.dumps(
                {
                    "log_iteration": self.log_iteration,
                    "checkpoint_iteration": self.checkpoint_iteration,
                }
            )
        )

    def _make_dashboard(
        self, summary_dir: Path, run_config: RunConfig | None = None
    ) -> LoggerBase | None:
        if run_config is None or not run_config.tensorboard:
            return None
        return TensorboardLogger(summary_dir.joinpath("tensorboard"))

    def _write_config(self, run_config: RunConfig):
        with open(self.model_dir.joinpath(self.CONFIG_FILE_NAME), "w") as fp:
            fp.write(str(run_config))
        if self.dashboard is not None:
            self.dashboard._write_config(run_config)

    def _add_metric(self, k, v, itr):
        if self.dashboard is None:
            return
        if isinstance(v, (list,np.ndarray)):
            v = np.array(v)
            if v.dtype.kind in ["b", "i", "u", "f", "c"]:
                v_dict = {str(i): _v for i, _v in enumerate(v)}
                self.dashboard._add_scalars(k, v_dict, itr)

            else:
                self.dashboard._add_text(k, " ".join(v), itr)
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                self.dashboard._add_scalar(f"{k}_{sub_k}", sub_v, itr)
        elif isinstance(v, MovingAverage):
            self.dashboard._add_scalar(k, v.get(), itr)
        elif isinstance(v, str):
            self.dashboard._add_text(k, v, itr)

        elif isinstance(v, Image.Image):
            self.dashboard._add_image(
                k, np.array(v).transpose(2, 0, 1), itr, dataformats="CHW"
            )
        elif isinstance(v, pd.DataFrame):
            self.dashboard._add_table(k, v, itr)
        elif isinstance(v, (int, float)):
            self.dashboard._add_scalar(k, v, itr)
        else:
            raise ValueError(f"Unsupported dashboard value {v}. Must be [int,float, pd.DataFrame, Image.Image, str, MovingAverage, dict[str,float|int], list[float,int], np.ndarray] ")

    def _append_metrics(self, metrics):
        if self.result_json_path is not None:
            with open(self.result_json_path, "a") as fp:
                fp.write(futils.dict_to_json(metrics) + "\n")

    def update(
        self,
        metrics: Union[TrainMetrics, Dict],
        aux_metrics: dict | None = None,
        itr: Optional[int] = None,
    ):
        if itr is None:
            itr = self.log_iteration
            self.log_iteration += 1
        else:
            assert (
                itr > self.log_iteration
            ), f"Current iteration > {itr}. Can not add metrics."
            self.log_iteration = itr
        if isinstance(metrics, TrainMetrics):
            dict_metrics = metrics.to_dict()
        else:
            dict_metrics = metrics

        for k, v in dict_metrics.items():
            self._add_metric(k, v, itr)
        if aux_metrics is not None:
            for k, v in aux_metrics.items():
                self._add_metric(k, v, itr)

        self._update_metadata()

    def checkpoint(
        self,
        save_dict: object,
        file_name: str,
        itr: int | None = None,
        dir_name: Union[Literal["recent", "best"], str] = "recent",
    ):
        if self.keep_n_checkpoints > 0:
            if dir_name not in self.checkpoint_iteration:
                self.checkpoint_iteration[dir_name] = {}
            if file_name not in self.checkpoint_iteration[dir_name]:
                self.checkpoint_iteration[dir_name][file_name] = -1
            if itr is None:
                self.checkpoint_iteration[dir_name][file_name] += 1
                itr = self.checkpoint_iteration[dir_name][file_name]
            else:
                assert (
                    itr > self.checkpoint_iteration[dir_name][file_name]
                ), f"Current iteration > {itr}. Can not save checkpoint."
                self.checkpoint_iteration[dir_name][file_name] = itr

            dir_path = self.model_dir.joinpath(self.CHKPT_DIRS[dir_name])

            file_path = dir_path.joinpath(f"{file_name}_{itr:010}.pt")

            assert not file_path.exists(), f"Checkpoint exists: {file_path}"
            futils.save_checkpoint(save_dict, file_path)
            futils.clean_checkpoints(dir_path, self.keep_n_checkpoints)
            self._update_metadata()

    def clean_checkpoints(self, keep_n_checkpoints: int):
        for chkpt_dir in self.CHKPT_DIR_VALUES:
            dir_path = self.model_dir.joinpath(chkpt_dir)
            futils.clean_checkpoints(dir_path, keep_n_checkpoints)

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def warn(self, *args, **kwargs):
        self.logger.warn(*args, **kwargs)

    def error(self, *args, **kwargs):
        self.logger.error(*args, **kwargs)

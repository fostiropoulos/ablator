import copy
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

import trainer.modules.model.utils as mutils
import trainer.utils.file as futils
from trainer.config.main import ConfigBase
from trainer.config.run import RunConfig
from trainer.modules.logging.base import LoggerBase
from trainer.modules.logging.file import FileLogger
from trainer.modules.main import Metrics, MovingAverage


class SummaryLogger:

    SUMMARY_DIR_NAME = "summary"
    RESULTS_JSON_NAME = "results.json"
    LOG_FILE_NAME = "train.log"
    CONFIG_FILE_NAME = "config.yaml"
    CHKPT_DIR_NAMES = ["best", "recent"]
    CHKPT_DIR_VALUES = ["best_checkpoints", "checkpoints"]
    CHKPT_DIRS: Dict[str, Path]

    def __init__(
        self,
        run_config: RunConfig,
        model_dir: Optional[str] = None,
        is_slave: bool = False,
        resume: Optional[bool] = None,
    ):

        run_config = copy.deepcopy(run_config)
        self.loggers: List[LoggerBase] = []
        train_config = run_config.train_config
        self.uid = run_config.uid
        self.keep_n_checkpoints: int = train_config.keep_n_checkpoints
        self.resume: bool = train_config.resume if resume is None else resume
        self.is_slave: bool = is_slave
        if model_dir is not None:
            self.model_dir: Path = Path(model_dir)

            (self.summary_dir, *chkpt_dirs) = futils.make_sub_dirs(
                model_dir, self.SUMMARY_DIR_NAME, *self.CHKPT_DIR_VALUES
            )
            self.CHKPT_DIRS = {}
            for name, path in zip(self.CHKPT_DIR_NAMES, chkpt_dirs):
                self.CHKPT_DIRS[name] = path

            self.result_json_path = self.model_dir / self.RESULTS_JSON_NAME
            self.log_file_path = self.model_dir.joinpath(self.LOG_FILE_NAME)
        self.verbose = False if self.is_slave else train_config.verbose
        if run_config.logger_configs is not None and not self.is_slave:
            self.loggers = self._make_loggers(
                run_config, self.summary_dir.as_posix(), resume=self.resume
            )

        self.logger = FileLogger(
            path=self.log_file_path.as_posix(), verbose=self.verbose
        )
        if self.is_slave:
            self.logger.set_prefix(f"Device Rank: {run_config.rank} - ")

    @classmethod
    def _make_loggers(
        cls, run_config: RunConfig, summary_dir: str, resume=False
    ) -> List[LoggerBase]:

        loggers = []
        if run_config.logger_configs is None:
            return loggers

        for logger_config in run_config.logger_configs:
            loggers.append(
                logger_config.arguments.make(
                    resume,
                    run_config.uid,
                    summary_dir,
                )
            )
        return loggers

    def _write_config(self, run_config: ConfigBase):
        futils.write_yaml(
            run_config.to_dict(),
            self.model_dir.joinpath(self.CONFIG_FILE_NAME),
            clean_dict=False,
        )

    def _add_metric(self, k, v, itr):
        if isinstance(v, list):
            v = np.array(v)
            if v.dtype.kind in ["b", "i", "u", "f", "c"]:
                v_dict = {i: _v for i, _v in enumerate(v)}
                self.add_scalars(k, v_dict, itr)
            else:
                self.add_text(k, " ".join(v), itr)
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                self.add_scalar(f"{k}_{sub_k}", sub_v, itr)
        elif isinstance(v, MovingAverage):
            self.add_scalar(k, v.get(), itr)
        elif isinstance(v, str):
            self.add_text(k, v, itr)
        elif isinstance(v, Image.Image):
            self.add_image(k, np.array(v).transpose(2, 0, 1), itr)
        else:
            self.add_scalar(k, v, itr)

    def add_image(self, k, v, itr, dataformats="CHW"):
        for logger in self.loggers:
            logger._add_image(k, v, itr, dataformats=dataformats)

    def add_table(self, k, v: pd.DataFrame, itr):
        for logger in self.loggers:
            logger._add_table(k, v, itr)

    def add_text(self, k, v, itr):
        for logger in self.loggers:
            logger._add_text(k, v, itr)

    def add_scalars(self, k, v, itr):
        for logger in self.loggers:
            logger._add_scalars(k, v, itr)

    def add_scalar(self, k, v, itr):
        for logger in self.loggers:
            logger._add_scalar(k, v, itr)

    def _append_metrics(self, metrics):
        if self.result_json_path is not None:
            with open(self.result_json_path, "a") as fp:
                fp.write(futils.dict_to_json(metrics) + "\n")

    def _update_dict(self, metrics_dict: Dict):
        self._append_metrics(metrics_dict)
        if "current_iteration" not in metrics_dict:
            return
        itr = metrics_dict["current_iteration"]
        if "curret_loss" in metrics_dict:
            current_loss = metrics_dict["current_loss"]
            self.add_scalar("current_loss", current_loss, itr)
        if "lr" in metrics_dict:
            lr = metrics_dict["lr"]
            self.add_scalar("learning_rate", lr, itr)

    def _update_obj(self, metrics: Metrics):
        aux_metrics = metrics.get_added_metrics()
        itr = metrics.current_iteration
        for k, v in aux_metrics.items():
            self._add_metric(k, v, itr)
        metrics_dict = metrics.to_dict()
        self._update_dict(metrics_dict)

    def update(self, metrics: Union[Metrics, Dict]):
        if isinstance(metrics, Metrics):
            self._update_obj(metrics)
        else:
            metrics_dict = copy.deepcopy(metrics)
            self._update_dict(metrics_dict)

    def checkpoint(
        self,
        save_dict,
        file_name,
        dir_name: Union[Literal["recent", "best"], str] = "recent",
    ):

        if not self.is_slave and self.keep_n_checkpoints > 0:
            dir_path = (
                self.CHKPT_DIRS[dir_name]
                if dir_name in self.CHKPT_DIRS
                else futils.make_sub_dirs(self.model_dir, dir_name)[0]
            )
            file_path = dir_path.joinpath(file_name)
            mutils.save_checkpoint(save_dict, file_path)
            futils.clean_checkpoints(dir_path, self.keep_n_checkpoints)

    def write_config(self, config: ConfigBase):
        for logger in self.loggers:
            logger._write_config(config)
        self._write_config(config)



    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def warn(self, *args, **kwargs):
        self.logger.warn(*args, **kwargs)

    def error(self, *args, **kwargs):
        self.logger.error(*args, **kwargs)

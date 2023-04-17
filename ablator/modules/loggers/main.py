import copy
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

import ablator.utils.file as futils
from ablator.main.configs import RunConfig
from ablator.modules.loggers import LoggerBase
from ablator.modules.loggers.file import FileLogger
from ablator.modules.loggers.tensor import TensorboardLogger
from ablator.modules.metrics.main import TrainMetrics
from ablator.modules.metrics.stores import MovingAverage


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
    CHKPT_DIRS: dict[str, Path]

    def __init__(
        self,
        run_config: RunConfig,
        model_dir: str | None | Path = None,
        resume: bool = False,
        keep_n_checkpoints: int | None = None,
        verbose: bool = True,
    ):
        run_config = copy.deepcopy(run_config)
        self.uid = run_config.uid
        self.keep_n_checkpoints: int = (
            keep_n_checkpoints if keep_n_checkpoints is not None else int(1e6)
        )
        self.log_iteration: int = 0
        self.checkpoint_iteration: dict[str, dict[str, int]] = {}
        self.log_file_path: Path | None = None
        self.dashboard: LoggerBase | None = None
        self.model_dir: Path | None = None
        self.result_json_path: Path | None = None
        self.CHKPT_DIRS = {}
        if model_dir is not None:
            self.model_dir = Path(model_dir)
            if not resume and self.model_dir.exists():
                raise DuplicateRunError(
                    f"SummaryLogger: Resume is set to {resume} but {self.model_dir} exists."
                )
            if resume and self.model_dir.exists():
                _run_config = type(run_config).load(
                    self.model_dir.joinpath(self.CONFIG_FILE_NAME)
                )
                assert (
                    run_config.train_config == _run_config.train_config
                    and run_config.model_config == _run_config.model_config
                ), (
                    "Different supplied run_config than"
                    f" existing run_config {self.model_dir}. \n{run_config}----\n{_run_config}"
                )

                metadata = json.loads(
                    self.model_dir.joinpath(self.METADATA_JSON).read_text(
                        encoding="utf-8"
                    )
                )
                self.checkpoint_iteration = metadata["checkpoint_iteration"]
                self.log_iteration = metadata["log_iteration"]

            (self.summary_dir, *chkpt_dirs) = futils.make_sub_dirs(
                model_dir, self.SUMMARY_DIR_NAME, *self.CHKPT_DIR_VALUES
            )
            for name, path in zip(self.CHKPT_DIR_NAMES, chkpt_dirs):
                self.CHKPT_DIRS[name] = path

            self.result_json_path = self.model_dir / self.RESULTS_JSON_NAME
            self.log_file_path = self.model_dir.joinpath(self.LOG_FILE_NAME)
            self.dashboard = self._make_dashboard(self.summary_dir, run_config)
            self._write_config(run_config)
            self._update_metadata()
        self.logger = FileLogger(path=self.log_file_path, verbose=verbose)

    def _update_metadata(self):
        if self.model_dir is None:
            return
        metadata_path = self.model_dir.joinpath(self.METADATA_JSON)
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
        if run_config is None or not run_config.tensorboard:
            return None
        return TensorboardLogger(summary_dir.joinpath("tensorboard"))

    def _write_config(self, run_config: RunConfig):
        if self.model_dir is None:
            return
        self.model_dir.joinpath(self.CONFIG_FILE_NAME).write_text(
            str(run_config), encoding="utf-8"
        )

        if self.dashboard is not None:
            self.dashboard.write_config(run_config)

    def _add_metric(self, k, v, itr):
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
        if self.result_json_path is not None:
            with open(self.result_json_path, "a", encoding="utf-8") as fp:
                fp.write(futils.dict_to_json(metrics) + "\n")

    def update(
        self,
        metrics: Union[TrainMetrics, dict],
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
        self._append_metrics(dict_metrics)
        self._update_metadata()

    def checkpoint(
        self,
        save_dict: object,
        file_name: str,
        itr: int | None = None,
        is_best: bool = False,
    ):
        if self.model_dir is None:
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
                ), f"Checkpoint iteration {cur_iter} > training iteration {itr}. Can not save checkpoint."
                self.checkpoint_iteration[dir_name][file_name] = itr

            dir_path = self.model_dir.joinpath(self.CHKPT_DIRS[dir_name])

            file_path = dir_path.joinpath(f"{file_name}_{itr:010}.pt")

            assert not file_path.exists(), f"Checkpoint exists: {file_path}"
            futils.save_checkpoint(save_dict, file_path)
            futils.clean_checkpoints(dir_path, self.keep_n_checkpoints)
            self._update_metadata()

    def clean_checkpoints(self, keep_n_checkpoints: int):
        if self.model_dir is None:
            return
        for chkpt_dir in self.CHKPT_DIR_VALUES:
            dir_path = self.model_dir.joinpath(chkpt_dir)
            futils.clean_checkpoints(dir_path, keep_n_checkpoints)

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def warn(self, *args, **kwargs):
        self.logger.warn(*args, **kwargs)

    def error(self, *args, **kwargs):
        self.logger.error(*args, **kwargs)

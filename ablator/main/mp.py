import copy
import multiprocessing as mp
import os
import shutil
import socket
import subprocess
import sys
import traceback
import types as tys
import typing as ty
from pathlib import Path
import builtins
import json

import numpy as np
import ray
import torch

import ablator as ablator_module
from ablator.main.configs import ParallelConfig, allowed_rclone_remote_configs
from ablator.main.model.main import CheckpointNotFoundError, TrainPlateauError
from ablator.main.model.wrapper import ModelWrapper
from ablator.main.proto import ProtoTrainer
from ablator.main.state import ExperimentState, TrialState
from ablator.modules.loggers.file import FileLogger
from ablator.modules.loggers.main import DuplicateRunError
from ablator.modules.metrics.main import LossDivergedError
from ablator.utils.base import get_gpu_max_mem
import ablator.utils.base as butils
import pyrclone
import time
# The exceptions that are unrecoverable i.e.  [DuplicateRunError]
CRASH_EXCEPTION_TYPES: list[type] = []


def parse_metrics(optim_direction: list[str], metrics: dict[str, float] | None):
    """
    Parse metrics to be optimized.

    Parameters
    ----------
    optim_direction: list[str]
        The metrics to be optimized, defined in the ``ParallelConfig``.
    metrics: dict[str, float]
        The metrics returned after a ray job finishes.

    Returns
    -------
    dict[str, float]
        A dictionary of metric names and their corresponding metric values.
    """
    return (
        {k: v for k, v in metrics.items() if k in optim_direction}
        if metrics is not None
        else None
    )


def evaluate_remote(model: ModelWrapper, eval_config: ParallelConfig, logger: FileLogger):
    metrics = model.evaluate(eval_config)
    metrics_dict = {k: v.to_dict() for k, v in metrics.items()}
    logger.info(f"Evaluation: {butils.parse_dict_to_str(metrics_dict)}")
    with open(eval_config.experiment_dir/"metrics.json", "w", encoding="utf-8") as f:
        formatter_str = json.dumps(metrics_dict, indent=4)
        f.write(formatter_str)
    return metrics_dict, eval_config


def train_main_remote(
    model: ModelWrapper,
    run_config: ParallelConfig,
    mp_logger: FileLogger,
    root_dir: Path,
    fault_tollerant: bool = True,
    crash_exceptions_types: list[type] | None = None,
    resume: bool = False,
    clean_reset: bool = False,
) -> tuple[ParallelConfig, dict[str, float] | None, TrialState]:
    """
    The trial job that will be executed remotely at a ray node. This is where model training happens.
    In addition, experiment directory will be synchronized to the Google Cloud storage and remote nodes.
    Synchronization is done via GcpConfig and RemoteConfig ``rsync_up()`` methods. Refer to documentation of
    these 2 classes for more details.

    Parameters
    ----------
    model : ModelWrapper
        The ModelWrapper that is used to train a model.
    run_config : ParallelConfig
        Runtime configuration for this trial.
    mp_logger : FileLogger
        The file logger that's used to log training progress.
    root_dir : Path
        The root directory that stores experiment states (experiment directory).
    fault_tollerant : bool, optional, default=True
        Whether to tollerate crashes, aka to cease execution when the ray job crashes.
    crash_exceptions_types : list[type], None, optional, default=None
        Types of exceptions that are considered as crashes.
    resume : bool, default=False
        Whether to resume training the model from existing checkpoints and existing experiment state.
    clean_reset : bool, default=False
        Whether to remove model directory when ``CheckpointNotFoundError`` is raised.

    Returns
    -------
    ParallelConfig
        Running configuration of the trial.
    dict[str, float], None
        If exception raised (Except for LossDivergedError and TrainPlateauError),
        this will be ``None`` object. Otherwise, this will be a dictionary of metrics.
    TrialState
        A TrialState object indicating the state of the trial job

        - If ``LossDivergedError`` or ``TrainPlateauError`` is raised while training,
          returned state will be ``TrialState.PRUNED_POOR_PERFORMANCE``

        - If ``DuplicateRunError``, ``RuntimeError`` (with message ``'CUDA out of memory'``),
          or ``CheckpointNotFoundError`` (with ``clean_reset=True``) is raised while training,
          returned state will be ``TrialState.RECOVERABLE_ERROR``

        - If other types of error or ``CheckpointNotFoundError`` (with ``clean_reset=False``) is raised,
          returned state will be ``TrialState.FAIL``

    """
    if crash_exceptions_types is None:
        crash_exceptions_types = []

    def handle_exception(e):
        exception_str = traceback.format_exc()
        if hasattr(model, "logger"):
            model.logger.error(exception_str)
        mp_logger.error(f"Error Occured {run_config.uid}")
        traceback.print_exc()
        if not fault_tollerant or isinstance(e, tuple(crash_exceptions_types)):
            error_msg = (
                f"Error {type(e).__name__} in"
                f"{' '.join([c.__name__ for c in crash_exceptions_types])}. Exiting."
            )
            mp_logger.error(error_msg)
            raise type(e)(error_msg)

        return run_config, None, TrialState.FAIL

    try:
        res = model.train(run_config, resume=resume)
        mp_logger.info(f"Finished training - {run_config.uid}")
        return run_config, res.to_dict(), TrialState.COMPLETE
    except (LossDivergedError, TrainPlateauError):
        return (
            run_config,
            model.metrics.to_dict(),
            TrialState.PRUNED_POOR_PERFORMANCE,
        )
    except DuplicateRunError:
        return (
            run_config,
            None,
            TrialState.RECOVERABLE_ERROR,
        )
    except CheckpointNotFoundError:
        if clean_reset:
            if model.model_dir is not None:
                shutil.rmtree(model.model_dir.as_posix())
            return (
                run_config,
                None,
                TrialState.RECOVERABLE_ERROR,
            )

        return (
            run_config,
            None,
            TrialState.FAIL,
        )
    except RuntimeError as e:
        if str(e).startswith("CUDA out of memory."):
            mp_logger.warn(f"Cuda out of memory for {run_config.uid}. Restarting...")
            return (
                run_config,
                None,
                TrialState.RECOVERABLE_ERROR,
            )

        return handle_exception(e)

    except builtins.Exception as e:
        return handle_exception(e)


class ParallelTrainer(ProtoTrainer):
    """
    A class for parallelizing training of models of different configurations with ray.
    Metrics of these models are for optuna to tune hyperparameters. They are also logged to optuna storage.

    Attributes
    ----------
    run_config : ParallelConfig
        Running configuration for parallel training.
    device : str
        The device to use for training.
    experiment_dir : Path
        The directory that stores experiment information (optuna storage, experiment state database).
    logger : FileLogger
        The logger that writes messages to a file and prints them to the console.
    experiment_state : ExperimentState
        This attribute manages optuna trials.
    total_trials : int
        Number of trials to run.
    gpu_mem_bottleneck : int
        The minimum memory capacity of all available gpus.
    cpu : float
        The number of cpu used per trial.
    gpu : float
        The number of gpu used per trial.
    total_mem_usage : int
        Total amount of memory usage.

    """

    def __init__(self, *args, run_config: ParallelConfig, **kwargs):
        """
        Initialize ``ParallelTrainer`` using config from ``run_config``.

        Parameters
        ----------
        run_config : ParallelConfig
            The runtime configuration for this trainer.
        *args : tuple
            Extra arguments used for ``ProtoTrainer``
        **kwargs : dict, optional
            Extra arguments to  ``ProtoTrainer``, this can be ``{'wrapper': ModelWrapper}``.
        """
        # Distributed config parser
        run_config = copy.deepcopy(run_config)
        experiment_dir = run_config.experiment_dir or ""
        # TODO {junzhu} write a test case for relative path. The trials have
        # different relative path and fail to find the main directory.
        experiment_path = Path(experiment_dir).absolute()
        run_config.experiment_dir = str(
            experiment_path.joinpath(f"experiment_{run_config.uid}")
        )

        super().__init__(*args, run_config=run_config, **kwargs)  # type: ignore

        assert issubclass(
            type(self.run_config), ParallelConfig
        ), f"run_config must be of a type - { ParallelConfig.__name__} received {type(self.run_config)}"

        self.run_config: ParallelConfig
        self.run_config = run_config
        self.device = butils.parse_device(self.run_config.device)
        self.experiment_dir: Path = Path(run_config.experiment_dir)
        self.make_rclone_config()
        if self.run_config.rclone_config is not None:
            self.mock_rclone_config = copy.deepcopy(self.run_config.rclone_config)
            self.mock_rclone_config.startMount(self.experiment_dir)
        self.logger = FileLogger(path=self.experiment_dir / "mp.log")

        self.experiment_state: ExperimentState
        self.total_trials = self.run_config.total_trials
        self.gpu: float = 0.0
        self.cpu: float = self._make_cpu()
        if self.device.startswith("cuda"):
            self.gpu_mem_bottleneck = min(get_gpu_max_mem())
            if min(get_gpu_max_mem()) != max(get_gpu_max_mem()):
                self.logger.warn(
                    f"Bottlenecked memory utilization by {self.gpu_mem_bottleneck}."
                )
            self.gpu = self._make_gpu()
        self.experiment_dir.joinpath("default_config.yaml").write_text(
            str(self.run_config), encoding="utf-8"
        )
        self.total_mem_usage = 0

    def make_rclone_config(self):
        count = 0
        rclone_config = None
        for rclone_config_name in allowed_rclone_remote_configs:
            config = getattr(self.run_config, rclone_config_name)
            if config:
                count += 1
                rclone_config = config
        assert count <= 1, "You can just have one central remote repository"
        if rclone_config is not None:
            self.run_config.rclone_config = rclone_config

    def _make_gpu(self):
        if (gpu := self.run_config.gpu_mb_per_experiment / self.gpu_mem_bottleneck) > 0:
            mem_util = int(gpu * self.run_config.concurrent_trials)
            sys_mem = int(sum(get_gpu_max_mem()))
            if mem_util > sys_mem * 0.8:
                self.logger.warn(
                    f"Expected GPU memory utilization {mem_util}MiB > 80% "
                    f"of system available memory {sys_mem}MiB."
                )
                self.logger.warn(
                    "Consider adjusting `concurrent_trials` or `gpu_mb_per_experiment`."
                )
        return gpu

    def _make_cpu(self) -> int:
        if (cpu := self.run_config.cpus_per_experiment) > 1:
            cpu = np.floor(cpu)

        assert cpu > 0, "Invalid experiment_per_cpu count"

        if cpu * self.run_config.concurrent_trials > mp.cpu_count():
            self.logger.warn(
                f"Expected CPU core util. exceed system capacity {mp.cpu_count()}."
            )
            self.logger.warn(
                "Consider adjusting `concurrent_trials` or `cpus_per_experiment`."
            )

        return int(cpu)

    def kill_idle(self):
        """
        Kill any ray processes that are idle.
        """
        p = subprocess.Popen(
            [
                "ps aux | grep ray::IDLE | grep -v grep | awk '{print $2}' | xargs kill -9"
            ],
            shell=True,
        )
        os.waitpid(p.pid, 0)

    def _make_remote_fn(
        self,
        max_error_retries: int = 0,
    ) -> ty.Any:
        return ray.remote(
            num_gpus=self.gpu,
            num_cpus=self.cpu,
            max_calls=1,
            max_retries=max_error_retries,
        )(train_main_remote)

    def _make_remotes(
        self,
        trials: list[ParallelConfig],
    ):
        model_obj = ray.put(copy.deepcopy(self.wrapper))
        mp_logger = ray.put(copy.deepcopy(self.logger))
        remotes = []
        for run_config in trials:
            if (remote_fn := self._make_remote_fn()) is not None:
                diffs = self.run_config.diff_str(run_config)
                diffs = "\n\t".join(diffs)
                resume = run_config.uid in [
                    cfg.uid for cfg in self.experiment_state.resumed_trials
                ]
                action = "Scheduling" if resume is False else "Resuming"
                msg = f"{action} uid: {run_config.uid}\nParameters: \n\t{diffs}\n-----"
                self.logger.info(msg)
                self.experiment_state.update_trial_state(
                    run_config.uid, None, TrialState.RUNNING
                )
                remotes.append(
                    remote_fn.remote(
                        model_obj,
                        copy.deepcopy(run_config),
                        mp_logger,
                        self.experiment_dir,
                        True,
                        None,
                        resume,
                        True,
                    )
                )

        return remotes

    def _init_state(
        self,
        working_dir: str = "",
        address: str | None = "auto",
        modules: list[tys.ModuleType] | None = None,
        resume: bool = False,
    ):
        if not ray.is_initialized():
            if modules is None:
                modules = [ablator_module]
            if ablator_module not in modules:
                modules.append(ablator_module)
            runtime_env = {
                "working_dir": working_dir,
                "py_modules": modules,
            }
            ray.init(
                address=address,
                runtime_env=runtime_env,
            )
        super()._init_state()
        if resume:
            self.logger.info("Trying to run from resumed experiment...")
        self.experiment_state = ExperimentState(
            self.experiment_dir, self.run_config, self.logger, resume=resume
        )

    def __make_remotes_from_trials(self, trials: list[ParallelConfig] | None):
        if trials is None or len(trials) == 0:
            return []
        self.logger.info(f"Making {len(trials)} trials.")
        futures = self._make_remotes(trials)
        return futures

    def evaluate(self, parallel=False, working_directory: str = "", ray_head_address: str | None = "auto", auxilary_modules: list[tys.ModuleType] | None = None):
        """
        Evaluate model performance in trials that are completed, using evaluation functions defined
        in the model wrapper. Evaluation results will be logged to the console and log files in the
        experiment directory. This method also synchronizes the experiment directory to Google cloud
        storage and remote servers.
        """
        self._init_state(
            working_dir=working_directory,
            address=ray_head_address,
            modules=auxilary_modules,
            resume=True,
        )
        eval_configs = []
        trial_uids = self.experiment_state.complete_trials
        for config in trial_uids:
            config_path = self.experiment_dir.joinpath(config.uid, "config.yaml")
            if os.path.exists(
                config_path
            ):
                model_config = type(self.run_config).load(
                    config_path
                )
            eval_configs.append(model_config)
            self.logger.info(f"Evaluating trials...uid: {config.uid}")

        # TODO evaluate in parallel
        futures = []
        all_config_metrics = {}
        for model_config in eval_configs:
            if ray.is_initialized() and parallel:
                self.logger.info("Evaluating in parallel...")
                futures.append(
                    ray.remote(num_gpus=self.gpu, num_cpus=self.cpu)(
                        evaluate_remote
                    ).remote(self.wrapper, model_config, self.logger)
                )
            else:
                metrics_dict = self.wrapper.evaluate(model_config)
                all_config_metrics[model_config.uid] = metrics_dict
        while len(futures) > 0:
            done_id, futures = ray.wait(futures, num_returns=1)
            metrics_dict, eval_config = ray.get(done_id[0])
            all_config_metrics[eval_config.uid] = metrics_dict
        return all_config_metrics

    def launch(  # type: ignore
        self,
        working_directory: str,
        auxilary_modules: list[tys.ModuleType] | None = None,
        ray_head_address: str | None = "auto",
        resume: bool = False,
    ):
        """
        Set up and launch the parallel training and tuning process. This includes:

        - prepare ray cluster for running optuna trials to tune hyperparameters.

        - if available, synchronize Google Cloud storage buckets to working directory defined in runtime configuration.

        - initialize optuna trials and add them to optuna storage and experiment state database
          for tracking training progress (or retrieve existing trials from optuna storage).

        Trials initialized (or retrieved), :obj:`experiment_state.pending_trials`,
        will be pushed to ray nodes so they can be executed in parallel. After all trials
        have finished and progress is recorded in sqlite databases in the working directory,
        these changes will be synchronized back to the GCP nodes via ``rsync_up()`` method.

        Parameters
        ----------
        working_directory : str
            The working directory that stores codes, modules that will be used by ray.
        auxilary_modules : list[tys.ModuleType], None
            A list of modules to be used as ray clusters' working environment.
        ray_head_address : str, default='auto'
            Ray cluster address.
        resume : bool, default=False
            Whether to resume training the model from existing checkpoints and existing experiment state.
        """
        try:
            torch.multiprocessing.set_start_method("spawn")
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        self._init_state(
            working_dir=working_directory,
            address=ray_head_address,
            modules=auxilary_modules,
            resume=resume,
        )

        futures = []
        trials: list[ParallelConfig] | None = self.experiment_state.pending_trials
        futures = self.__make_remotes_from_trials(trials)
        config: ParallelConfig
        metrics: dict[str, float] | None
        trial_state: TrialState
        n_trials_to_sample = 1
        while len(futures) > 0:
            try:
                done_id, futures = ray.wait(
                    futures, num_returns=n_trials_to_sample, timeout=60
                )
                if len(done_id) > 0:
                    config, metrics, trial_state = ray.get(done_id[0])
                    metrics = parse_metrics(
                        list(self.run_config.optim_metrics.keys()), metrics
                    )
                    self.experiment_state.update_trial_state(
                        config.uid, metrics, trial_state
                    )
                    trials = self.experiment_state.sample_trials(n_trials_to_sample)
                    futures += self.__make_remotes_from_trials(trials)
                else:
                    self.logger.info(
                        f"Waiting for {len(futures)} trials to finish running."
                    )
            except builtins.Exception as e:
                # NOTE we do not know which trial caused the error, only
                # the pending trials (which we can assume one is the errored)
                exception = traceback.format_exc()
                self.logger.error(exception)

                if isinstance(e, KeyboardInterrupt):
                    pending_trials = self.experiment_state.pending_trials
                    self.logger.warn(
                        f"There are {len(pending_trials)} unfinished trials. with ids: {pending_trials}"
                    )
                    sys.exit(0)

        complete_ids = [c.uid for c in self.experiment_state.complete_trials]
        self.logger.info(
            f"There are {len(self.experiment_state.complete_trials)} complete trials. with ids: {complete_ids}"
        )
        errored: list[ParallelConfig] = (
            self.experiment_state.pending_trials + self.experiment_state.failed_trials
        )

        if len(errored) > 0:
            errored_ids = [c.uid for c in errored]
            self.logger.error(
                f"There are {len(errored)} unfinished trials. with ids: {errored_ids}"
            )

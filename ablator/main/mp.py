import copy
import multiprocessing as mp
import os
import socket
import subprocess
import sys
import traceback
import types as tys
import typing as ty
from pathlib import Path

import numpy as np
import ray
import torch

import ablator as ablator_module
from ablator.main.configs import ParallelConfig
from ablator.main.model.main import TrainPlateauError
from ablator.main.model.wrapper import ModelWrapper
from ablator.main.proto import ProtoTrainer
from ablator.main.state import ExperimentState, TrialState
from ablator.modules.loggers.file import FileLogger
from ablator.modules.metrics.main import LossDivergedError
from ablator.utils.base import get_gpu_max_mem

# The exceptions that are unrecoverable i.e.  [DuplicateRunError]
CRASH_EXCEPTION_TYPES: list[type] = []


def parse_rsync_paths(
    rsynced_folder: Path | str,
    root_folder: Path | str | None = None,
) -> dict[str, Path | str]:
    rsync_path = Path(rsynced_folder)
    root_path = rsync_path if root_folder is None else Path(root_folder)

    return {
        "local_path": rsync_path,
        "remote_path": rsync_path.relative_to(root_path.parent).as_posix(),
    }


def parse_metrics(optim_direction: list[str], metrics: dict[str, float] | None):
    return (
        {k: v for k, v in metrics.items() if k in optim_direction}
        if metrics is not None
        else None
    )


def train_main_remote(
    model: ModelWrapper,
    run_config: ParallelConfig,
    mp_logger: FileLogger,
    root_dir: Path,
    fault_tollerant: bool = True,
    crash_exceptions_types: list[type] | None = None,
) -> tuple[ParallelConfig, dict[str, float] | None, TrialState]:
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
        res = model.train(run_config)
        mp_logger.info(f"Finished training - {run_config.uid}")
        return run_config, res.to_dict(), TrialState.COMPLETE
    except (LossDivergedError, TrainPlateauError):
        return (
            run_config,
            model.metrics.to_dict(),
            TrialState.PRUNED_POOR_PERFORMANCE,
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

    except Exception as e:
        return handle_exception(e)
    finally:
        if model.model_dir is not None:
            kwargs = parse_rsync_paths(model.model_dir, root_dir)
            if run_config.gcp_config is not None:
                run_config.gcp_config.rsync_up(**kwargs, logger=mp_logger)
            elif run_config.remote_config is not None:
                run_config.remote_config.rsync_up(**kwargs, logger=mp_logger)


class ParallelTrainer(ProtoTrainer):
    def __init__(self, *args, run_config: ParallelConfig, **kwargs):
        # Distributed config parser
        run_config = copy.deepcopy(run_config)
        run_config.experiment_dir = os.path.join(
            run_config.experiment_dir, f"experiment_{run_config.uid}"
        )
        super().__init__(*args, run_config=run_config, **kwargs)  # type: ignore

        assert issubclass(
            type(self.run_config), ParallelConfig
        ), f"run_config must be of a type - { ParallelConfig.__name__} received {type(self.run_config)}"

        self.run_config: ParallelConfig
        self.run_config = run_config
        self.device = self.run_config.device
        self.experiment_dir: Path = Path(run_config.experiment_dir)
        self.logger = FileLogger(path=self.experiment_dir / "mp.log")
        self.experiment_state: ExperimentState
        self.total_trials = self.run_config.total_trials

        self.gpu_mem_bottleneck = min(get_gpu_max_mem())
        if min(get_gpu_max_mem()) != max(get_gpu_max_mem()):
            self.logger.warn(
                f"Bottlenecked memory utilization by {self.gpu_mem_bottleneck}."
            )
        self.cpu: float = self._make_cpu()
        self.gpu: float = self._make_gpu()
        self.experiment_dir.joinpath("default_config.yaml").write_text(
            str(self.run_config), encoding="utf-8"
        )
        self.total_mem_usage = 0

    def _make_gpu(self):
        if (gpu := self.run_config.gpu_mb_per_experiment / self.gpu_mem_bottleneck) > 0:
            assert (
                self.device == "cuda"
            ), "Device must be set to 'cuda' if `gpu_mb_per_experiment` > 0"
            assert (
                torch.cuda.is_available()
            ), "Could not find a torch.cuda installation on your system."
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

        return cpu

    def kill_idle(self):
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
            diffs = self.run_config.diff_str(run_config)
            diffs = "\n\t".join(diffs)
            msg = f"Scheduling uid: {run_config.uid}\nParameters: \n\t{diffs}\n-----"
            self.logger.info(msg)
            if (remote_fn := self._make_remote_fn()) is not None:
                remotes.append(
                    remote_fn.remote(
                        model_obj,
                        copy.deepcopy(run_config),
                        mp_logger,
                        self.experiment_dir,
                    )
                )

        return remotes

    def _init_state(
        self,
        working_dir: str,
        address: str | None = "auto",
        modules: list[tys.ModuleType] | None = None,
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
        self.sync_down()
        self.experiment_state = ExperimentState(
            self.experiment_dir, self.run_config, self.logger
        )

    def _rsync_nodes(self):
        if self.run_config.gcp_config is None:
            return
        node_hostnames = [
            str(node["NodeManagerHostname"])
            for node in ray.nodes()
            if node["NodeManagerHostname"] != socket.gethostname()
        ]
        for hostname in node_hostnames:
            self.run_config.gcp_config.rsync_down_node(
                hostname, self.experiment_dir, self.logger
            )

    def _rsync_remote_up(self):
        if self.run_config.remote_config is None:
            return
        kwargs = parse_rsync_paths(self.experiment_dir)
        self.run_config.remote_config.rsync_up(logger=self.logger, **kwargs)

    def _rsync_remote_down(self):
        if self.run_config.remote_config is None:
            return
        kwargs = parse_rsync_paths(self.experiment_dir)
        self.run_config.remote_config.rsync_down(logger=self.logger, **kwargs)

    def _rsync_gcp_up(self):
        if self.run_config.gcp_config is None:
            return
        kwargs = parse_rsync_paths(self.experiment_dir)
        self.run_config.gcp_config.rsync_up(logger=self.logger, **kwargs)

    def _rsync_gcp_down(self):
        if self.run_config.gcp_config is None:
            return
        kwargs = parse_rsync_paths(self.experiment_dir)
        self.run_config.gcp_config.rsync_down(logger=self.logger, **kwargs)

    def __make_remotes_from_trials(self, trials: list[ParallelConfig] | None):
        if trials is None or len(trials) == 0:
            return []
        self.logger.info(f"Making {len(trials)} trials.")
        futures = self._make_remotes(trials)
        return futures

    def sync_down(self):
        # Can be previously run trials if we are resuming the state.
        # First sync down from the remote
        self._rsync_gcp_down()
        self._rsync_nodes()

    def sync_up(self):
        self._rsync_gcp_up()
        self._rsync_remote_up()

    def evaluate(self):
        eval_configs = []
        trial_uids = self.experiment_state.complete_trials
        for config in trial_uids:
            model_config = type(self.run_config).load(
                self.experiment_dir.joinpath(config.uid, "config.yaml")
            )
            eval_configs.append(model_config)

        # TODO evaluate in parallel
        for model_config in eval_configs:
            self.wrapper.evaluate(model_config)
        self.sync_up()

    def launch(
        self,
        working_directory: str,
        auxilary_modules: list[tys.ModuleType] | None = None,
        ray_head_address: str | None = "auto",
    ):
        try:
            torch.multiprocessing.set_start_method("spawn")
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        self._init_state(
            working_dir=working_directory,
            address=ray_head_address,
            modules=auxilary_modules,
        )

        futures = []
        trials: list[ParallelConfig] | None = self.experiment_state.running_trials
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
            except Exception as e:
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
        self.sync_up()

import copy
import multiprocessing
import multiprocessing as mp
import os
import socket
import subprocess
import traceback
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from trainer.modules.logging.file import FileLogger
from trainer.modules.storage.main import ExperimentState, TrialState

try:
    import ray
except ImportError:
    raise ImportError(
        "You need to install model-trainer with option [mp]. `pip install ablator[mp]`"
    )
import torch

from trainer.base import BaseTrainer
from trainer.config.run import ParallelConfig
from trainer.modules.main import LossDivergedError, Metrics
from trainer.modules.model.wrapper import (ModelWrapper,
                                           TrainPlateauError)
from trainer.utils.mp import get_gpu_max_mem, get_least_used_gpu
from trainer.utils.train import debugger_is_active


class MPTrainer(BaseTrainer):
    def __init__(self, *args, run_config: ParallelConfig, **kwargs):
        # Distributed config parser
        run_config = copy.deepcopy(run_config)
        run_config.experiment_dir = self.make_experiment_dir(run_config=run_config)

        super().__init__(*args, run_config=run_config, **kwargs)  # type: ignore

        assert issubclass(
            type(self.run_config), ParallelConfig
        ), f"run_config must be of a type - { ParallelConfig.__name__} received {type(self.run_config)}"

        self.run_config: ParallelConfig
        self.run_config = run_config
        self.experiment_type = run_config.experiment_type
        self.tune = self.run_config.tune
        self.device = self.run_config.train_config.device

        self.derived_mem_multiplier = self.run_config.derived_mem_multiplier
        assert self.derived_mem_multiplier > 0 and self.derived_mem_multiplier <= 1
        self.experiment_dir: Path = Path(run_config.experiment_dir)
        self.experiment_state: ExperimentState
        self.total_trials = self.run_config.total_trials

        self.cpu: float = self.make_cpu()
        self.gpu: Optional[float] = self.make_gpu()
        self.gpu_mem_bottleneck = min(get_gpu_max_mem())
        self.system_avail_mem = sum(get_gpu_max_mem())
        self.total_mem_usage = 0

    def make_experiment_dir(self, run_config: ParallelConfig):

        return os.path.join(run_config.experiment_dir, f"mp_run_{run_config.uid}")

    def make_gpu(self):
        gpu = self.run_config.num_experiment_per_gpu

        if gpu is not None:
            gpu = 1 / gpu
        if gpu is None or gpu > 0:
            assert (
                self.device == "cuda"
            ), f"Device must be set to 'cuda' for {type(self).__name__}"
        return gpu

    def make_cpu(self):
        cpu = self.run_config.num_experiment_per_cpu

        if cpu is None:
            cpu = 1000

        if cpu > 0:
            cpu = multiprocessing.cpu_count() / cpu
            if cpu > 1:
                # ray expects int if cpu>1
                cpu = np.floor(cpu)

        assert cpu > 0, "Invalid experiment_per_cpu count"
        return cpu

    def kill_idle(self):
        p = subprocess.Popen(
            [
                "ps aux | grep ray::IDLE | grep -v grep | awk '{print $2}' | xargs kill -9"
            ],
            shell=True,
        )
        os.waitpid(p.pid, 0)

    def find_mem_usage(self, run_config: ParallelConfig):

        run_config = copy.deepcopy(run_config)
        least_used_gpu = get_least_used_gpu()
        run_config.train_config.device = f"cuda:{least_used_gpu}"
        mem_usage = self.get_memory_use(run_config)
        return mem_usage

    def get_gpu_frac(self, mem_usage):

        # NOTE naive approach of getting the min of the maximum gpu memory in a system
        # This should work fine for homegeous GPU clusters but will be a bottleneck
        # for heteregenous clusters with different size GPUs
        # This is due to caching delays or other ops taking more memory
        mem_threshold = mem_usage * self.derived_mem_multiplier
        model_per_gpu = max(
            np.floor(self.gpu_mem_bottleneck / (mem_usage + mem_threshold)), 1
        )
        return 1 / model_per_gpu

    def _find_num_gpus(self, run_config: ParallelConfig):
        uid = run_config.uid
        try:
            mem_usage = self.find_mem_usage(run_config)

        except Exception as e:
            traceback.print_exc()
            self.logger.error(
                f"Error! Model caused error issue with trial {uid}. \nSkipping..."
            )
            if debugger_is_active():
                raise e
            else:
                return -1

        gpu_frac = self.get_gpu_frac(mem_usage)
        if gpu_frac > 1:
            self.logger.warn(
                f"Warning! Possible out-of-memory: {uid} required {mem_usage:_.0f}MiB with num_gpus {gpu_frac:.2f} -> 1"
            )

        num_gpus = min(gpu_frac, 1)
        return num_gpus, mem_usage

    def _train_main_remote_wrapper(
        self,
        num_gpus: float,
        ignore_errors=True,
        max_error_retries=0,
    ) -> Any:
        # TODO: fault tolerance strategies.
        # restart experiment when max_restries>0 and the experiment fails.
        # Either by resuming or if there is no valid checkpoint exiting / terminating

        # crash_exception_types = [DuplicateRunError]
        crash_exception_types = []

        @ray.remote(
            num_gpus=num_gpus,
            num_cpus=self.cpu,
            max_calls=1,
            max_retries=max_error_retries,
        )
        def train_main_remote(
            model: ModelWrapper,
            run_config: ParallelConfig,
            mp_logger: FileLogger,
            experiment_dir: Path,
        ) -> Tuple[ParallelConfig, Optional[Metrics], TrialState]:
            # TODO init data state specific to the node
            # NOTE this is because of errors in the node of directories. need a way to synchronize
            # experiment_dir = Path(run_config.experiment_dir).joinpath(
            #     f"mp_run_{run_config.uid}"
            # )
            # experiment_dir.mkdir(exist_ok=True, parents=True)
            # FileLogger(path=join(experiment_dir, "trainer.log"))

            def handle_exception(e):
                exception_str = traceback.format_exc()
                if hasattr(model, "logger"):
                    model.logger.error(exception_str)
                mp_logger.error(f"Error Occured {run_config.uid}")
                traceback.print_exc()
                if not ignore_errors or isinstance(e, tuple(crash_exception_types)):
                    error_msg = (
                        f"Error {type(e).__name__} in"
                        f"{' '.join([c.__name__ for c in crash_exception_types])}. Exiting."
                    )
                    mp_logger.error(error_msg)
                    raise type(e)(error_msg)

                return run_config, None, TrialState.FAIL

            try:
                res = model.train(run_config)
                mp_logger.info(f"Finished training - {run_config.uid}")
                return run_config, res, TrialState.COMPLETE
            except (LossDivergedError, TrainPlateauError) as e:
                return run_config, model.metrics, TrialState.PRUNED_POOR_PERFORMANCE
            except RuntimeError as e:
                if str(e).startswith("CUDA out of memory."):
                    mp_logger.warn(
                        f"Cuda out of memory for {run_config.uid}. Restarting..."
                    )
                    return run_config, model.metrics, TrialState.RECOVERABLE_ERROR
                else:
                    return handle_exception(e)

            except Exception as e:
                return handle_exception(e)
            finally:
                if run_config.gcp_config is not None:
                    bucket = run_config.gcp_config.bucket
                    destination = (
                        Path(bucket)
                        / Path(model.model_dir).parent.parent.name
                        / Path(model.model_dir).parent.name
                    )
                    # TODO fixme error prone, in case it hangs or returns an error.
                    run_config.gcp_config.rsync_up(
                        Path(model.model_dir),
                        destination.as_posix(),
                        mp_logger,
                        verbose=False,
                    )

        return train_main_remote

    def debug_remote(self, run_config: ParallelConfig):
        return self.model.mock_train(run_config, run_async=False)

    def _make_remote(self, run_config: ParallelConfig, mock_run=False):
        if self.gpu is None:
            num_gpus, mem_usage = self._find_num_gpus(run_config=run_config)
        else:
            num_gpus = self.gpu
            mem_usage = self.gpu_mem_bottleneck * num_gpus

        self.total_mem_usage += mem_usage

        if mock_run:
            self.debug_remote(run_config=run_config)
            return None
        elif num_gpus > 0:
            diffs = self.run_config.diff_str(run_config)
            diffs = "\n\t".join(diffs)
            run_stats = f"uid: {run_config.uid} mem_usage/total_usage: {mem_usage}MiB/{self.total_mem_usage}MiB num_gpus: {num_gpus} num_cpus: {self.cpu}\n"
            config_stats = f"config diff: \n\t{diffs}\n-----"
            self.logger.info(run_stats + config_stats)

            remote = self._train_main_remote_wrapper(
                num_gpus=num_gpus,
            )
            return remote
        return None

    def _make_remotes(
        self,
        trials: List[ParallelConfig],
        mock_run: bool = False,
    ):
        model_obj = ray.put(copy.deepcopy(self.model))
        mp_logger = ray.put(copy.deepcopy(self.logger))
        remotes = []
        for run_config in trials:

            remote_fn = self._make_remote(run_config=run_config, mock_run=mock_run)
            if remote_fn is not None:
                remotes.append(
                    remote_fn.remote(
                        model_obj,
                        copy.deepcopy(run_config),
                        mp_logger,
                        self.experiment_dir,
                    )
                )

        return remotes

    def init_state(self):
        if not ray.is_initialized():
            ray.init(address="auto")
        # TODO automatic ray.init. It could be error prone.
        super().init_state()
        self.sync_down()
        self.experiment_state = ExperimentState(
            self.experiment_dir, self.run_config, self.logger
        )

    def sync_down(self):
        # Can be previously run trials if we are resuming the state.
        # First sync down from the remote

        self._rsync_gcp_down()
        self._rsync_nodes()

    def _rsync_nodes(self, verbose=True):

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

        pass

    def sync_up(self):
        # self._sync_trials(self.experiment_state.trials)
        self._rsync_gcp_up()

    def _rsync_gcp_up(self):
        if self.run_config.gcp_config is None:
            return
        self.run_config.gcp_config.rsync_up(self.experiment_dir, logger=self.logger)

    def _rsync_gcp_down(self):
        if self.run_config.gcp_config is None:
            return

        self.run_config.gcp_config.rsync_down(self.experiment_dir, self.logger)

    def evaluate(self, model_dir: Optional[str] = None, chkpt: Optional[str] = None):
        eval_configs = []
        trial_uids = self.experiment_state.complete_trials
        for config in trial_uids:
            model_config = type(self.run_config).load(
                self.experiment_dir.joinpath(config.uid, "config.yaml")
            )
            eval_configs.append(model_config)

        # TODO evaluate in parallel
        for model_config in eval_configs:
            self.model.evaluate(model_config, model_dir=model_dir, chkpt=chkpt)
        self.sync_up()

    def __make_remotes_from_trials(self, trials: List[ParallelConfig]):
        if len(trials) == 0:
            return []
        self.logger.info(f"Making {len(trials)} trials.")
        futures = self._make_remotes(trials)
        return futures

    def launch(self):
        try:
            torch.multiprocessing.set_start_method("spawn")
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        self.init_state()

        futures = []
        trials = self.experiment_state.running_trials
        futures = self.__make_remotes_from_trials(trials)
        config: ParallelConfig
        metric: Optional[Metrics]
        trial_state: TrialState
        n_trials_to_sample = 1
        while len(futures):
            try:
                # ray hangs if a worker crashes
                done_id, futures = ray.wait(
                    futures, num_returns=n_trials_to_sample, timeout=60
                )
                if len(done_id):
                    # TODO only works for n_trials_to_sample = 1
                    config, metric, trial_state = ray.get(done_id[0])
                    self.experiment_state.update_trial_state(
                        config.uid, metric, trial_state
                    )
                    trials = self.experiment_state.sample_trials(n_trials_to_sample)
                    futures += self.__make_remotes_from_trials(trials)
                    # TODO rsync files like *.db *.yaml and *.log because they are ignored otherwise
                else:
                    self.logger.info(
                        f"Waiting for {len(futures)} trials to finish running."
                    )
            except Exception as e:
                # NOTE we do not know which trial caused the error, only the pending trials (which we can assume one is the errored)
                exception = traceback.format_exc()
                self.logger.error(exception)

                if isinstance(e, KeyboardInterrupt):
                    pending_trials = self.experiment_state.pending_trials
                    self.logger.warn(
                        f"There are {len(pending_trials)} unfinished trials. with ids: {pending_trials}"
                    )
                    exit(0)

        complete_ids = [c.uid for c in self.experiment_state.complete_trials]
        self.logger.info(
            f"There are {len(self.experiment_state.complete_trials)} complete trials. with ids: {complete_ids}"
        )
        errored: List[ParallelConfig] = (
            self.experiment_state.pending_trials + self.experiment_state.failed_trials
        )

        if len(errored) > 0:
            errored_ids = [c.uid for c in errored]
            self.logger.error(
                f"There are {len(errored)} unfinished trials. with ids: {errored_ids}"
            )
        self.sync_up()

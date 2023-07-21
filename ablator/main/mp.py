import builtins
import copy
import multiprocessing as mp
import os
import shutil
import sys
import traceback
import types as tys
import typing as ty
import uuid
from collections import OrderedDict
from functools import cached_property
from pathlib import Path

import numpy as np
import ray
import torch

import ablator.utils.base as butils
from ablator.config.mp import ParallelConfig
from ablator.main.model.main import CheckpointNotFoundError, TrainPlateauError
from ablator.main.model.wrapper import ModelWrapper
from ablator.main.proto import ProtoTrainer
from ablator.main.state import ExperimentState, TrialState
from ablator.modules.loggers.file import RemoteFileLogger
from ablator.modules.metrics.main import LossDivergedError
from ablator.mp.node_manager import NodeManager, Resource
from ablator.mp.utils import _sorted_nodes_by_util
from ablator.utils.base import get_gpu_mem
from ablator.utils.progress_bar import RemoteDisplay, RemoteProgressBar


# TODO refactor into a seperate file and reduce complexity
# pylint: disable=too-complex,broad-exception-caught
def train_main_remote(
    model: ModelWrapper,
    run_config: ParallelConfig,
    mp_logger: RemoteFileLogger,
    uid: int,
    fault_tollerant: bool = True,
    crash_exceptions_types: list[type] | None = None,
    resume: bool = False,
    clean_reset: bool = False,
    progress_bar: ty.Optional[RemoteProgressBar] = None,
) -> tuple[int, dict[str, float] | None, TrialState]:
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
    uid : int
        the trial unique identifier.
    fault_tollerant : bool, optional, default=True
        Whether to tollerate crashes, aka to cease execution when the ray job crashes.
    crash_exceptions_types : list[type], None, optional, default=None
        Types of exceptions that are considered as crashes.
    resume : bool, default=False
        Whether to resume training the model from existing checkpoints and existing experiment state.
    clean_reset : bool, default=False
        Whether to remove model directory when ``CheckpointNotFoundError`` is raised.
    progress_bar : RemoteProgressBar, optional
        Optionally, we can use a remote progress bar to update the results of the trial.

    Returns
    -------
    int
        The trial uid corresponding to the results.
    dict[str, float], None
        If exception raised (Except for LossDivergedError and TrainPlateauError),
        this will be ``None`` object. Otherwise, this will be a dictionary of metrics.
    TrialState
        A TrialState object indicating the state of the trial job

        - If ``LossDivergedError`` or ``TrainPlateauError`` is raised while training,
          returned state will be ``TrialState.PRUNED_POOR_PERFORMANCE``

        - If ``RuntimeError`` (with message ``'CUDA out of memory'``),
          or ``CheckpointNotFoundError`` (with ``clean_reset=True``) is raised while training,
          returned state will be ``TrialState.FAIL_RECOVERABLE``

        - If other types of error or ``CheckpointNotFoundError`` (with ``clean_reset=False``) is raised,
          returned state will be ``TrialState.FAIL``

    """
    if crash_exceptions_types is None:
        crash_exceptions_types = []

    if torch.cuda.is_available():
        least_busy_gpu = np.argmax(list(get_gpu_mem("free").values()))
        # TODO instead of CUDA_VISIBLE_DEVICES, using a context could be cleaner.
        # with torch.cuda.device(1):
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{least_busy_gpu}"

    def handle_exception(e):
        exception_str = traceback.format_exc()
        if hasattr(model, "logger"):
            model.logger.error(exception_str)
        else:
            mp_logger.error(f"{run_config.uid}:\n{exception_str}")
        mp_logger.error(f"Error Occured {run_config.uid}: {str(e)}")
        if not fault_tollerant:
            raise e
        if isinstance(e, tuple(crash_exceptions_types)):
            error_msg = (
                f"Exception `{str(e)}` of type: `{type(e).__name__}` in crash_exceptions_types="
                f"{{c.__name__ for c in crash_exceptions_types}}. Exiting."
            )
            mp_logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        return uid, None, TrialState.FAIL

    try:
        res = model.train(run_config, resume=resume, remote_progress_bar=progress_bar)
        mp_logger.info(f"Finished training - {run_config.uid}")
        return uid, res, TrialState.COMPLETE
    except (LossDivergedError, TrainPlateauError) as e:
        mp_logger.warn(
            f"Trial {run_config.uid} was pruned for poor performance. {str(e)}"
        )
        return (
            uid,
            model.metrics,
            TrialState.PRUNED_POOR_PERFORMANCE,
        )
    except CheckpointNotFoundError as e:
        # This is the case for corrupt artifacts.
        if clean_reset:
            if model.experiment_dir is not None:
                shutil.rmtree(model.experiment_dir.as_posix())
            return (
                uid,
                None,
                TrialState.FAIL_RECOVERABLE,
            )

        return handle_exception(e)
    except RuntimeError as e:
        if butils.is_oom_exception(e):
            mp_logger.warn(f"Cuda out of memory for {run_config.uid}. Restarting...")
            return (
                uid,
                None,
                TrialState.FAIL_RECOVERABLE,
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
    logger : RemoteFileLogger
        A centralized logger that writes messages to a file and prints them to the console.
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

    """

    def __init__(self, wrapper: ModelWrapper, run_config: ParallelConfig):
        """
        Initialize ``ParallelTrainer`` using config from ``run_config``.

        Parameters
        ----------
        wrapper: ModelWrapper
            The model wrapper for the ParallelTrainer
        run_config : ParallelConfig
            The runtime configuration for this trainer.
        """

        self.run_config: ParallelConfig
        super().__init__(wrapper=wrapper, run_config=run_config)
        # Distributed config parser
        experiment_dir = self.run_config.experiment_dir or ""
        experiment_path = Path(experiment_dir).absolute().resolve()
        if not experiment_path.stem.startswith("experiment_"):
            experiment_path = experiment_path.joinpath(
                f"experiment_{self.run_config.uid}"
            )
        self.run_config.experiment_dir = str(experiment_path)
        self.experiment_dir: Path = Path(self.run_config.experiment_dir)

        assert issubclass(
            type(self.run_config), ParallelConfig
        ), f"run_config must be of a type - { ParallelConfig.__name__} received {type(self.run_config)}"

        assert issubclass(
            type(self.wrapper), ModelWrapper
        ), f"run_config must be of a type - { ModelWrapper.__name__} received {self.wrapper}"

        self.logger: RemoteFileLogger
        self.experiment_state: ExperimentState
        self.total_trials: int | None
        self.ray_address: str
        self.available_resources: dict[str, Resource]
        self._progress_bar: ty.Optional[RemoteProgressBar] = None
        self._display: butils.Dummy | RemoteDisplay = butils.Dummy()
        self.node_manager: NodeManager

    @cached_property
    def _gpu(self) -> float:
        """
        _gpu virtual number of GPUs used to schedule remotes on a GPU nodes.
        We handle GPU allocation internally.

        Returns
        -------
        float
            mock gpu value i.e. 0.001
        """
        device = butils.parse_device(self.run_config.device)
        if not device.startswith("cuda"):
            return 0
        return 0.001

    @cached_property
    def _cpu(self) -> float:
        """
        _cpu expected to be run AFTER _init_state as it requires the cluser to be initialized.
        it is used as a virtual number of `num_cpus` for ray while we handle resource allocation
        manually.

        Returns
        -------
        int | float
            a virtual number of _cpus to use i.e. 0.001
        """
        if (
            self.run_config.concurrent_trials is None
            or self.run_config.concurrent_trials > mp.cpu_count()
        ):
            self.logger.warn(
                f"Expected CPU core util. can exceed system capacity {mp.cpu_count()}.\n"
                "Consider adjusting `concurrent_trials`."
            )

        return 0.001

    def _make_remote(
        self,
        trial_id: int,
        run_config: ParallelConfig,
        node_ip: str,
        max_error_retries: int = 0,
        resume: bool = False,
    ):
        trial_uuid = f"{run_config.uid}_{str(uuid.uuid4())[:4]}"
        wrapper = copy.deepcopy(self.wrapper)
        # pylint: disable=protected-access
        wrapper._uid = trial_uuid
        model_obj = ray.put(wrapper)

        remote_fn = ray.remote(
            num_gpus=self._gpu,
            num_cpus=self._cpu,
            max_calls=1,
            max_retries=max_error_retries,
        )(
            train_main_remote
        ).options(  # type: ignore
            resources={f"node:{node_ip}": 0.001}, name=trial_uuid
        )
        run_config.experiment_dir = (self.experiment_dir / trial_uuid).as_posix()
        diffs = self.run_config.diff_str(run_config)
        diffs = "\n\t".join(diffs)
        action = "Scheduling" if resume is False else "Resuming"
        msg = f"{action} uid: {trial_uuid}\nParameters: \n\t{diffs}\n-----"
        self.logger.info(msg)
        self.experiment_state.update_trial_state(trial_id, None, TrialState.RUNNING)
        return remote_fn.remote(
            model_obj,
            copy.deepcopy(run_config),
            self.logger,
            trial_id,
            True,
            None,
            resume,
            True,
            self._progress_bar,
        )

    def _heartbeat(self):
        self._display.refresh(force=True)
        self.available_resources = self.node_manager.available_resources()
        # TODO find which tasks have died from available_resources and update experiment_state

    def _make_futures(self, current_futures: list | None = None, soft_limit: int = 10):
        # make enough futures such that there are concurrent_trials running.
        futures = [] if current_futures is None else current_futures
        concurrent_trial_limit: int | None = self.run_config.concurrent_trials
        gpu_util = self.run_config.gpu_mb_per_experiment if self._gpu > 0 else None

        resources = self.available_resources
        node_ips = _sorted_nodes_by_util(
            resources=resources,
            gpu_util_requirement=gpu_util,
        )
        n_running_tasks: dict[str, int] = {
            node_ip: len(v.running_tasks) for node_ip, v in resources.items()
        }
        node_scheduled_tasks: dict[str, int] = OrderedDict()
        for node_ip in node_ips:
            node_scheduled_tasks[node_ip] = 0
        # a soft limit to the number of trials to sample at a time.
        # There is no benefit in sampling many trials at once as the function
        # is called on every heart-beat.
        while (
            len(futures) < soft_limit
            and (
                self.total_trials is None
                or len(self.experiment_state.valid_trials()) < self.total_trials
            )
            and len(node_scheduled_tasks) > 0
        ):
            # NOTE updating available_resources is slow, and better to keep it fixed and sort by least
            # utilized node. It assumes that soft_limit is sufficiently
            # small that it will not cause unexpected over-utilization in between sampling
            # available_resources = self.node_manager.available_resources()
            node_ip, scheduled_tasks = sorted(
                node_scheduled_tasks.items(), key=lambda item: item[1]
            )[0]
            if (
                concurrent_trial_limit is not None
                and scheduled_tasks + n_running_tasks[node_ip] >= concurrent_trial_limit
            ):
                del node_scheduled_tasks[node_ip]
                continue
            if (
                gpu_util is not None
                and len(resources[node_ip].gpu_free_mem_arr) > 0
                and resources[node_ip].gpu_free_mem_arr.max() > gpu_util
            ):
                least_util_gpu_name = resources[node_ip].least_used_gpu
                resources[node_ip].gpu_free_mem[least_util_gpu_name] -= gpu_util
            elif gpu_util is not None:
                del node_scheduled_tasks[node_ip]
                continue

            node_scheduled_tasks[node_ip] += 1
            try:
                trial_id, trial = self.experiment_state.sample_trial()
            except StopIteration:
                self.logger.warn(
                    f"Received StopIteration signal, trial limit possibly reached {self.total_trials}"
                )
                break

            future = self._make_remote(trial_id, trial, node_ip)
            futures.append(future)

        return futures

    def pre_train_setup(self):
        """
        Used to prepare resources to avoid stalling during training or when resources are
        shared between trainers.
        """
        mock_wrapper = copy.deepcopy(self.wrapper)
        mock_config = copy.deepcopy(self.run_config)
        mock_config.experiment_dir = None
        # pylint: disable=protected-access
        future = (
            ray.remote(
                num_gpus=self._gpu,
                num_cpus=self._cpu,
                max_calls=1,
                max_retries=0,
            )(
                lambda wrapper: wrapper._init_state(
                    run_config=mock_config, smoke_test=True, debug=True
                )
            )
            .options()
            .remote(ray.put(mock_wrapper))
        )
        ray.get(future)

    @property
    def total_trials(self):
        return self.run_config.total_trials

    @total_trials.setter
    def total_trials(self, value):
        self.run_config.total_trials = value

    def _init_state(
        self,
        working_dir: str = "",
        address: str | None = None,
        modules: list[tys.ModuleType] | None = None,
        resume: bool = False,
        excluding_files: list[str] | None = None,
    ):
        verbose = self.run_config.verbose

        if self.experiment_dir.exists() and not resume:
            raise RuntimeError(f"Experiment Directory {self.experiment_dir} exists.")
        self.logger = RemoteFileLogger(
            path=self.experiment_dir / "mp.log", verbose=verbose == "console"
        )
        self.experiment_dir.joinpath("default_config.yaml").write_text(
            str(self.run_config), encoding="utf-8"
        )
        self.experiment_state = ExperimentState(
            self.experiment_dir, self.run_config, self.logger, resume=resume
        )
        if excluding_files is None:
            excluding_files = [".git/**"]

        if verbose == "progress":
            # pylint: disable=no-member
            self._progress_bar = RemoteProgressBar.remote(self.total_trials)  # type: ignore
            self._display = RemoteDisplay(self._progress_bar)  # type: ignore

        if ray.is_initialized():
            self.logger.warn(
                "Ray is already initialized. Can not start another instance. "
                "Unexpected behavior can occur. We recommend to perform `ray.shutdown()` "
                "or `ray stop` before starting the experiment. "
                "You can set 'address=\"local\"' on `.launch` to start another cluster."
            )
            ray_context = ray.get_runtime_context()
            self.ray_address = ray_context.gcs_address
            # TODO find a way to set-up runtime env on running cluster
            # NOTE this is because https://docs.ray.io/en/latest/ray-core/handling-dependencies.html
            # `Note: Setting options (1) and (3) per-task or per-actor is
            # currently unsupported, it can only be set per-job (i.e., in ray.init()).`
        else:
            runtime_env = {
                "working_dir": working_dir,
                "excludes": [".git"] + excluding_files,
                "py_modules": modules,
            }
            # pylint: disable=cyclic-import,import-outside-toplevel
            import ablator as ablator_module

            if modules is None:
                modules = [ablator_module]
            if ablator_module not in modules:
                modules.append(ablator_module)
            runtime_env["py_modules"] = modules
            ray_cluster = ray.init(
                log_to_driver=verbose == "console",
                logging_level="warning",
                include_dashboard=True,  # required for `list_nodes` function
                address=address,
                runtime_env=runtime_env,
            )
            self.ray_address = ray_cluster.address_info["address"]
        self.node_manager = NodeManager(private_key_home=Path.home())
        self.logger.to_remote()
        # first heartbeat <3
        self._heartbeat()
        super()._init_state()

    # pylint: disable=arguments-renamed
    def launch(  # type: ignore
        self,
        working_directory: str,
        auxilary_modules: list[tys.ModuleType] | None = None,
        ray_head_address: str | None = None,
        resume: bool = False,
        excluding_files: list[str] | None = None,
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
        ray_head_address : str, None
            Ray cluster address.
        resume : bool, default=False
            Whether to resume training the model from existing checkpoints and existing experiment state.
        excluding_files: list[str], None
            A list of files in `.gitignore` format, that will be excluded from being uploaded to the ray cluster.
            If unspecified it ignores `.git/**` folder.
        """
        try:
            torch.multiprocessing.set_start_method("spawn")
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        # TODO move inside __init__
        self._init_state(
            working_dir=working_directory,
            address=ray_head_address,
            modules=auxilary_modules,
            resume=resume,
            excluding_files=excluding_files,
        )
        valid_trials = self.experiment_state.valid_trials()
        if self.total_trials is not None and len(valid_trials) >= self.total_trials:
            self.logger.error(f"Trial limit {self.total_trials} was reached. Exiting.")
            return

        futures = self._make_futures()
        metrics: dict[str, float] | None
        trial_state: TrialState
        heart_beat_interval = 1
        while len(futures) > 0:
            # pylint: disable=broad-exception-caught
            try:
                done_id, futures = ray.wait(
                    futures, num_returns=1, timeout=heart_beat_interval
                )
                if len(done_id) > 0:
                    uid, metrics, trial_state = ray.get(done_id[0])
                    self.experiment_state.update_trial_state(uid, metrics, trial_state)
                futures = self._make_futures(futures)
            except KeyboardInterrupt:
                self.logger.warn("KeyboardInterrupt signal received.")
                self._print_summary()
                sys.exit(0)
            except StopIteration:
                # Reached maximum number of sample trials
                continue
            except Exception:
                exception = traceback.format_exc()
                self.logger.error(f"Unhandled Exception: {exception}")
            finally:
                self._heartbeat()
        self._print_summary()

    def _print_summary(self):
        pending_trials = [
            c.id
            for c in self.experiment_state.get_trials_by_state(TrialState.WAITING)
            + self.experiment_state.get_trials_by_state(TrialState.RUNNING)
        ]
        complete_trials = [
            c.id for c in self.experiment_state.get_trials_by_state(TrialState.COMPLETE)
        ]
        errored_trials = [
            c.id for c in self.experiment_state.get_trials_by_state(TrialState.FAIL)
        ]
        self.logger.info(
            f"There are {len(complete_trials)} complete trials. with ids: {complete_trials}"
        )

        if len(pending_trials) > 0:
            self.logger.warn(
                f"There are {len(pending_trials)} unfinished trials. with ids: {pending_trials}"
            )
        if len(errored_trials) > 0:
            self.logger.error(
                f"There are {len(errored_trials)} errored trials. with ids: {errored_trials}"
            )

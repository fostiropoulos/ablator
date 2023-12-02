import copy
import itertools
import multiprocessing as mp
import sys
import traceback
import types as tys
import typing as ty
import uuid
from collections import defaultdict
from functools import cached_property
from pathlib import Path
import numpy as np

import ray
import torch
from ablator.modules.loggers.file import RemoteFileLogger
from ablator.mp.gpu import GPUError

import ablator.utils.base as butils
from ablator.config.mp import ParallelConfig
from ablator.main.model.wrapper import ModelWrapper
from ablator.main.proto import ProtoTrainer
from ablator.main.state import ExperimentState, TrialState
from ablator.mp.cluster import ClusterManager
from ablator.mp.utils import get_node_ip, ray_init
from ablator.utils.progress_bar import RemoteDisplay, RemoteProgressBar
from ablator.mp.train_remote import train_main_remote
from ablator.config.types import Optional


class ParallelTrainer(ProtoTrainer):
    """
    A class for parallelizing multiple training processes of models of different configurations with ray.

    Parameters
    ----------
    wrapper : ModelWrapper
        The model wrapper for the ``ParallelTrainer``.
    run_config : ParallelConfig
        The runtime configuration for this trainer.

    Attributes
    ----------
    run_config : ParallelConfig
        Running configuration for parallel training.
    logger : RemoteFileLogger
        A centralized logger that writes messages to a file and prints them to the console.
    experiment_state : ExperimentState
        This attribute manages optuna trials.
    gpu_manager : ty.Optional[GPUManager]
        A GPU manager that manages GPU resources in the cluster.
    available_resources : dict[str, Resource]
        A dictionary of available resources on each node.
    node_manager : NodeManager
        A node manager that manages nodes and their resources.
    ray_address : str
        The address of the ray cluster.
    total_trials : int
        Total number of trials to run.
    gpu_mem_bottleneck : int
        The minimum memory capacity of all available gpus.
    cpu : float
        The number of cpu used per trial.
    gpu : float
        The number of gpu used per trial.
    running_futures : dict[str, list]
        A dictionary with keys the Node IP and values a list of Ray remote tasks executing
        on the node aka `futures`.
    cluster_manager : ClusterManager
        The cluster manager responsible for scheduling tasks and managing resources

    Examples
    --------
    Below is a complete workflow on how to launch a parallel experiment with ``ParallelTrainer``,
    from defining config, getting the model wrapper ready, to launching the experiment:

    - Define training config:

    >>> my_optimizer_config = OptimizerConfig("sgd", {"lr": 0.5, "weight_decay": 0.5})
    >>> my_scheduler_config = SchedulerConfig("step", arguments={"step_size": 1, "gamma": 0.99})
    >>> train_config = TrainConfig(
    ...     dataset="[Dataset Name]",
    ...     batch_size=32,
    ...     epochs=10,
    ...     optimizer_config = my_optimizer_config,
    ...     scheduler_config = my_scheduler_config
    ... )

    - Define model config, we want to run HPO on activation functions and model hidden size:

    >>> @configclass
    >>> class CustomModelConfig(ModelConfig):
    >>>     hidden_size: int
    >>>     activation: str
    >>> model_config = CustomModelConfig(hidden_size=100, activation="relu")

    - Define search space:

    >>> search_space = {
    ...     "train_config.optimizer_config.arguments.lr": SearchSpace(
    ...         value_range = [0.001, 0.01],
    ...         value_type = 'float'
    ...         ),
    ...     "model_config.hidden_size": SearchSpace(value_range = [32, 64], value_type = 'int'),
    ...     "model_config.activation": SearchSpace(categorical_values = ["relu", "elu", "leakyRelu"]),
    ... }

    - Define run config (remember to redefine the parallel config to update the model config type to
      be ``CustomModelConfig``):

    >>> @configclass
    >>> class CustomParallelConfig(ParallelConfig):
    ...    model_config: CustomModelConfig
    >>>
    >>> parallel_config = CustomParallelConfig(
    ...     train_config=train_config,
    ...     model_config=model_config,
    ...     metrics_n_batches = 800,
    ...     experiment_dir = "/tmp/experiments/",
    ...     device="cuda",
    ...     amp=True,
    ...     random_seed = 42,
    ...     total_trials = 20,
    ...     concurrent_trials = 3,
    ...     search_space = search_space,
    ...     optim_metrics = {"val_loss": "min"},
    ...     optim_metric_name = "val_loss",
    ...     gpu_mb_per_experiment = 1024
    ... )

    - Create model wrapper:

    >>> class MyModelWrapper(ModelWrapper):
    >>>     def __init__(self, *args, **kwargs):
    >>>         super().__init__(*args, **kwargs)
    >>>
    >>>     def make_dataloader_train(self, run_config: CustomParallelConfig):
    >>>         return torch.utils.data.DataLoader(<train_dataset>, batch_size=32, shuffle=True)
    >>>
    >>>     def make_dataloader_val(self, run_config: CustomParallelConfig):
    >>>         return torch.utils.data.DataLoader(<val_dataset>, batch_size=32, shuffle=False)

    - After gathering all configurations and model wrapper, we can initialize and launch the parallel trainer:

    >>> wrapper = MyModelWrapper(
    ...     model_class=<your_ModelModule_class>,
    ... )
    >>> ablator = ParallelTrainer(
    ...     wrapper=wrapper,
    ...     run_config=parallel_config,
    ... )
    >>> ablator.launch(working_directory = os.getcwd(), ray_head_address=None)
    """

    def __init__(self, wrapper: ModelWrapper, run_config: ParallelConfig):
        # Initialize ``ParallelTrainer`` using config from ``run_config``.

        self.run_config: ParallelConfig
        super().__init__(wrapper=wrapper, run_config=run_config)
        assert issubclass(type(self.run_config), ParallelConfig), (
            f"run_config must be of a type - { ParallelConfig.__name__} received"
            f" {type(self.run_config)}"
        )

        assert issubclass(type(self.wrapper), ModelWrapper), (
            f"wrapper must be of a type - { ModelWrapper.__name__} received"
            f" {self.wrapper}"
        )

        self.logger: RemoteFileLogger
        self.experiment_state: ExperimentState
        self.total_trials: int | None
        self.ray_address: str
        self._progress_bar: ty.Optional[RemoteProgressBar] = None
        self._display: butils.Dummy | RemoteDisplay = butils.Dummy()
        self.running_futures: dict[str, list] = defaultdict(lambda: [])
        self.cluster_manager: ClusterManager

    @cached_property
    def _gpu(self) -> float:
        """
        _gpu virtual number of GPUs used to schedule remotes on a GPU nodes.
        We handle GPU allocation internally.

        Returns
        -------
        float
            mock gpu value i.e. 0.001

        Raises
        ------
        ValueError
            if the `gpu_mb_per_experiment` configuration is not specified when using `device='cuda'`
        """
        device = butils.parse_device(self.run_config.device)
        if not device.startswith("cuda"):
            return 0
        if self.run_config.gpu_mb_per_experiment is None:
            raise ValueError(
                "config attribute `gpu_mb_per_experiment` can not be `None` when"
                " device=`cuda`"
            )
        return 0.001

    @cached_property
    def _cpu(self) -> float:
        """
        _cpu expected to be run AFTER _init_state as it requires the cluser to be initialized.
        it is used as a virtual number of `num_cpus` for ray while we handle resource allocation
        manually.

        Returns
        -------
        float
            a virtual number of _cpus to use i.e. 0.001
        """
        if (
            self.run_config.concurrent_trials is None
            or self.run_config.concurrent_trials > mp.cpu_count()
        ):
            self.logger.warn(
                "Expected CPU core util. can exceed system capacity"
                f" {mp.cpu_count()}.\nConsider adjusting `concurrent_trials`."
            )

        return 0.01

    def _make_remote(
        self,
        trial_id: int,
        run_config: ParallelConfig,
        node_ip: str,
        max_error_retries: int = 0,
        resume: bool = False,
    ):
        trial_uuid = f"{run_config.uid}_{str(uuid.uuid4())[:4]}"
        gpu, manager = (None, None)
        if self._gpu > 0:
            gpu, manager = self.cluster_manager.get_gpu(
                node_ip=node_ip, process_name=trial_uuid
            )
            for node in ray.nodes():
                if (
                    node["NodeManagerAddress"] == node_ip
                    and "GPU" not in node["Resources"]
                ):
                    raise RuntimeError("Misconfigured Ray cluster.")

        wrapper = copy.deepcopy(self.wrapper)
        # pylint: disable=protected-access
        wrapper._uid = trial_uuid
        model_obj = ray.put(wrapper)

        remote_fn = ray.remote(
            num_gpus=self._gpu,
            num_cpus=self._cpu,
            max_calls=1,
            max_retries=max_error_retries,
        )(train_main_remote).options(
            resources={f"node:{node_ip}": 0.001}, name=trial_uuid
        )
        if node_ip == get_node_ip():
            run_config.experiment_dir = (self.experiment_dir / trial_uuid).as_posix()
        elif run_config.remote_config is None:
            # NOTE this should never happen during normal use-case
            # the remote_config is automatically created on multi-node cluster
            # to be the head node of the cluster.
            raise RuntimeError(
                "Could not identify remote_config. Critical error encountered."
                " remote_config unspecified when scheduling remotes on multi-node"
                " cluster."
            )
        else:
            run_config.experiment_dir = (
                (Path("~") / "ablator").joinpath(
                    *Path(run_config.remote_config.local_path).parts[1:]
                )
                / trial_uuid
            ).as_posix()

        list_diffs = self.run_config.diff_str(run_config)
        diffs = "\n\t".join(list_diffs)
        action = "Scheduling" if resume is False else "Resuming"
        msg = (
            f"{action} @ {node_ip} with uid: {trial_uuid}\nParameters:"
            f" \n\t{diffs}\n-----"
        )
        self.logger.info(msg)
        self.experiment_state.update_trial_state(trial_id, None, TrialState.RUNNING)
        data_lock = butils.Lock()
        return remote_fn.remote(
            model=model_obj,
            run_config=copy.deepcopy(run_config),
            mp_logger=self.logger,
            resource_manager=manager,
            gpu=gpu,
            uid=trial_id,
            fault_tollerant=True,
            crash_exceptions_types=None,
            resume=resume,
            clean_reset=True,
            progress_bar=self._progress_bar,
            data_lock=data_lock,
        )

    def _heartbeat(self):
        self._display.refresh(force=True)

    # pylint: disable=too-complex
    def _make_futures(self, soft_limit: int = 10) -> list:
        # make enough futures such that there are concurrent_trials running.
        concurrent_trial_limit: int | None = self.run_config.concurrent_trials
        gpu_util = self.run_config.gpu_mb_per_experiment if self._gpu > 0 else None

        starting_futures = np.array([len(v) for v in self.running_futures.values()])

        def is_limit(node_ip: str | None = None):
            futures = np.array([len(v) for v in self.running_futures.values()])
            return (
                futures.sum() - starting_futures.sum() >= soft_limit
                or (
                    node_ip is not None
                    and len(self.running_futures[node_ip]) > 0
                    and concurrent_trial_limit is not None
                    and (futures >= concurrent_trial_limit).all()
                )
                or (
                    self.total_trials is not None
                    and len(self.experiment_state.valid_trials()) >= self.total_trials
                )
            )

        def interleaved_running_futures():
            # interleaves the futures from all nodes that are running.
            return [
                x
                for x in itertools.chain(
                    *itertools.zip_longest(*self.running_futures.values())
                )
                if x is not None
            ]

        while not is_limit():
            resources = self.cluster_manager.sorted_resources(gpu_mem=gpu_util)
            remote_config = self.cluster_manager.remote_config
            if len(resources) == 0:
                break
            for node_ip in resources:
                if is_limit(node_ip):
                    return interleaved_running_futures()

                if (
                    concurrent_trial_limit is not None
                    and len(self.running_futures[node_ip]) >= concurrent_trial_limit
                ):
                    continue
                try:
                    trial_id, trial = self.experiment_state.sample_trial()
                except StopIteration:
                    self.logger.warn(
                        "Received StopIteration signal, trial limit possibly reached"
                        f" {self.total_trials}"
                    )
                    return interleaved_running_futures()
                try:
                    trial.remote_config = remote_config
                    future = self._make_remote(trial_id, trial, node_ip)
                    self.running_futures[node_ip].append(future)
                except GPUError:
                    self.logger.warn(f"Not Enough GPU resources for {node_ip}.")
                    continue
        return interleaved_running_futures()

    def pre_train_setup(self):
        """
        Used to prepare resources to avoid stalling during training or when resources are
        shared between trainers.
        """
        mock_wrapper = copy.deepcopy(self.wrapper)
        mock_config = copy.deepcopy(self.run_config)
        mock_config.experiment_dir = None
        future = (
            ray.remote(
                num_gpus=self._gpu,
                num_cpus=self._cpu,
                max_calls=1,
                max_retries=0,
            )(
                lambda wrapper: wrapper.init_state(
                    run_config=mock_config, smoke_test=True, debug=True
                )
            )
            .options()
            .remote(ray.put(mock_wrapper))
        )
        ray.get(future)

    @property
    def total_trials(self) -> Optional[int]:
        return self.run_config.total_trials

    @total_trials.setter
    def total_trials(self, value):
        self.run_config.total_trials = value

    def _init_ray(
        self,
        working_dir: str = "",
        address: str | None = None,
        modules: list[tys.ModuleType] | None = None,
        excluding_files: list[str] | None = None,
        verbose: ty.Literal["console", "progress", "silent"] = "silent",
    ):
        if excluding_files is None:
            excluding_files = [".git/**"]

        _is_ray_init = False
        if ray.is_initialized():
            _is_ray_init = True

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

            ray_kwargs = {
                "log_to_driver": verbose == "console",
                "logging_level": "warning",
                "include_dashboard": True,  # required for `list_nodes` function
                "address": address,
                "runtime_env": runtime_env,
            }

            ray_cluster = ray_init(**ray_kwargs)
            self.ray_address = ray_cluster.address_info["address"]
        return _is_ray_init

    def _init_state(
        self,
        working_dir: str = "",
        address: str | None = None,
        modules: list[tys.ModuleType] | None = None,
        resume: bool = False,
        excluding_files: list[str] | None = None,
        debug: bool = False,
    ):
        self.stop()
        verbose = self.run_config.verbose
        if self.experiment_dir.exists() and not resume:
            raise RuntimeError(f"Experiment Directory {self.experiment_dir} exists.")
        self._mount(resume=resume, debug=debug)
        _is_ray_init = self._init_ray(
            working_dir=working_dir,
            address=address,
            modules=modules,
            excluding_files=excluding_files,
            verbose=verbose,
        )
        self.cluster_manager = ClusterManager(
            private_key_home=Path.home(),
            sync_directory=self.experiment_dir,
            ray_address=self.ray_address,
            remote_config=self.run_config.remote_config,
        )
        self.logger = RemoteFileLogger(
            path=self.experiment_dir / "mp.log", verbose=verbose == "console"
        )
        self.experiment_dir.joinpath("master_config.yaml").write_text(
            self.run_config.to_yaml(), encoding="utf-8"
        )
        self.experiment_state = ExperimentState(
            self.experiment_dir, self.run_config, self.logger, resume=resume
        )
        self.logger.to_remote()

        # TODO check if this causes an error because it was placed before the ray init
        if verbose == "progress":
            raise NotImplementedError(
                "verbose='progress' currently not supported for mp-training."
            )

        if _is_ray_init:
            self.logger.warn(
                "Ray is already initialized. Can not start another instance. Unexpected"
                " behavior can occur. We recommend to perform `ray.shutdown()` or `ray"
                " stop` before starting the experiment. You can set 'address=\"local\"'"
                " on `.launch` to start another cluster."
            )

        # first heartbeat <3
        self._heartbeat()
        diffs = self._get_diffs(working_dir)
        self.logger.warn(diffs)

    # flake8: noqa: DOC201, DOC502
    # pylint: disable=arguments-renamed,too-complex
    def launch(  # type: ignore[override]
        self,
        working_directory: str,
        auxilary_modules: list[tys.ModuleType] | None = None,
        ray_head_address: str | None = None,
        resume: bool = False,
        excluding_files: list[str] | None = None,
        debug: bool = False,
    ):
        """
        Set up and launch the parallel ablation experiment. This sets up a ray cluster, and trials of different
        configuration initialized (or retrieved) will be pushed to the ray cluster to run in parallel.

        Parameters
        ----------
        working_directory : str
            The working directory that stores codes and modules that will be used by ray.
        auxilary_modules : list[tys.ModuleType] | None
            A list of modules to be used as ray clusters' working environment.
        ray_head_address : str | None
            Ray cluster address.
        resume : bool
            Whether to resume training the model from existing checkpoints and
            existing experiment state, by default ``False``.
        excluding_files : list[str] | None
            A list of files in `.gitignore` format, that will be excluded from being uploaded to the ray cluster.
            If unspecified it ignores `.git/**` folder.
        debug : bool, optional
            Whether to train model in debug mode. By default ``False``

        Raises
        ------
        RuntimeError
            If the `config.experiment_id` is unspecified but resuming an experiment or the
            experiment directory is not empty but uses a remote storage configuration.
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
            excluding_files=excluding_files,
            debug=debug,
        )
        if debug:
            self.pre_train_setup()

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
                    done_future = done_id[0]
                    for v in self.running_futures.values():
                        if done_future in v:
                            v.remove(done_future)
                    uid, metrics, trial_state = ray.get(done_future)
                    self.experiment_state.update_trial_state(uid, metrics, trial_state)
                futures = self._make_futures()
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
            f"There are {len(complete_trials)} complete trials. with ids:"
            f" {complete_trials}"
        )

        if len(pending_trials) > 0:
            self.logger.warn(
                f"There are {len(pending_trials)} unfinished trials. with ids:"
                f" {pending_trials}"
            )
        if len(errored_trials) > 0:
            self.logger.error(
                f"There are {len(errored_trials)} errored trials. with ids:"
                f" {errored_trials}"
            )

    def stop(self):
        super().stop()
        if hasattr(self, "cluster_manager") and self.cluster_manager is not None:
            self.cluster_manager.stop()

import copy
import time
import typing as ty
from copy import deepcopy
from pathlib import Path

import git
import torch
from git import exc

try:
    from rmount import RemoteMount
except ImportError:
    RemoteMount = None

from ablator.config.proto import RunConfig
from ablator.main.model.wrapper import ModelWrapper
from ablator.utils.file import expand_path


class ProtoTrainer:
    """
    Manages resources for Prototyping. This trainer runs an experiment of a single
    prototype model (Therefore no ablation study nor HPO).

    Parameters
    ----------
    wrapper : ModelWrapper
        The main model wrapper.
    run_config : RunConfig
        Running configuration for the model.

    Attributes
    ----------
    wrapper : ModelWrapper
        The main model wrapper.
    run_config : RunConfig
        Running configuration for the model.
    experiment_dir : Path
        The path object to the experiment directory.

    Raises
    ------
    RuntimeError
        If the experiment directory is not defined in the running configuration.

    Examples
    --------
    Below is a complete workflow on how to launch a prototype experiment with ``ProtoTrainer``,
    from defining the config to launching the experiment:

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

    - Define model config: we use the default one with no custom hyperparameters (sometimes you would
      want to customize it to run ablation study/ HPO on the model's hyperparameters in a parallel
      experiment, which needs ``ParallelTrainer`` and ``ParallelConfig`` instead of ``ProtoTrainer``
      and ``RunConfig``):

    >>> model_config = ModelConfig()

    - Define run config:

    >>> run_config = RunConfig(
    ...     train_config=train_config,
    ...     model_config=model_config,
    ...     metrics_n_batches = 800,
    ...     experiment_dir = "/tmp/experiments",
    ...     device="cpu",
    ...     amp=False,
    ...     random_seed = 42
    ... )

    - Create model wrapper:

    >>> class MyModelWrapper(ModelWrapper):
    >>>     def __init__(self, *args, **kwargs):
    >>>         super().__init__(*args, **kwargs)
    >>>
    >>>     def make_dataloader_train(self, run_config: RunConfig):
    >>>         return torch.utils.data.DataLoader(<train_dataset>, batch_size=32, shuffle=True)
    >>>
    >>>     def make_dataloader_val(self, run_config: RunConfig):
    >>>         return torch.utils.data.DataLoader(<val_dataset>, batch_size=32, shuffle=False)

    - After gathering all configurations and model wrapper, it's time we initialize and launch the
      prototype trainer. When launching the experiment, we must provide a working directory, which
      points to a git repository that is used for keeping track of the code differences:

    >>> wrapper = MyModelWrapper(
    ...     model_class=<your_ModelModule_class>,
    ... )
    >>> ablator = ProtoTrainer(
    ...     wrapper=wrapper,
    ...     run_config=run_config,
    ... )
    >>> metrics = ablator.launch(working_directory=os.getcwd())  # suppose current directory is tracked by git
    """

    def __init__(
        self,
        wrapper: ModelWrapper,
        run_config: RunConfig,
    ):
        # Initialize model wrapper and running configuration for the model.
        super().__init__()
        self.wrapper = copy.deepcopy(wrapper)
        self.run_config: RunConfig = copy.deepcopy(run_config)
        self.mount_server: None | "RemoteMount" = None
        if self.run_config.experiment_dir is None:
            raise RuntimeError("Must specify an experiment directory.")
        experiment_dir = self.run_config.experiment_dir
        experiment_path = expand_path(experiment_dir)
        self.experiment_dir = experiment_path
        self.run_config.experiment_dir = experiment_dir
        self._is_new_experiment = False
        if self.run_config.experiment_id is None:
            # we choose time.time as a id as opposed to a uuid, because
            # it is informative of the creation time, and it is almost certain
            # to be unique, unless two experiments are created on the same
            # moment by the same user and in the same experiment directory....
            # unlikely. (Who is f*ing with us?)
            self.run_config.experiment_id = f"experiment_{int(time.time())}"
            self._is_new_experiment = True
        self.experiment_id: str = self.run_config.experiment_id

    def pre_train_setup(self):
        """
        Used to prepare resources to avoid stalling during training or when resources are
        shared between trainers.
        """

    # pylint: disable=too-complex
    def _mount(self, resume: bool = False, debug: bool = False, timeout: int = 60):
        if resume and self._is_new_experiment:
            raise RuntimeError(
                "Can not leave `experiment_id` unspecified in the configuration when"
                " resuming an experiment."
            )
        if resume or not self._is_new_experiment:
            self.stop()

        self._is_new_experiment = False
        if self.run_config.remote_config is None:
            return False
        # optional import that must be installed with [server]
        # pylint: disable=import-outside-toplevel,redefined-outer-name
        try:
            from rmount import RemoteMount
        except ImportError as e:
            raise ImportError(
                "remote_config is only supported for Linux systems."
            ) from e

        local_path = self.experiment_dir
        if len(list(local_path.glob("*"))) > 0:
            raise RuntimeError(
                f"The experiment directory `{local_path}` is not empty and it"
                " will lead to errors when synchronizing with a remote storage."
            )
        remote_config = self.run_config.remote_config
        config = remote_config.get_config()
        if remote_config.ssh is not None and remote_config.remote_path is None:
            self.run_config.remote_config.remote_path = "/ablator"
        if remote_config.s3 is not None and remote_config.remote_path is None:
            raise ValueError

        self.run_config.remote_config.local_path = str(local_path)
        remote_path = self.run_config.remote_config.remote_path
        self.mount_server = RemoteMount(
            settings=config,
            remote_path=Path(remote_path) / self.experiment_id,
            local_path=local_path,
            verbose=debug,
            timeout=timeout,
        )
        try:
            self.mount_server.mount()
        except RuntimeError as e:
            raise RuntimeError(
                f"Could not mount to remote directory {remote_path}. Make sure"
                f" that:\n\t1. The {remote_path} directory exists and it has the"
                " correct permisions, i.e. it is writable by the provided"
                " configuration user and/or access key. \n\t2. That the"
                f" configuration {config} is correct."
            ) from e
        return True

    def _get_diffs(self, working_dir: str = ""):
        try:
            repo = git.Repo(expand_path(working_dir).as_posix())
            t = repo.head.commit.tree
            diffs = repo.git.diff(t)
            return f"Git Diffs for {repo.head.ref} @ {repo.head.commit}: \n{diffs}"
        except ValueError as e:
            raise RuntimeError(
                f"Could not parse repo at {working_dir}. Error: {str(e)}"
            ) from e
        except exc.NoSuchPathError as e:
            raise FileNotFoundError(f"Directory {working_dir} was not found. ") from e
        except exc.InvalidGitRepositoryError:
            return (
                f"No git repository was detected at {working_dir}. "
                "We recommend setting the working directory to a git repository "
                "to keep track of changes."
            )

    # flake8: noqa: DOC502
    def launch(
        self, working_directory: str, resume: bool = False, debug: bool = False
    ) -> dict[str, float]:
        """
        Launch the prototype experiment (train, evaluate the single prototype model) and return metrics.

        Parameters
        ----------
        working_directory : str
            The working directory points to a git repository that is used for keeping track of
            the code differences.
        resume : bool
            Whether to resume training the model from existing checkpoints and
            existing experiment state. By default False
        debug : bool, optional
            Whether to train models in debug mode, by default ``False``.

        Returns
        -------
        dict[str, float]
            Metrics returned after training.

        Raises
        ------
        RuntimeError
            If the `config.experiment_id` is unspecified but resuming an experiment or the
            experiment directory is not empty but using a remote storage configuration.

        """
        self._mount(resume=resume, debug=debug)
        self.pre_train_setup()

        self.wrapper.init_state(
            run_config=self.run_config, smoke_test=False, debug=debug, resume=resume
        )
        diffs = self._get_diffs(working_directory)
        self.wrapper.logger.info(diffs)
        metrics = self.wrapper.train(debug=debug)
        return metrics

    def evaluate(self) -> dict[str, dict[str, ty.Any]]:
        """
        Run model evaluation on the training results, sync evaluation results to external logging services
        (e.g. Google cloud storage, other remote servers).

        Returns
        -------
        dict[str, dict[str, ty.Any]]
            Metrics returned after evaluation.
        """
        # TODO load model if it is un-trained
        metrics = self.wrapper.evaluate(self.run_config)
        return metrics

    def smoke_test(self, config: RunConfig | None = None) -> bool:
        """
        Run a smoke test training process on the model.

        Parameters
        ----------
        config : RunConfig | None
            Running configuration for the model.

        Returns
        -------
        bool
            Whether the smoke test was successful.

        Examples
        --------
        >>> try:
        ...    ablator.smoke_test(run_config)
        ... except err:
        ...    raise err
        """
        if config is None:
            config = self.run_config
        run_config = deepcopy(config)
        wrapper = type(self.wrapper)(self.wrapper.model_class)
        wrapper.train(run_config=run_config, smoke_test=True)
        del wrapper
        torch.cuda.empty_cache()
        return True

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        self.stop()

    def stop(self):
        if hasattr(self, "mount_server") and self.mount_server is not None:
            self.mount_server.unmount()

    # pylint: disable=broad-exception-caught
    def __del__(self):
        try:
            self.stop()
        except Exception:
            ...

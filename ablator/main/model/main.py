from contextlib import nullcontext
import copy
import math
import traceback
import typing as ty
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from functools import cached_property
from pathlib import Path

import setproctitle
import torch
from torch import nn
from torch.utils.data import DataLoader

import ablator.utils.base as butils
from ablator.config.proto import Optim, RunConfig
from ablator.modules.loggers.main import SummaryLogger
from ablator.modules.metrics.main import Metrics
from ablator.utils.base import Dummy, Lock
from ablator.utils.file import expand_path
from ablator.utils.progress_bar import ProgressBar, RemoteProgressBar


class EvaluationError(Exception):
    pass


class TrainPlateauError(Exception):
    pass


class LogStepError(Exception):
    pass


class CheckpointNotFoundError(FileNotFoundError):
    pass


# pylint: disable=too-many-instance-attributes
class ModelBase(ABC):
    """
    Base class that removes training boiler-plate code with extensible support
    for multiple use-cases. The class follows a stateful initialization paradigm.
    Requires the user to implement specific to their use-case load model and
    creation functionality.

    Attributes
    ----------
    model_class : Type[nn.Module]
        The class definition of the model's structure, which is a subclass of ``nn.Module``.
    run_config : RunConfig
        An instance of ``RunConfig`` containing configuration details.
    train_dataloader : DataLoader
        A DataLoader object responsible for model training.
    val_dataloader : Optional[DataLoader]
        An optional DataLoader object used for model evaluation.
    test_dataloader : Optional[DataLoader]
        An optional DataLoader object used for model testing.
    logger : Union[SummaryLogger, Dummy]
        Records information on the program's operation and model training, such as progress and performance metrics.
    device : str
        The type of device used for running the experiment. i.e. ``"cuda"``, ``"cpu"``, ``"cuda:0"``.
    model_dir : Path
        The model directory.
    experiment_dir : Path
        The experiment directory.
    verbose : bool
        If ``True``, prints additional information while training. Only applied for the master process.
    amp : bool
        If ``True``, apply automatic mixed precision training, otherwise default precision.
    random_seed : Optional[int]
        Sets the seed for generating random numbers.
    progress_bar : Union[ProgressBar, Dummy]
        An optional instance of ``ProgressBar`` that displays real-time information during training.
        e.g. time remaining. Only applied for the master process.
    current_checkpoint : Optional[Path]
        Directory for the current checkpoint file, by default ``None``.
    train_metrics : Metrics
        Training metrics including model information. i.e. learning rate and loss value.
    eval_metrics : Metrics | None
        Evaluation metrics for when a ``val_dataloader`` is provided.
    current_state : dict
        The currrent state of the model, including run_config, metrics and other necessary states.
    learning_rate : float
        The current learning rate.
    total_steps : int
        The total steps for the training process.
    epochs : int
        The total epochs for the training process.
    current_iteration : int
        The current iteration of training.
    best_iteration : int
        The iteration with the best loss value.
    best_metrics : dict[str, float]
        The lowest optim values encountered during training.
    optim_metric_name : str | None
        The name of the optimization metric.
    optim_metric_direction : Optim | None
        The optimization direction

    Parameters
    ----------
    model_class : type[nn.Module]
        The base class for user's model, which defines the neural network.

    Notes
    -----
    1. Class properties are simply listed by name. Please check out property docstring for more information.

    2. Users must implement the abstract methods to customize the model's behavior.

    3. Mixed precision training enables some operations to use the ``torch.float32`` datatype and
       other operations use lower precision floating point datatype ``torch.float16``.
       This is for saving time and reducing memory usage. Ordinarily,
       "automatic mixed precision training" means training with ``torch.autocast`` and
       ``torch.cuda.amp.GradScaler`` together.
       More information: https://pytorch.org/docs/stable/amp.html

    """

    def __init__(
        self,
        model_class: type[nn.Module],
    ):
        self.model_class = model_class
        self.run_config: RunConfig
        self.train_dataloader: DataLoader
        self.val_dataloader: DataLoader | None
        self.test_dataloader: DataLoader | None
        self.logger: ty.Union[SummaryLogger, Dummy]
        self.device: str
        self.experiment_dir: Path | None = None
        self.verbose: ty.Literal["progress", "console", "silent"]
        self.amp: bool
        self.random_seed: ty.Optional[int]
        self.progress_bar: ProgressBar | butils.Dummy

        self.optim_metric_name: str | None
        self.optim_metric_direction: Optim | None

        self.current_checkpoint: Path | None
        # Runtime metrics
        self.train_metrics: Metrics
        self.eval_metrics: Metrics | None
        self.current_state: dict

        # stats
        self.learning_rate: float
        self.total_steps: int
        self.epochs: int
        # self.current_epoch: int
        self.current_iteration: int
        self.best_iteration: int | None
        self.best_metrics: dict[str, float]
        # Attributes updated during training
        self._running_stats_names: list[str] = [
            "best_iteration",
            "best_metrics",
            "current_iteration",
            "learning_rate",
        ]
        # Attributes derived from configuration
        self._derived_stats_names: list[str] = [
            "epoch_len",
            "current_epoch",
            "log_itr",
            "eval_itr",
            "uid",
            "total_steps",
            "epochs",
        ]
        self._cached_properties: list[str] = [
            "epoch_len",
            "eval_itr",
            "log_itr",
        ]
        self._overridable_stats_names: list[str] = []
        # internal properties
        self._uid: str
        self._autocast: torch.autocast
        self._is_init = False
        # functions that call init_state
        self._init_function_names = ["init_state"]

    def _init_attrs(self):
        self.current_iteration = 0
        self.best_iteration = None
        self.best_metrics = {}
        self.learning_rate = float("inf")
        self.current_state = {}
        self.eval_metrics = None
        self.current_checkpoint = None
        self.val_dataloader = None
        self.test_dataloader = None

    def _reset_cached_attributes(self):
        for p in self._cached_properties:
            if hasattr(self, p):
                delattr(self, p)

    @property
    def train_stats(self) -> OrderedDict:
        """
        Returns an ordered dictionary containing the current training statistics.

        Returns
        -------
        OrderedDict
            An ordered dictionary with the following keys and values:

            - learning_rate: The current learning rate.

            - total_steps: The total steps for the training process.

            - epochs: The number of epochs for training.

            - current_epoch: The current epoch during training.

            - current_iteration: The current iteration during training.

            - best_iteration: The iteration with the best loss value so far.

            - best_*: The best (lowest) optim metric value achieved during training, for example `best_val_loss`

        """
        train_stats = OrderedDict(
            learning_rate=self.learning_rate,
            total_steps=self.total_steps,
            epochs=self.epochs,
            current_epoch=self.current_epoch,
            current_iteration=self.current_iteration,
            best_iteration=self.best_iteration,
        )
        for k, v in self.best_metrics.items():
            train_stats[f"best_{k}"] = v
        return train_stats

    @property
    def current_epoch(self) -> int:
        """
        Calculates and returns the current epoch during training.

        Returns
        -------
        int
            The current epoch number.
        """
        if self.current_iteration > 0:
            return math.floor(self.current_iteration / self.total_steps * self.epochs)
        return 0

    @cached_property
    def epoch_len(self) -> int:
        """
        Returns the length of an epoch, which is the number of batches in the ``train_dataloader``.

        Returns
        -------
        int
            The length of an epoch, represented as the number of batches in the ``train_dataloader``.

        Raises
        ------
        RuntimeError
            If the ``train_dataloader`` is not defined or its length is 0.
        """
        if not hasattr(self, "train_dataloader") or len(self.train_dataloader) == 0:
            raise RuntimeError("Undefined train_dataloader.")
        return len(self.train_dataloader)

    @cached_property
    def eval_itr(self) -> int:
        """
        Calculate the interval between evaluations.

        Returns
        -------
        int
            The interval between evaluations.
        """
        return math.ceil(self.run_config.eval_epoch * self.epoch_len)

    @cached_property
    def log_itr(self) -> int:
        """
        Calculate the interval between logging steps.

        Returns
        -------
        int
            The interval between logging steps.
        """
        return math.ceil(self.run_config.log_epoch * self.epoch_len)

    @property
    def uid(self) -> str:
        """
        Returns a unique identifier (UID) for the current run configuration.

        Returns
        -------
        str
            A string representing the unique identifier of the current run configuration.
        """
        return getattr(self, "_uid", self.run_config.uid)

    @abstractmethod
    def create_model(
        self,
        save_dict: dict[str, ty.Any] | None = None,
        strict_load: bool = True,
    ) -> None:
        """
        Abstract method to create and initialize the model. Must be implemented by subclasses.
        Example implementation: Please see the ``create_model`` method in the ``ModelWrapper`` class.

        Parameters
        ----------
        save_dict : dict[str, ty.Any] | None
            A dictionary containing saved model data, such as weights, optimizer state, etc.,
            to be loaded into the model, by default ``None``.
        strict_load : bool
            If True, the model will be loaded strictly, ensuring that the saved state
            matches the model's structure exactly. If False, the model can be loaded
            with a partially matching state, by default ``True``.

        Raises
        ------
        NotImplementedError
            If this method is not implemented by the subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def checkpoint(self, is_best: bool = False):
        """
        Abstract method to save a checkpoint of the model. Must be implemented by subclasses.
        Example implementation: Please see the ``checkpoint`` method in the ``ModelWrapper`` class.


        Parameters
        ----------
        is_best : bool
            Indicates if the current checkpoint is the best model so far, by default ``False``.

        Raises
        ------
        NotImplementedError
            If this method is not implemented by the subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        run_config: RunConfig,
        smoke_test: bool = False,
    ):
        """
        Abstract method to train the model. Must be implemented by subclasses.
        Example implementation: Please see the ``train`` method in the ``ModelWrapper`` class.

        Parameters
        ----------
        run_config : RunConfig
            An instance of ``RunConfig`` containing configuration details.
        smoke_test : bool
            Whether to run as a smoke test, by default ``False``.

        Raises
        ------
        NotImplementedError
            If this method is not implemented by the subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        run_config: RunConfig,
    ):
        """
        Abstract method to evaluate the model. Must be implemented by subclasses.
        Example implementation: Please see the ``evaluate`` method in the ``ModelWrapper`` class.

        Parameters
        ----------
        run_config : RunConfig
            An instance of ``RunConfig`` containing configuration details.

        Raises
        ------
        NotImplementedError
            If this method is not implemented by the subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def make_dataloaders(self, run_config: RunConfig):
        """
        Abstract method to create dataloaders for the training, validation, and testing datasets.

        This method should define the process of loading the data and creating dataloaders
        for the training, validation, and testing datasets based on the provided ``run_config``.

        Must be implemented by subclasses.
        Example implementation: Please see the ``make_dataloaders`` method in the ``ModelWrapper`` class.

        Parameters
        ----------
        run_config : RunConfig
            An instance of ``RunConfig`` containing configuration details.

        Raises
        ------
        NotImplementedError
            If this method is not implemented by the subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def config_parser(self, run_config: RunConfig):
        """
        Abstract method to parse the provided configuration.

        Must be implemented by subclasses.
        Example implementation: Please see the ``make_dataloaders`` method in the ``ModelWrapper`` class.

        Parameters
        ----------
        run_config : RunConfig
            An instance of ``RunConfig`` containing configuration details.

        Raises
        ------
        NotImplementedError
            If this method is not implemented by the subclasses.
        """
        raise NotImplementedError

    def _config_parser(self, run_config: RunConfig) -> RunConfig:
        """
        Internal method to process a run_config. It is called internally
        and wraps `config_parser`

        Parameters
        ----------
        run_config : RunConfig
            An instance of ``RunConfig`` containing configuration details.

        Returns
        -------
        RunConfig
            The processed configuration.
        """
        return self.config_parser(run_config)

    def _init_logger(self, resume: bool = False, debug: bool = False):
        """
        Initializes the logger used for recording experiment details and progress.

        Parameters
        ----------
        resume : bool
            If True, the logger will resume logging from a previous experiment, by default ``False``.
        debug : bool
            If True, no artifacts will be saved by the ``SummaryLogger``, by default ``False``.
        """
        self.logger = SummaryLogger(
            run_config=self.run_config,
            experiment_dir=self.experiment_dir if not debug else None,
            resume=resume,
            keep_n_checkpoints=self.run_config.keep_n_checkpoints,
            verbose=self.run_config.verbose == "console",
        )
        if butils.debugger_is_active() and not debug:
            self.logger.warn("Debug flag is False but running debugger.")
        elif debug:
            self.logger.warn("Debug flag is True, will not save any checkpoints.")
        if self.experiment_dir is not None:
            self.logger.info(f"Model directory: {self.experiment_dir}")

    def _make_dataloaders(
        self, run_config: RunConfig, data_lock: ty.Optional[Lock] = None
    ):
        """
        Creates the data loaders for the training process.

        Parameters
        ----------
        run_config : RunConfig
            An instance of ``RunConfig`` containing configuration details.
        data_lock: ty.Optional[Lock]
            A lock for multiprocessing context that prevents simultaneous processing and
            downloading of the dataset, by default ``None``.
        """
        context_lock: ty.Union[nullcontext, Lock]
        if data_lock is None:
            context_lock = nullcontext()
        else:
            context_lock = data_lock
        with context_lock:
            self.make_dataloaders(run_config)

        assert (
            len(self.train_dataloader) > 0
        ), "Must define a train dataloader in `make_dataloader`"

    def _parse_optim_metrics(
        self, run_config: RunConfig
    ) -> tuple[Optim, str] | tuple[None, None]:
        """
        parses the optimization metrics and their direction to validate they meet
        several training constraints. For example, the scheduler optimization mode
        should be aligned with the configuration optimization mode. Other configurations
        such as EarlyStopping also depend on the optimization metrics.

        Parameters
        ----------
        run_config : RunConfig
            The configuration to parse

        Returns
        -------
        tuple[Optim, str] | tuple[None, None]
            returns the optimization direction and metric name or a tuple of None if they are
            unspecified.

        Raises
        ------
        ValueError
            is raised when the optimization metrics are incompatible with other user configurations.
        """
        scheduler_config = run_config.train_config.scheduler_config

        optim_metric_name = run_config.optim_metric_name
        optim_metrics = run_config.optim_metrics
        missing_metrics = [
            run_config.optim_metrics is None,
            run_config.optim_metric_name is None,
        ]
        if any(missing_metrics) and not all(missing_metrics):
            raise ValueError(
                "Invalid configuration. Must specify both `optim_metrics` and"
                " `optim_metric_name` or neither."
            )
        optim_metric_name = str(optim_metric_name)
        if (
            optim_metric_name is not None
            and optim_metrics is not None
            and optim_metric_name not in optim_metrics
        ):
            raise ValueError(
                f"optim_metric_name={optim_metric_name} was not found in"
                f" optim_metrics={optim_metrics}"
            )
        if optim_metric_name is not None and optim_metrics is not None:
            optim_direction = optim_metrics[optim_metric_name]

        scheduler_requires_metric = (
            scheduler_config is not None
            and hasattr(scheduler_config, "arguments")
            and hasattr(scheduler_config.arguments, "mode")
        )
        if all(missing_metrics) and scheduler_requires_metric:
            raise ValueError(
                "Must provide `optim_metrics` when using Scheduler ="
                f" `{getattr(scheduler_config,'name', 'N/A')}`."
            )
        if all(missing_metrics) and run_config.early_stopping_iter is not None:
            raise ValueError(
                "Must provide `optim_metrics` when using early_stopping_iter ="
                f" `{run_config.early_stopping_iter}`."
            )
        if all(missing_metrics):
            return None, None
        if scheduler_requires_metric:
            mode = scheduler_config.arguments.mode  # type: ignore[union-attr]
            if (direction := optim_direction.value) != mode:
                self.logger.warn(
                    f"Different optim_metric_direction {direction} than"
                    f" scheduler.arguments.mode {mode}. Overwriting"
                    " scheduler.arguments.mode."
                )
        return optim_direction, optim_metric_name

    def _init_class_attributes(self):
        """
        Initializes the class attributes based on the provided configuration.

        This function sets up various class attributes related to device, mixed precision,
        warnings handling, early stopping, metrics, experiment and model directories, and
        process title.

        """
        run_config = self.run_config
        self.device = butils.parse_device(run_config.device)

        self.optim_metric_direction, self.optim_metric_name = self._parse_optim_metrics(
            run_config
        )
        if self.optim_metric_direction is not None:
            self.best_metrics = {
                self.optim_metric_name: (
                    float("inf")
                    if self.optim_metric_direction == Optim.min
                    else float("-inf")
                )
            }
        else:
            self.best_metrics = {}

        self.amp = run_config.amp
        if self.device == "cpu" and self.amp:
            self.logger.warn(
                "Automatic Mixed Precision (AMP) is not supported for CPU. Setting"
                " `amp` to False."
            )
            self.amp = False

        if (batch_lim := run_config.metrics_n_batches) > len(
            self.train_dataloader
        ) * 0.2:
            self.logger.warn(
                f"Metrics batch-limit {batch_lim} is larger than 20% of the train"
                f" dataloader length {len(self.train_dataloader)}. You might experience"
                " slow-down during training. Consider decreasing `metrics_n_batches`."
            )
        self._autocast = torch.autocast(
            enabled=self.amp,
            device_type="cuda" if "cuda" in self.device else "cpu",
        )

        self.verbose = run_config.verbose

        if self.verbose == "silent":
            warnings.filterwarnings("ignore")

        if (
            run_config.early_stopping_iter is not None
            and run_config.early_stopping_iter > 0
        ):
            assert self.val_dataloader is not None, (
                "dataloader function has to return validation set when setting early"
                " stopping to True"
            )

        self.train_metrics = Metrics(
            batch_limit=run_config.metrics_n_batches,
            memory_limit=int(run_config.metrics_mb_limit * 1e6),
            moving_average_limit=self.epoch_len,
            evaluation_functions=self.evaluation_functions(),
            static_aux_metrics=self.train_stats,
            moving_aux_metrics=["loss"],
        )
        if self.val_dataloader is not None:
            self.eval_metrics = Metrics(
                batch_limit=None,
                memory_limit=int(run_config.metrics_mb_limit * 1e6),
                moving_average_limit=None,
                evaluation_functions=self.evaluation_functions(),
                moving_aux_metrics=["loss"],
            )
        setproctitle.setproctitle(self.uid)

    def _init_model_state(
        self,
        resume: bool = False,
        smoke_test: bool = False,
        from_chkpt: str | Path | None = None,
    ):
        """
        Initializes the model state based on provided parameters and configuration.

        Parameters
        ----------
        resume : bool
            If True, tries to resume training from a checkpoint, by default ``False``.
        smoke_test : bool
            Whether to run as a smoke test, by default ``False``.
        from_chkpt: str | Path | None, optional
            Path to the checkpoint to initialize the state from.

        Raises
        ------
        RuntimeError
            If the directory containing checkpoints is not found.
        """

        if from_chkpt is not None:
            self.current_checkpoint = Path(from_chkpt)
            self._load_model(self.current_checkpoint, model_only=False)
        elif self.run_config.init_chkpt is not None and not resume:
            # Loads only the weights
            self.current_checkpoint = Path(self.run_config.init_chkpt)
            self.logger.info(
                "Initializing model weights ONLY from checkpoint."
                f" {self.current_checkpoint}"
            )

            self._load_model(self.current_checkpoint, model_only=True)

        elif resume and not smoke_test:
            if "recent" not in self.logger.CHKPT_DIRS:
                raise RuntimeError("Checkpoint folder was not found.")
            recent_checkpoint_dir = self.logger.CHKPT_DIRS["recent"]
            # NOTE: current_checkpoint is found in _find_load_valid_checkpoint
            self._find_load_valid_checkpoint(recent_checkpoint_dir)
        else:
            self.current_checkpoint = None
            if not smoke_test:
                self.logger.info("Creating new model")
            self.create_model()
            self._update_save_dict()

    def init_state(
        self,
        run_config: RunConfig,
        smoke_test: bool = False,
        debug: bool = False,
        resume: bool = False,
        remote_progress_bar: ty.Optional[RemoteProgressBar] = None,
        from_chkpt: Path | str | None = None,
        data_lock: ty.Optional[Lock] = None,
    ):
        """
        Initializes the state of the wrapper based on provided configuration and parameters.
        The lazy-initialization of the wrapper is neccessary, as the wrapper must be pickable.
        Initializing some objects (e.g. Dataloaders) leads to the opposite.

        Parameters
        ----------
        run_config : RunConfig
            An instance of ``RunConfig`` containing configuration details.
        smoke_test : bool
            Whether to run as a smoke test, by default ``False``.
        debug : bool
            If True, disables logging and model directory creation, by default ``False``.
        resume : bool
            If True, tries to resume training from a checkpoint, by default ``False``.
        remote_progress_bar : ty.Optional[RemoteProgressBar]
            A remote progress bar can be used to report metrics from the internal progress bar
        from_chkpt: Path | str | None, optional
            Path to the checkpoint to initialize the state from.
        data_lock: ty.Optional[Lock], optional
            Use a Lock to avoid downloading data concurrently.

        Raises
        ------
        RuntimeError
            if the state is already initialized and `smoke_test`, `debug` and `resume` flag are ``False``
        """
        if (
            self._is_init
            and not (smoke_test or debug or resume)
            # Check if the wrapper was previously initialized in dummy mode (e.g. debug or smoke_test)
            and not isinstance(getattr(self, "logger", butils.Dummy()), butils.Dummy)
        ):
            raise RuntimeError(f"{self.__class__.__name__} is already initialized. ")
        if self._is_init:
            self._reset_cached_attributes()
        self._init_attrs()
        self._is_init = True
        self.run_config = copy.deepcopy(run_config)
        self.run_config._unfreeze()  # pylint: disable=protected-access
        self.random_seed = self.run_config.random_seed
        if self.random_seed is not None:
            butils.set_seed(self.random_seed)
        self._make_dataloaders(self.run_config, data_lock=data_lock)

        self.run_config = self._config_parser(self.run_config)

        v = self.run_config.train_config.epochs
        self.__setattr__internal("epochs", v)

        if self.run_config.experiment_dir is not None:
            self.experiment_dir = expand_path(self.run_config.experiment_dir)
            self.run_config.experiment_dir = self.experiment_dir.as_posix()

        self.run_config.assert_unambigious()
        self.run_config.freeze()

        # Does not create log artifacts during smoke test
        if not smoke_test:
            self._init_logger(resume=resume, debug=debug)
        else:
            self.logger = butils.Dummy()

        self._init_class_attributes()
        if debug and self.experiment_dir is not None:
            self.logger.warn(
                f"Experiment Directory specified {self.experiment_dir} while running on"
                " debug mode. If saving artifacts is unnecessary you can disable the"
                " file system by setting `run_config.experiment_dir=None`. "
            )
        self._init_model_state(resume, smoke_test or debug, from_chkpt=from_chkpt)
        if self.verbose == "progress" and not smoke_test:
            self.progress_bar = ProgressBar(
                epoch_len=self.epoch_len,
                total_steps=self.total_steps,
                logfile=self.logger.log_file_path,
                remote_display=remote_progress_bar,
                uid=self.uid,
            )
        else:
            self.progress_bar = butils.Dummy()

    def _find_load_valid_checkpoint(self, chkpt_dir: Path):
        """
        Finds and loads the latest valid checkpoint from the given directory.

        Parameters
        ----------
        chkpt_dir : Path
            The directory containing the checkpoints.

        Raises
        ------
        CheckpointNotFoundError
            If no valid checkpoint is found in the specified directory.
        RuntimeError
            If a checkpoint is not found.
        """
        latest_checkpoints = butils.get_latest_chkpts(chkpt_dir)
        current_checkpoint = None
        if len(latest_checkpoints) > 0:
            # Try to load first valid chkpt in case there was a crash and some checkpoint is unrecoverable
            for i, _checkpoint in enumerate(latest_checkpoints):
                try:
                    self.logger.info(f"Loading checkpoint {_checkpoint}")
                    self._load_model(_checkpoint, model_only=False)
                    current_checkpoint = _checkpoint
                    break
                # pylint: disable=broad-exception-caught
                except Exception as e:
                    if i == len(latest_checkpoints) - 1:
                        # if it is the last checkpoint raise exception
                        raise RuntimeError("Checkpoint not found") from e

                    # ignore exception
                    self.logger.error(
                        f"Error loading checkpoint {_checkpoint}. Trying"
                        f" another....\n{traceback.format_exc()}"
                    )
        if current_checkpoint is None:
            raise CheckpointNotFoundError(
                f"Could not find a valid checkpoint in {chkpt_dir}"
            )
        self.current_checkpoint = current_checkpoint

    # flake8: noqa: DOC201
    def _load_model(self, checkpoint_path: Path, model_only: bool = False):
        """
        Loads the model and its state from the checkpoint file at the specified path.

        Parameters
        ----------
        checkpoint_path : Path
            The path to the checkpoint file containing the model and its state.
        model_only : bool
            If True, only the model's weights will be loaded, ignoring other state information, by default ``False``.

        Raises
        ------
        NotImplementedError
            If the model's run configuration is not initialized before attempting to load the model.
        RuntimeError
            If no valid checkpoint was found, such as an invalid path, and when ``model_only=True`` we check
            for differences between loaded and current configuration.

        """

        if not hasattr(self, "run_config") or self.run_config is None:
            raise NotImplementedError(
                "Can not load model on an uninitialized model state. Consider run"
                " init_experiment_state function first"
            )
        try:
            save_dict = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(
                f"{checkpoint_path} is not a valid checkpoint e.g. a `.pt` file. "
            ) from e
        if model_only:
            self.load_checkpoint(save_dict, model_only=model_only)
            return
        run_config = type(self.run_config)(**save_dict["run_config"])
        if run_config.uid != self.run_config.uid:
            diffs = "\n\t".join(
                run_config.diff_str(self.run_config, ignore_stateless=True)
            )
            raise RuntimeError(
                f"Mismatching loaded and current configurations. \n{diffs}"
            )
        diffs = "\n\t".join(
            run_config.diff_str(self.run_config, ignore_stateless=False)
        )
        if len(diffs) > 0:
            self.logger.warn(
                "Differences between initial configuration and current configuration."
                f" \n{diffs}"
            )
        self._load_stats(save_dict)
        self.load_checkpoint(save_dict, model_only=model_only)
        self.current_state = save_dict

    @abstractmethod
    def load_checkpoint(self, save_dict: dict[str, ty.Any], model_only: bool = False):
        """
        Abstract method to load the model and its state from a given save dictionary.

        Must be implemented by subclasses.
        Example implementation: Please see the ``load_checkpoint`` method in the ``ModelWrapper`` class.

        Parameters
        ----------
        save_dict : dict[str, ty.Any]
            A dictionary containing the saved model state and other necessary information.
        model_only : bool
            If ``True``, only the model's weights will be loaded, ignoring other state information, by
            default ``False``.

        Raises
        ------
        NotImplementedError
            If this method is not implemented by the subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def save_dict(self) -> dict[str, ty.Any] | None:
        """
        Abstract method to create and return a save dictionary containing the model's state
        and other necessary information.

        Must be implemented by subclasses.
        Example implementation: Please see the ``save_dict`` method in the ``ModelWrapper`` class.

        Returns
        -------
        dict[str, ty.Any] | None
            A dictionary containing the saved model state and other necessary information.

        Raises
        ------
        NotImplementedError
            If this method is not implemented by the subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluation_functions(self) -> dict[str, Callable] | None:
        """
        Abstract method to create and return a dictionary of evaluation functions used during
        training and validation.

        Must be implemented by subclasses.
        Example implementation: Please see the ``evaluation_functions`` method in the ``ModelWrapper`` class.

        Returns
        -------
        dict[str, Callable] | None
            A dictionary containing evaluation functions as values and their names as keys.

        Raises
        ------
        NotImplementedError
            If this method is not implemented by the subclasses.
        """
        raise NotImplementedError

    def __setattr__internal(self, k, v):
        super().__setattr__(k, v)

    def __setattr__(self, k, v):
        try:
            _derived_stats_names = super().__getattribute__("_derived_stats_names")
        except AttributeError:
            _derived_stats_names = []
        try:
            _overridable_stats_names = super().__getattribute__(
                "_overridable_stats_names"
            )
        except AttributeError:
            _overridable_stats_names = []
        if k in _derived_stats_names and k not in _overridable_stats_names:
            raise RuntimeError(f"Can not set derived attribute {k}.")
        self.__setattr__internal(k, v)

    def __getattribute__(self, name, _ignore_init=False):
        if name.startswith("_") or super().__getattribute__("_is_init"):
            return super().__getattribute__(name)

        try:
            _init_function_names = super().__getattribute__("_init_function_names")
        except AttributeError:
            _init_function_names = []
        if name not in _init_function_names and not _ignore_init:
            raise RuntimeError(
                f"Can not read property {name} of unitialized"
                f" {self.__class__.__name__}. It must be initialized with `init_state`"
                " before using."
            )
        return super().__getattribute__(name)

    def _load_stats(self, save_dict: dict) -> None:
        """
        Loads the saved training and validation metrics from the save_dict and updates
        the model's internal metrics with the loaded values.

        Parameters
        ----------
        save_dict : dict
            A dictionary containing the saved model state, metrics, and other necessary information.
        """
        metrics = copy.deepcopy(save_dict["train_metrics"])
        best_metrics = [f"best_{k}" for k in self.best_metrics]
        for k in self.train_stats:
            if k in self._running_stats_names:
                continue
            if k in self._derived_stats_names:
                # We skip assigning this metric as it will be derived by other metrics.
                if getattr(self, k, None) != metrics[k]:
                    self.logger.warn(
                        f"Current attribute {k} value derived to {getattr(self, k)} and"
                        f" is different than loaded value {metrics[k]}. Will use the"
                        " current value."
                    )
                del metrics[k]

        for k in self.train_stats:
            if k in metrics and k in best_metrics:
                self.best_metrics[k.lstrip("best_")] = metrics[k]
                del metrics[k]
            elif k in metrics:
                setattr(self, k, metrics[k])
                del metrics[k]

        self.train_metrics.update_static_metrics(self.train_stats)
        # pylint: disable=protected-access
        self.train_metrics._update_ma_metrics(metrics)

        if "eval_metrics" in save_dict and self.eval_metrics is not None:
            metrics = copy.deepcopy(save_dict["eval_metrics"])
            # pylint: disable=protected-access
            self.eval_metrics._update_ma_metrics(metrics)

    def _update_save_dict(self, user_save_dict: dict[str, ty.Any] | None = None):
        """
        Updates the current state dictionary with run_config and metrics. If a user_save_dict is provided,
        it is also merged into the current state dictionary.

        Parameters
        ----------
        user_save_dict : dict[str, ty.Any] | None
            A dictionary containing user-defined information to be saved, by default ``None``.
        """
        self.current_state = {
            "run_config": self.run_config.to_dict(),
            "train_metrics": self.train_metrics.to_dict(),
        }
        if self.eval_metrics is not None:
            self.current_state["eval_metrics"] = self.eval_metrics.to_dict()
        if user_save_dict is not None:
            self.current_state.update(**user_save_dict)

    def _checkpoint(self, is_best: bool = False):
        """
        Updates the current state dictionary with user-defined save_dict and calls the checkpoint method.

        Parameters
        ----------
        is_best : bool
            Indicates if the current checkpoint is the best model so far, by default ``False``.
        """
        user_save_dict = self.save_dict()
        self._update_save_dict(user_save_dict)
        self.checkpoint(is_best=is_best)

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
from tqdm import tqdm

import ablator.utils.base as butils
from ablator.main.configs import RunConfig
from ablator.modules.loggers.main import SummaryLogger
from ablator.modules.metrics.main import TrainMetrics
from ablator.utils.base import Dummy


class EvaluationError(Exception):
    pass


class TrainPlateauError(Exception):
    pass


class LogStepError(Exception):
    pass


class CheckpointNotFoundError(FileNotFoundError):
    pass


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
    logger : Union[SummaryLogger, tutils.Dummy]
        Records information on the program's operation and model training, such as progress and performance metrics.
    device : str
        The type of device used for running the experiment. i.e. ``"cuda"``, ``"cpu"``, ``"cuda:0"``.
    model_dir : Path
        The model directory.
    experiment_dir : Path
        The experiment directory.
    autocast : torch.autocast
        Enables autocasting for chosen regions. Autocasting automatically chooses the precision for GPU operations
        to improve performance while maintaining accuracy.
    verbose : bool
        If ``True``, prints additional information while training. Only applied for the master process.
    amp : bool
        If ``True``, apply automatic mixed precision training, otherwise default precision.
    random_seed : Optional[int]
        Sets the seed for generating random numbers.
    train_tqdm : tqdm, optional
        An optional instance of ``tqdm`` that creates progress bars and displays real-time information during training.
        i.e. time remaining. Only applied for the master process.
    current_checkpoint : Optional[Path]
        Directory for the current checkpoint file, by default None.
    metrics : Metrics
        Training metrics including model information. i.e. learning rate and loss value.
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
    best_loss : float
        The lowest loss value encountered during training.

    Notes
    -----
    1. Class properties are simply listed by name. Please check out property docstring for more information.

    2. Users must implement the abstract methods to customize the model's behavior.

    3. Mixed precision training enables some operations to use the ``torch.float32`` datatype and other operations use lower
    precision floating point datatype ``torch.float16``. This is for saving time and reducing memory usage. Ordinarily,
    "automatic mixed precision training" means training with ``torch.autocast`` and ``torch.cuda.amp.GradScaler`` together.
    More information: https://pytorch.org/docs/stable/amp.html

    """

    def __init__(
        self,
        model_class: type[nn.Module],
    ):
        """Initializes the ModelBase class with the required ``model_class`` and optional configurations.

        Parameters
        ----------
        model_class : type[nn.Module]
            The base class for user's model, which defines the neural network.
        """

        self.model_class = model_class
        self.run_config: RunConfig
        self.train_dataloader: DataLoader
        self.val_dataloader: DataLoader | None = None
        self.test_dataloader: DataLoader | None = None
        self.logger: ty.Union[SummaryLogger, Dummy]
        self.device: str
        self.model_dir: Path | None = None
        self.experiment_dir: Path | None = None
        self.autocast: torch.autocast
        self.verbose: ty.Literal["tqdm", "console", "silent"]
        self.amp: bool
        self.random_seed: ty.Optional[int]
        self.train_tqdm: tqdm = None

        self.current_checkpoint: Path | None = None
        # Runtime metrics
        self.metrics: TrainMetrics
        self.current_state: dict = {}

        # stats
        self.learning_rate = float("inf")
        self.total_steps: int
        self.epochs: int
        self.current_iteration = 0
        self.best_iteration = 0
        self.best_loss = float("inf")

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

            - best_loss: The best (lowest) loss value achieved during training.

        """
        return OrderedDict(
            learning_rate=self.learning_rate,
            total_steps=self.total_steps,
            epochs=self.epochs,
            current_epoch=self.current_epoch,
            current_iteration=self.current_iteration,
            best_iteration=self.best_iteration,
            best_loss=self.best_loss,
        )

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
    def epoch_len(self):
        """
        Returns the length of an epoch, which is the number of batches in the ``train_dataloader``.

        Returns
        -------
        int
            The length of an epoch, represented as the number of batches in the ``train_dataloader``.

        Raises
        ------
        AssertionError
            If the ``train_dataloader`` is not defined or its length is 0.
        """
        assert (
            hasattr(self, "train_dataloader") and len(self.train_dataloader) > 0
        ), "Undefined train_dataloader."
        return len(self.train_dataloader)

    @cached_property
    def eval_itr(self):
        """
        Calculate the interval between evaluations.

        Returns
        -------
        int
            The interval between evaluations.
        """
        return math.ceil(self.run_config.eval_epoch * self.epoch_len)

    @cached_property
    def log_itr(self):
        """
        Calculate the interval between logging steps.

        Returns
        -------
        int
            The interval between logging steps.
        """
        return math.ceil(self.run_config.log_epoch * self.epoch_len)

    @property
    def uid(self):
        """
        Returns a unique identifier (UID) for the current run configuration.

        Returns
        -------
        str
            A string representing the unique identifier of the current run configuration.
        """
        return self.run_config.uid

    def _get_process_name(self) -> str:
        """
        Retrieves the process name based on the model directory, experiment directory, or UID.

        Returns
        -------
        str
            The process name for the current instance.
        """
        if self.model_dir is not None and self.experiment_dir is not None:
            proc_title = self.model_dir.relative_to(
                self.experiment_dir.parent
            ).as_posix()
        else:
            proc_title = self.uid
        return proc_title

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
        save_dict : dict[str, ty.Any] | None, optional
            A dictionary containing saved model data, such as weights, optimizer state, etc.,
            to be loaded into the model, by default ``None``.
        strict_load : bool, optional
            If True, the model will be loaded strictly, ensuring that the saved state
            matches the model's structure exactly. If False, the model can be loaded
            with a partially matching state, by default ``True``.


        """
        raise NotImplementedError

    @abstractmethod
    def checkpoint(self, is_best=False):
        """
        Abstract method to save a checkpoint of the model. Must be implemented by subclasses.
        Example implementation: Please see the ``checkpoint`` method in the ``ModelWrapper`` class.


        Parameters
        ----------
        is_best : bool, optional
            Indicates if the current checkpoint is the best model so far, by default ``False``.
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
        smoke_test : bool, optional
            Whether to run as a smoke test, by default ``False``.
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
        """
        raise NotImplementedError

    def _init_logger(self, resume=False, debug=False):
        """
        Initializes the logger used for recording experiment details and progress.

        Parameters
        ----------
        resume : bool, optional
            If True, the logger will resume logging from a previous experiment, by default False.
        debug : bool, optional
            If True, logger will log additional debug information, by default False.
        """
        self.logger = SummaryLogger(
            run_config=self.run_config,
            model_dir=self.model_dir,
            resume=resume,
            keep_n_checkpoints=self.run_config.keep_n_checkpoints,
            verbose=self.run_config.verbose == "console",
        )
        if butils.debugger_is_active() and not debug:
            self.logger.warn("Debug flag is False but running in debug mode.")

        self.logger.info(f"Model directory: {self.model_dir}")

    def _make_dataloaders(self, run_config: RunConfig):
        """
        Creates the data loaders for the training process.

        Parameters
        ----------
        run_config : RunConfig
            An instance of ``RunConfig`` containing configuration details.
        """
        self.make_dataloaders(run_config)
        assert (
            len(self.train_dataloader) > 0
        ), "Must define a train dataloader in `make_dataloader`"
        self.epochs = self.run_config.train_config.epochs

    def _init_class_attributes(self, debug=False):
        """
        Initializes the class attributes based on the provided configuration.

        This function sets up various class attributes related to device, mixed precision,
        warnings handling, early stopping, metrics, experiment and model directories, and
        process title.

        Parameters
        ----------
        debug : bool, optional
            If True, disables the experiment and model directories creation, by default False.
        """
        run_config = self.run_config
        self.device = butils.parse_device(run_config.device)

        self.amp = run_config.amp
        if self.device == "cpu" and self.amp:
            raise ValueError(
                "AMP is not supported for CPU. You will need to set `run_config.amp` to False."
            )

        self.autocast = torch.autocast(
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
            assert (
                self.val_dataloader is not None
            ), "dataloader function has to return validation set when setting early stopping to True"

        self.metrics = TrainMetrics(
            batch_limit=run_config.metrics_n_batches,
            memory_limit=int(run_config.metrics_mb_limit * 1e6),
            moving_average_limit=self.epoch_len,
            evaluation_functions=self.evaluation_functions(),
            tags=["train"] + (["val"] if self.val_dataloader is not None else []),
            static_aux_metrics=self.train_stats,
            moving_aux_metrics=["loss"] + getattr(self, "aux_metric_names", []),
        )
        if self.run_config.experiment_dir is not None and not debug:
            self.experiment_dir = Path(self.run_config.experiment_dir)
            self.model_dir = self.experiment_dir.joinpath(self.uid)

        if debug and (self.experiment_dir is not None or self.model_dir is not None):
            self.experiment_dir = self.model_dir = None

        setproctitle.setproctitle(self._get_process_name())

    def _init_model_state(self, resume: bool = False, smoke_test: bool = False):
        """
        Initializes the model state based on provided parameters and configuration.

        Parameters
        ----------
        resume : bool, optional
            If True, tries to resume training from a checkpoint, by default False.
        smoke_test : bool, optional
            Whether to run as a smoke test, by default False.
        """
        if self.run_config.init_chkpt is not None and resume:
            self.current_checkpoint = Path(self.run_config.init_chkpt)
            self._load_model(self.current_checkpoint, model_only=False)

        elif self.run_config.init_chkpt is not None and not resume:
            # Loads only the weights
            self.current_checkpoint = Path(self.run_config.init_chkpt)
            self.logger.info(
                f"Initializing model weights ONLY from checkpoint. {self.current_checkpoint}"
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
            self.logger.info("Creating new model")
            self.create_model()
            self._update_save_dict()

    def _init_state(
        self,
        run_config: RunConfig,
        smoke_test: bool = False,
        debug: bool = False,
        resume: bool = False,
    ):
        """
        Initializes the state of the trainer based on provided configuration and parameters.

        Parameters
        ----------
        run_config : RunConfig
            An instance of ``RunConfig`` containing configuration details.
        smoke_test : bool, optional
            Whether to run as a smoke test, by default False.
        debug : bool, optional
            If True, disables logging and model directory creation, by default False.
        resume : bool, optional
            If True, tries to resume training from a checkpoint, by default False.
        """
        self.run_config = run_config
        self.random_seed = self.run_config.random_seed
        if self.random_seed is not None:
            butils.set_seed(self.random_seed)
        self.run_config = run_config
        _run_config = copy.deepcopy(run_config)
        self._make_dataloaders(self.run_config)

        self.run_config = self.config_parser(run_config)
        self._init_class_attributes(debug=debug)
        # Does not create log artifacts during smoke test
        if not smoke_test:
            self._init_logger(resume=resume, debug=debug)
        else:
            self.logger = butils.Dummy()
        self._init_model_state(resume, smoke_test)
        self.run_config.assert_state(_run_config)

        if self.verbose == "tqdm" and not smoke_test:
            self.train_tqdm = tqdm(
                total=self.epoch_len,
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                position=0,
                leave=True,
                dynamic_ncols=True,
            )
        else:
            self.train_tqdm = butils.Dummy()

    def _find_load_valid_checkpoint(self, chkpt_dir):
        """
        Finds and loads the latest valid checkpoint from the given directory.

        Parameters
        ----------
        chkpt_dir : str
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
                except Exception as e:
                    if i == len(latest_checkpoints) - 1:
                        # if it is the last checkpoint raise exception
                        raise RuntimeError("Checkpoint not found") from e

                    # ignore exception
                    self.logger.error(
                        f"Error loading checkpoint {_checkpoint}. Trying another....\n{traceback.format_exc()}"
                    )
        if current_checkpoint is None:
            raise CheckpointNotFoundError(f"Could not find a valid checkpoint in {chkpt_dir}")
        self.current_checkpoint = current_checkpoint

    def _load_model(self, checkpoint_path: Path, model_only: bool = False) -> None:
        """
        Loads the model and its state from the checkpoint file at the specified path.

        Parameters
        ----------
        checkpoint_path : Path
            The path to the checkpoint file containing the model and its state.
        model_only : bool, optional, default=False
            If True, only the model's weights will be loaded, ignoring other state information.

        Raises
        ------
        NotImplementedError
            If the model's run configuration is not initialized before attempting to load the model.
        """

        if not hasattr(self, "run_config") or self.run_config is None:
            raise NotImplementedError(
                "Can not load model on an unitialzed model state. Consider run init_experiment_state function first"
            )

        save_dict = torch.load(checkpoint_path, map_location="cpu")

        run_config = type(self.run_config)(**save_dict["run_config"])
        assert run_config.uid == self.run_config.uid

        self._load_stats(save_dict)
        self.load_checkpoint(save_dict, model_only=model_only)
        self.current_state = save_dict

    @abstractmethod
    def load_checkpoint(
        self, save_dict: dict[str, ty.Any], model_only: bool = False
    ) -> None:
        """
        Abstract method to load the model and its state from a given save dictionary.

        Must be implemented by subclasses.
        Example implementation: Please see the ``load_checkpoint`` method in the ``ModelWrapper`` class.

        Parameters
        ----------
        save_dict : dict[str, ty.Any]
            A dictionary containing the saved model state and other necessary information.
        model_only : bool, optional, default=False
            If ``True``, only the model's weights will be loaded, ignoring other state information.
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
        """
        raise NotImplementedError

    def _load_stats(self, save_dict) -> None:
        """
        Loads the saved training and validation metrics from the save_dict and updates
        the model's internal metrics with the loaded values.

        Parameters
        ----------
        save_dict : dict
            A dictionary containing the saved model state, metrics, and other necessary information.
        """
        metrics = copy.deepcopy(save_dict["metrics"])

        for k in self.train_stats:
            if (
                isinstance(getattr(type(self), k, None), property)
                and getattr(type(self), k).fset is None
            ):
                if getattr(self, k, None) != metrics[k]:
                    self.logger.warn(
                        f"Immutable class attribute {k} value {getattr(self, k)} "
                        f"different than loaded value {metrics[k]}"
                    )
                del metrics[k]

        for k in self.train_stats:
            if k in metrics:
                setattr(self, k, metrics[k])
                del metrics[k]

        self.metrics.update_static_metrics(self.train_stats)
        tags = {m.split("_")[0] for m in metrics}
        metric_names = {m.split("_")[1] for m in metrics}

        for tag in tags:
            self.metrics.update_ma_metrics(
                {m: metrics[f"{tag}_{m}"] for m in metric_names}, tag=tag
            )

    def _update_save_dict(self, user_save_dict: dict[str, ty.Any] | None = None):
        """
        Updates the current state dictionary with run_config and metrics. If a user_save_dict is provided,
        it is also merged into the current state dictionary.

        Parameters
        ----------
        user_save_dict : dict[str, ty.Any] | None, optional
            A dictionary containing user-defined information to be saved, by default None.
        """
        self.current_state = {
            "run_config": self.run_config.to_dict(),
            "metrics": self.metrics.to_dict(),
        }
        if user_save_dict is not None:
            self.current_state.update(**user_save_dict)

    def _checkpoint(self, is_best=False):
        """
        Updates the current state dictionary with user-defined save_dict and calls the checkpoint method.

        Parameters
        ----------
        is_best : bool, optional
            Indicates if the current checkpoint is the best model so far, by default False.
        """
        user_save_dict = self.save_dict()
        self._update_save_dict(user_save_dict)
        self.checkpoint(is_best=is_best)

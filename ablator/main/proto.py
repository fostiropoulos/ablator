import copy
from copy import deepcopy
from pathlib import Path

import git
from git import exc
import torch

from ablator.config.proto import RunConfig
from ablator.main.model.wrapper import ModelWrapper


class ProtoTrainer:
    """
    Manages resources for Prototyping.

    Attributes
    ----------
    wrapper : ModelWrapper
        The main model wrapper.
    run_config : RunConfig
        Running configuration for the model.

    Raises
    ------
    RuntimeError
        If experiment directory is not defined in the running configuration.
    """

    def __init__(
        self,
        wrapper: ModelWrapper,
        run_config: RunConfig,
    ):
        """
        Initialize model wrapper and running configuration for the model.

        Parameters
        ----------
        wrapper : ModelWrapper
            The main model wrapper.
        run_config : RunConfig
            Running configuration for the model.
        """
        super().__init__()
        self.wrapper = copy.deepcopy(wrapper)
        self.run_config: RunConfig = copy.deepcopy(run_config)
        if self.run_config.experiment_dir is None:
            raise RuntimeError("Must specify an experiment directory.")
        experiment_dir = self.run_config.experiment_dir
        experiment_path = Path(experiment_dir).absolute().resolve()
        self.experiment_dir = experiment_path
        self.run_config.experiment_dir = experiment_dir

    def pre_train_setup(self):
        """
        Used to prepare resources to avoid stalling during training or when resources are
        shared between trainers.
        """

    def _mount(self):
        # TODO
        # mount experiment directory
        # https://rclone.org/commands/rclone_mount/
        pass

    def _get_diffs(self, working_dir: str = ""):
        try:
            repo = git.Repo(Path(working_dir).resolve().absolute().as_posix())
            t = repo.head.commit.tree
            diffs = repo.git.diff(t)
            return f"Git Diffs for {repo.head.ref} @ {repo.head.commit}: \n{diffs}"
        except ValueError as e:
            raise RuntimeError(
                f"Could not parse repo at {working_dir}. Error: {str(e)}"
            ) from e
        except exc.NoSuchPathError as e:
            raise FileNotFoundError(f"Directory {working_dir} was not found. ") from e
        except exc.InvalidGitRepositoryError as e:
            return (
                f"No git repository was detected at {working_dir}. "
                "We recommend setting the working directory to a git repository "
                "to keep track of changes."
            )

    def launch(self, working_directory: str, debug: bool = False):
        """
        Initialize the data state of the wrapper and train the model inside the wrapper, then sync training
        results (logged to experiment directory while training) with external logging services (e.g Google
        cloud storage, other remote servers).

        Parameters
        ----------
        working_directory : str
            The working directory points to a git repository that is used for keeping track
            the code differences.
        debug : bool, default=False
            Whether to train model in debug mode.

        Returns
        -------
        metrics : Metrics
            Metrics returned after training.
        """
        self._mount()
        self.pre_train_setup()
        self.wrapper.init_state(
            run_config=self.run_config, smoke_test=False, debug=debug, resume=False
        )
        diffs = self._get_diffs(working_directory)
        self.wrapper.logger.info(diffs)
        metrics = self.wrapper.train(debug=debug)
        return metrics

    def evaluate(self):
        """
        Run model evaluation on the training results, sync evaluation results to external logging services
        (e.g Google cloud storage, other remote servers).

        Returns
        -------
        metrics : Metrics
            Metrics returned after evaluation.
        """
        # TODO load model if it is un-trained
        metrics = self.wrapper.evaluate(self.run_config)
        return metrics

    def smoke_test(self, config=None):
        """
        Run a smoke test training process on the model.

        Parameters
        ----------
        config : RunConfig
            Running configuration for the model.
        """
        if config is None:
            config = self.run_config
        run_config = deepcopy(config)
        wrapper = deepcopy(self.wrapper)
        wrapper.train(run_config=run_config, smoke_test=True)
        del wrapper
        torch.cuda.empty_cache()

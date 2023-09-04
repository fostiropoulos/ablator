import copy
import typing as ty
from copy import deepcopy
from pathlib import Path

import git
import torch
from git import exc

from ablator.config.proto import RunConfig
from ablator.main.model.wrapper import ModelWrapper


class ProtoTrainer:
    """
    Manages resources for Prototyping. This trainer runs an experiment of a single prototype model. (Therefore no HPO)

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

    Raises
    ------
    RuntimeError
        If experiment directory is not defined in the running configuration.

    Examples
    --------
    Below is a complete workflow on how to launch a prototype experiment with ``ProtoTrainer``, from defining config to
    launching the experiment:

    - Define training config:

    >>> my_optim_config = OptimizerConfig("sgd", {"lr": 0.5, "weight_decay": 0.5})
    >>> my_scheduler_config = SchedulerConfig("step", arguments={"step_size": 1, "gamma": 0.99})
    >>> train_config = TrainConfig(
    ...     dataset="[Dataset Name]",
    ...     batch_size=32,
    ...     epochs=10,
    ...     optimizer_config = my_optimizer_config,
    ...     scheduler_config = my_scheduler_config,
    ...     rand_weights_init = True
    ... )

    - Define model config, here we use the default one with no custom hyperparameters (sometimes you would
      want to customize the model config to run HPO on your model's hyperparameters in the parallel experiments,
      which uses ``ParallelTrainer`` and ``ParallelConfig`` instead of ``ProtoTrainer`` and ``RunConfig``):

    >>> model_config = ModelConfig()

    - Define run config:

    >>> run_config = CustomRunConfig(
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
    >>>     def make_dataloader_train(self, run_config: CustomRunConfig):
    >>>         return torch.utils.data.DataLoader(<train_dataset>, batch_size=32, shuffle=True)
    >>>
    >>>     def make_dataloader_val(self, run_config: CustomRunConfig):
    >>>         return torch.utils.data.DataLoader(<val_dataset>, batch_size=32, shuffle=False)

    - After gathering all configurations and model wrapper, it's time we initialize and launch the prototype trainer:

    >>> wrapper = MyModelWrapper(
    ...     model_class=<your_ModelModule_class>,
    ... )
    >>> ablator = ProtoTrainer(
    ...     wrapper=wrapper,
    ...     run_config=run_config,
    ... )
    >>> metrics = ablator.launch()
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
        except exc.InvalidGitRepositoryError:
            return (
                f"No git repository was detected at {working_dir}. "
                "We recommend setting the working directory to a git repository "
                "to keep track of changes."
            )

    def launch(self, working_directory: str, debug: bool = False) -> dict[str, float]:
        """
        Launch the prototype experiment (train, evaluate the single prototype model) and return metrics.

        Parameters
        ----------
        working_directory : str
            The working directory points to a git repository that is used for keeping track
            the code differences.
        debug : bool, optional
            Whether to train model in debug mode. By default False

        Returns
        -------
        metrics : dict[str, float]
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

    def evaluate(self) -> dict[str, dict[str, ty.Any]]:
        """
        Run model evaluation on the training results, sync evaluation results to external logging services
        (e.g Google cloud storage, other remote servers).

        Returns
        -------
        metrics : dict[str, dict[str, ty.Any]]
            Metrics returned after evaluation.
        """
        # TODO load model if it is un-trained
        metrics = self.wrapper.evaluate(self.run_config)
        return metrics

    def smoke_test(self, config: RunConfig | None = None):
        """
        Run a smoke test training process on the model.

        Parameters
        ----------
        config : RunConfig | None
            Running configuration for the model.

        Examples
        --------
        try:
            ablator.smoke_test(run_config)
        except err:
            raise err
        """
        if config is None:
            config = self.run_config
        run_config = deepcopy(config)
        wrapper = deepcopy(self.wrapper)
        wrapper.train(run_config=run_config, smoke_test=True)
        del wrapper
        torch.cuda.empty_cache()

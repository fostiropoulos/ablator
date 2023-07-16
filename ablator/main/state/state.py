import traceback
import builtins
import copy
import random
import typing as ty
from collections import OrderedDict
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.exc import MultipleResultsFound
from sqlalchemy.orm import Session

import ablator.utils.base as butils
from ablator.config.mp import (
    ParallelConfig,
    SearchAlgo,
)
from ablator.main.hpo import BaseSampler, GridSampler, OptunaSampler
from ablator.modules.loggers.file import FileLogger
from ablator.main.state.store import TrialState, Trial
from ablator.main.state._utils import (
    _verify_metrics,
    augment_trial_kwargs,
    _parse_metrics,
)


class ExperimentState:
    def __init__(
        self,
        experiment_dir: Path,
        config: ParallelConfig,
        logger: FileLogger | None = None,
        resume: bool = False,
        sampler_seed: int | None = None,
    ) -> None:
        """
        Initializes the ExperimentState.
        Initialize databases for storing training states and a sampler
        Create trials based on total num of trials specified in config

        Parameters
        ----------
        experiment_dir : Path
            The directory where the experiment data will be stored.
        config : ParallelConfig
            The configuration object that defines the experiment settings.
        logger : FileLogger, optional
            The logger to use for outputting experiment logs. If not specified, a dummy logger will be used.
        resume : bool, optional
            Whether to resume a previously interrupted experiment. Default is ``False``.
        sampler_seed : int | None
            The seed to use for the trial sampler. Default is ``None``.

        Raises
        ------
        RuntimeError
            If the specified ``search_space`` parameter is not found in the configuration.
        AssertionError
            If ``config.search_space`` is empty.
        RuntimeError
            if the experiment database already exists and ``resume`` is ``False``.
        """
        self.config = config
        self.logger: FileLogger = logger if logger is not None else butils.Dummy()  # type: ignore

        default_vals = [
            v
            for v in self.config.make_dict(self.config.annotations, flatten=True)
            if not v.startswith("search_space")
        ]
        assert len(self.config.search_space), "Must specify a config.search_space."
        paths = [
            f"{k}.{p}" if len(p) > 0 else k
            for k, v in self.config.search_space.items()
            for p in v.make_paths()
        ]
        for p in paths:
            if p not in default_vals:
                raise RuntimeError(
                    f"SearchSpace parameter {p} was not found in the configuration {sorted(default_vals)}."
                )
        study_name = config.uid
        self.experiment_dir = experiment_dir

        experiment_state_db = experiment_dir.joinpath(f"{study_name}_state.db")
        if experiment_state_db.exists() and not resume:
            raise RuntimeError(
                f"{experiment_state_db} exists. Please remove before starting another experiment or set `resume=True`."
            )

        self.engine = create_engine(f"sqlite:///{experiment_state_db}", echo=False)
        Trial.metadata.create_all(self.engine)

        search_algo = self.config.search_algo

        search_space = self.config.search_space
        self.optim_metrics = (
            OrderedDict(self.config.optim_metrics)
            if self.config.optim_metrics is not None
            else OrderedDict({})
        )
        self._ignore_errored_trials = self.config.ignore_invalid_params
        self.sampler: BaseSampler
        # TODO unit-test for resuming with different sampler
        if search_algo in {SearchAlgo.random, SearchAlgo.tpe} or search_algo is None:
            self.sampler = OptunaSampler(
                search_algo,
                search_space,
                self.optim_metrics,
                self.valid_trials(),
                seed=sampler_seed,
            )
        elif search_algo == SearchAlgo.grid:
            if len(self.optim_metrics):
                raise RuntimeError("Can not specify `optim_metrics` with GridSampler.")
            # TODO unit-test resuming with GridSampler for experiment state
            aug_cs: list[dict[str, ty.Any]] = [
                dict(c.aug_config_param) for c in self.valid_trials()
            ]
            self.sampler = GridSampler(search_space, aug_cs, seed=sampler_seed)
        else:
            raise NotImplementedError
        for trial in self.get_trials_by_state(TrialState.RUNNING):
            # mypy error for sqlalchemy types
            trial_id = int(trial.trial_num)  # type: ignore
            self.update_trial_state(trial_id, None, TrialState.WAITING)

    @staticmethod
    def search_space_dot_path(trial: ParallelConfig) -> dict[str, ty.Any]:
        """
        Returns a dictionary of parameter names and their corresponding values for a given trial.

        Parameters
        ----------
        trial : ParallelConfig
            The trial object to get the search space dot paths from.

        Returns
        -------
        dict[str, Any]
            A dictionary of parameter names and their corresponding values.

        Examples
        --------
        >>> search_space = {"train_config.optimizer_config.arguments.lr":
        SearchSpace(value_range=[0, 0.1], value_type="float")}
        >>> {"train_config.optimizer_config.arguments.lr": 0.1}
        """
        return {
            dot_path: trial.get_val_with_dot_path(dot_path)
            for dot_path in trial.search_space.keys()
        }

    @staticmethod
    def tune_trial_str(trial: ParallelConfig) -> str:
        """
        Generate a string representation of a trial object.

        Parameters
        ----------
        trial : ParallelConfig
            The trial object to generate a string representation for.

        Returns
        -------
        str
            A string representation of the trial object.
        """
        trial_map = ExperimentState.search_space_dot_path(trial)
        msg = f"\n{trial.uid}:\n\t"
        msg = "\n\t".join(
            [f"{dot_path} -> {val} " for dot_path, val in trial_map.items()]
        )

        return msg

    def sample_trial(self) -> tuple[int, ParallelConfig]:
        """
        Samples a trial from the search space and persists the trial state to the experiment database.

        Returns
        -------
        tuple[int, ParallelConfig]
            The unique trial_id with respect to the sampler, and the trial configuration.

        Raises
        ------
        StopIteration
            If the number of invalid trials sampled exceeds the internal upper bound (`20`) or the
            sampler raises a StopIteration exception indicating that the search space has been exhaustively
            evaluated.
        TypeError
            If the trial parameter are invalid and `config.ignore_invalid_params` is set to False
        """
        # Return pending trials when sampling first.
        pending_trials = self.get_trials_by_state(TrialState.WAITING)
        if len(pending_trials) > 0:
            trial = random.choice(pending_trials)
            # mypy errors for sqlalchemy types
            trial_id = int(trial.trial_num)  # type: ignore
            trial_config = type(self.config)(**trial.config_param)  # type: ignore
            self._update_internal_trial_state(trial_id, None, TrialState.RUNNING)
            return trial_id, trial_config

        trial_id, trial_config = self.__sample_trial(
            ignore_errors=self._ignore_errored_trials,
        )
        return trial_id, trial_config

    def __sample_trial(
        self,
        ignore_errors=False,
    ) -> tuple[int, ParallelConfig]:
        error_upper_bound = 20
        errored_trials = 0
        i = 0
        while i < error_upper_bound:
            drop = False
            try:
                # NOTE _optuna args is a monkey-patch for optuna compatibility
                # We store information about the sampling distribution to be able
                # to restore the sampler.
                trial_id, config, _optuna_args = self.sampler.eager_sample()
            except StopIteration as e:
                raise StopIteration(
                    f"Reached maximum number of trials, for sampler `{self.sampler.__class__.__name__}`."
                ) from e
            trial_kwargs = augment_trial_kwargs(
                trial_kwargs=self.config.to_dict(), augmentation=config
            )

            try:
                trial_config = type(self.config)(**trial_kwargs)
                trial_uid = trial_config.uid
            # pylint: disable=broad-exception-caught
            except builtins.Exception as e:
                if ignore_errors:
                    excp = traceback.format_exc()
                    self.logger.warn(f"ignoring: {config}. \n{excp}")
                    drop = True
                    errored_trials += 1

                else:
                    raise TypeError(f"Invalid trial parameters {config}") from e
            finally:
                self.sampler.unlock(drop)
                i += 1
            if not drop:
                # NOTE we want to update outside the try / except because we want to raise
                # errors for when adding the trial.
                trial_state: TrialState = TrialState.RUNNING
                if _optuna_args is None:
                    _optuna_args = {}
                self._append_trial_internal(
                    config_uid=trial_uid,
                    trial_kwargs=trial_kwargs,
                    trial_aug_kwargs=config,
                    trial_num=trial_id,
                    trial_state=trial_state,
                    **_optuna_args,
                )
                return trial_id, trial_config

        raise StopIteration(
            (
                f"Reached maximum limit of misconfigured trials, {error_upper_bound} "
                f"with {errored_trials} invalid trials."
            )
        )

    def update_trial_state(
        self,
        trial_id: int,
        metrics: dict[str, float] | None = None,
        state: TrialState = TrialState.RUNNING,
    ) -> None:
        """
        Update the state of a trial in both the Experiment database and tell Optuna.

        Parameters
        ----------
        trial_id : int
            The id of the trial to update.
        metrics : dict[str, float] | None, optional
            The metrics of the trial, by default ``None``.
        state : TrialState, optional
            The state of the trial, by default ``TrialState.RUNNING``.

        Examples
        --------
        >>> experiment.update_trial_state("fje_2211", {"loss": 0.1}, TrialState.COMPLETED)
        """
        if state == TrialState.FAIL_RECOVERABLE:
            self._inc_error_count(trial_id, state)
            return
        # TODO unit test
        internal_metrics = _parse_metrics(self.optim_metrics, metrics)
        _verify_metrics(internal_metrics)
        self.sampler.update_trial(trial_id, internal_metrics, state)
        try:
            self._update_internal_trial_state(trial_id, internal_metrics, state)
        except MultipleResultsFound as e:
            raise RuntimeError(
                "Corrupt experiment state, with repeating trials. "
            ) from e

    def _update_internal_trial_state(
        self, trial_id: int, metrics: dict[str, float] | None, state: TrialState
    ):
        """
        Update the state of a trial in the Experiment state database.

        Parameters
        ----------
        trial_id : int
            The id of the trial to update.
        metrics : dict[str, float] | None
            The metrics of the trial.
        state : TrialState
            The state of the trial.

        Returns
        -------
        bool
            True if the update was successful.
        """

        with Session(self.engine) as session:
            stmt = select(Trial).where(Trial.trial_num == trial_id)
            if (res := session.execute(stmt).scalar_one_or_none()) is None:
                raise RuntimeError(f"Trial {trial_id} was not found.")
            if metrics is not None:
                res.metrics.append(metrics)
            res.state = state  # type: ignore # TODO fix this
            session.commit()
            session.flush()

        return True

    def _inc_error_count(self, trial_id: int, state: TrialState):
        with Session(self.engine) as session:
            stmt = select(Trial).where(Trial.trial_num == trial_id)
            res = session.execute(stmt).scalar_one()
            assert state == TrialState.FAIL_RECOVERABLE
            runtime_errors = copy.deepcopy(res.runtime_errors)
            res.runtime_errors = Trial.runtime_errors + 1
            session.commit()
            session.flush()

        if runtime_errors < 10:
            self.logger.warn(f"Trial {trial_id} failed {runtime_errors+1} times.")
            self.update_trial_state(trial_id, None, TrialState.WAITING)
        else:
            self.logger.error(
                f"Trial {trial_id} exceed limit of runtime errors {runtime_errors}. Skipping."
            )
            self.update_trial_state(trial_id, None, TrialState.FAIL)

    def _append_trial_internal(
        self,
        config_uid: str,
        trial_kwargs: dict[str, ty.Any],
        trial_aug_kwargs: dict[str, ty.Any],
        trial_num: int,
        trial_state: TrialState,
        _opt_distributions_kwargs: dict[str, ty.Any] | None = None,
        _opt_distributions_types: dict[str, str] | None = None,
        _opt_params: dict[str, ty.Any] | None = None,
    ):
        """
        Append a trial to the Experiment state database.

        Parameters
        ----------
        config_uid : str
            The uid of the trial to update.
        trial_kwargs : dict[str, ty.Any]
            config dict with new sampled hyperparameters.
        trial_aug_kwargs : dict[str, ty.Any]
            the sampled trial keywords as opposed to the complete
            configuration from `trial_kwargs`
        trial_num : int
            The optuna trial number.
        trial_state : TrialState
            The state of the trial.
        """
        with Session(self.engine) as session:
            trial = Trial(
                config_uid=config_uid,
                config_param=trial_kwargs,
                aug_config_param=trial_aug_kwargs,
                trial_num=trial_num,
                state=trial_state,
                metrics=[],
                _opt_distributions_kwargs=_opt_distributions_kwargs,
                _opt_distributions_types=_opt_distributions_types,
                _opt_params=_opt_params,
            )
            session.add(trial)
            session.commit()

    def _get_trials_by_stmt(self, stmt) -> list[Trial]:
        with self.engine.connect() as conn:
            trials: list[Trial] = conn.execute(stmt).fetchall()  # type: ignore
        return trials

    def valid_trials_id(self) -> list[int]:
        return [c.id for c in self.valid_trials()]

    def valid_trials(self) -> list[Trial]:
        stmt = select(Trial).where(
            (Trial.state != TrialState.PRUNED_DUPLICATE)
            & (Trial.state != TrialState.PRUNED_INVALID)
        )
        trials = self._get_trials_by_stmt(stmt)
        return trials

    def get_trials_by_state(self, state: TrialState) -> list[Trial]:
        assert state in {
            TrialState.PRUNED,
            TrialState.COMPLETE,
            TrialState.PRUNED_INVALID,
            TrialState.PRUNED_DUPLICATE,
            TrialState.RUNNING,
            TrialState.WAITING,
            TrialState.COMPLETE,
            TrialState.FAIL,
        }
        stmt = select(Trial).where((Trial.state == state))

        trials = self._get_trials_by_stmt(stmt)
        return trials

    def get_trial_configs_by_state(self, state: TrialState) -> list[ParallelConfig]:
        assert (
            state != TrialState.PRUNED_INVALID
        ), "Can not return configuration for invalid trials due to configuration errors."
        configs = []
        trials = self.get_trials_by_state(state)
        for trial in trials:
            trial_config = type(self.config)(**dict(trial.config_param))
            configs.append(trial_config)
            assert trial_config.uid == trial.config_uid
        return configs

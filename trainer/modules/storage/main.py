import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
from sqlalchemy import String, PickleType, Column, create_engine, select, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from trainer.config.run import ParallelConfig

import enum

from optuna.trial import TrialState as OptunaTrialState

from trainer.config.main import ConfigBase
from trainer.config.run import ExperimentType, ParallelConfig, Optim
from trainer.modules.logging.file import FileLogger
from trainer.modules.main import Metrics
from trainer.utils.config import nested_set
from trainer.utils.train import Dummy

from sqlalchemy.orm import Session


class Base(DeclarativeBase):
    pass


class TrialState(enum.IntEnum):
    # extension of "optuna.trial.TrialState"
    RUNNING = 0  # A trial that has been succesfully scheduled to run
    COMPLETE = 1  # Succesfully completed trial
    PRUNED = 2  # Trial pruned because of various reasons
    FAIL = 3  # Trial that produced an error during execution
    WAITING = 4  # Trial that has been sampled but is not scheduled to run yet
    PRUNED_INVALID = 5  # Trial that was pruned during sampling as it was invalid
    PRUNED_DUPLICATE = 6  # Trial that was sampled but was already present
    PRUNED_POOR_PERFORMANCE = (
        7  # Trial that was pruned during execution for poor performance
    )
    RECOVERABLE_ERROR = 8  # Trial that was pruned during execution for poor performance

    def to_optuna_state(self) -> OptunaTrialState:
        if self in [
            TrialState.PRUNED,
            TrialState.PRUNED_INVALID,
            TrialState.PRUNED_DUPLICATE,
            TrialState.PRUNED_POOR_PERFORMANCE,
        ]:
            return OptunaTrialState.PRUNED
        else:
            return OptunaTrialState(self)


class Trial(Base):
    __tablename__ = "trial"
    id: Mapped[int] = mapped_column(primary_key=True)
    config_uid: Mapped[str] = mapped_column(String(30))
    metrics: Mapped[PickleType] = mapped_column(PickleType)
    config_param: Mapped[PickleType] = mapped_column(PickleType)
    optuna_trial_num: Mapped[str] = mapped_column(Integer)
    state: Mapped[PickleType] = mapped_column(PickleType, default=TrialState.WAITING)
    runtime_errors: Mapped[int] = mapped_column(Integer, default=0)

    def __repr__(self) -> str:
        return f"Trial(id={self.id!r}, config_uid={self.config_uid!r}, fullname={self.config_param!r})"


class ExperimentState:
    def __init__(
        self,
        experiment_dir: Path,
        config: ParallelConfig,
        logger: Optional[FileLogger] = None,
    ) -> None:

        self.optuna_trial_map: Dict[str, optuna.Trial] = {}
        self.config = config
        self.logger = logger if logger is not None else Dummy
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.search_space = config.tune.items()
        study_name = config.uid
        self.experiment_dir = experiment_dir
        experiment_state_db = experiment_dir.joinpath(f"{study_name}_state.db")
        optuna_db_path = experiment_dir.joinpath(f"{study_name}_optuna.db")

        sampler: optuna.samplers.BaseSampler
        if config.experiment_type == ExperimentType.grid:
            sampler = optuna.samplers.GridSampler(dict(self.search_space))
        elif config.experiment_type == ExperimentType.random:
            sampler = optuna.samplers.RandomSampler()
        elif config.experiment_type == ExperimentType.tpe:
            sampler = optuna.samplers.TPESampler()
        else:
            raise NotImplementedError

        if optuna_db_path.exists() and not config.train_config.resume:
            raise RuntimeError(
                f"{optuna_db_path} exists. Please remove before starting a study."
            )

        storage = f"sqlite:///{optuna_db_path}"
        if config.optim_directions is None:
            raise ValueError("Must specify optim_directions in the configuration")

        directions = [
            optuna.study.StudyDirection(2 if Optim(v) == Optim.max else 1)
            for k, v in config.optim_directions
        ]

        self.optuna_study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            directions=directions,
            load_if_exists=True,
            sampler=sampler,
        )
        self.engine = create_engine(f"sqlite:///{experiment_state_db}", echo=False)
        Trial.metadata.create_all(self.engine)

        self._init_trials()

    @staticmethod
    def format_trial_dot_path(trial: ParallelConfig) -> Dict[str, Any]:
        return {
            dot_path: trial.get_val_with_dot_path(dot_path)
            for dot_path in trial.tune.keys()
        }

    def _init_trials(self) -> List[ParallelConfig]:

        max_trials_conc = min(self.config.concurrent_trials, self.config.total_trials)

        if self.config.experiment_type in [
            ExperimentType.grid,
            ExperimentType.random,
        ]:
            trials_to_sample = self.n_trials_remaining
        else:
            trials_to_sample = max_trials_conc

        # if there are currently running trials, return those first and is resume. Otherwise
        # ERROR

        if len(self.running_trials) > 0 and not self.config.train_config.resume:
            raise RuntimeError(
                "Experiment was interupted and is attempted to be re-initialized. You need to use `.resume = True`"
            )
        running_trials = []
        for trial in self.running_trials:
            self.update_trial_state(trial.uid, None, TrialState.WAITING)
            trial.train_config.resume = True
            running_trials.append(trial)

        trials = self.__sample_trials(
            trials_to_sample,
            running_trials + self.pending_trials,
            ignore_errors=self.config.ignore_errored_trials,
        )[:max_trials_conc]

        for trial in trials:
            self.update_trial_state(trial.uid, None, TrialState.RUNNING)
        assert len(self.running_trials) > 0, "No trials could be scheduled."
        return trials

    @staticmethod
    def augment_trial_kwargs(
        trial_kwargs: Dict[str, Any], augmentation: Dict[str, Any]
    ) -> Dict[str, Any]:

        trial_kwargs = copy.deepcopy(trial_kwargs)
        config_dot_path: str
        dot_paths = list(augmentation.keys())

        assert len(set(dot_paths)) == len(
            dot_paths
        ), f"Duplicate tune paths: {set(dot_paths).difference(dot_paths)}"
        for config_dot_path, val in augmentation.items():

            path: List[str] = config_dot_path.split(".")
            trial_kwargs = nested_set(trial_kwargs, path, val)
        return trial_kwargs

    @staticmethod
    def tune_trial_str(trial: ParallelConfig) -> str:
        trial_map = ExperimentState.format_trial_dot_path(trial)
        msg = f"\n{trial.uid}:\n\t"
        msg = "\n\t".join(
            [f"{dot_path} -> {val} " for dot_path, val in trial_map.items()]
        )
        return msg

    @classmethod
    def _sample_trial_params(
        cls, optuna_trial: optuna.Trial, search_space, default_config: ConfigBase
    ) -> Dict[str, Any]:
        parameter: Dict[str, Any] = {}

        for k, v in search_space:
            val_type = default_config.get_type_with_dot_path(k)
            assert len(v) >= 2, f"Must provide an interval for {k}"
            # TODO conditional sampling
            if val_type == int and len(v) == 2:
                parameter[k] = optuna_trial.suggest_int(k, min(v), max(v))
            elif val_type == float and len(v) == 2:
                parameter[k] = optuna_trial.suggest_float(k, min(v), max(v))
            else:
                parameter[k] = optuna_trial.suggest_categorical(k, v)
        default_trial_kwargs = default_config.to_dict()
        trial_kwargs = cls.augment_trial_kwargs(
            trial_kwargs=default_trial_kwargs, augmentation=parameter
        )
        return trial_kwargs

    def sample_trials(self, n_trials_to_sample: int) -> List[ParallelConfig]:
        # Return pending trials when sampling first.
        assert n_trials_to_sample > 0
        n_trials_to_sample = min(self.n_trials_remaining, n_trials_to_sample)

        trials = self.__sample_trials(
            n_trials_to_sample,
            prev_trials=self.pending_trials,
            ignore_errors=self.config.ignore_errored_trials,
        )[:n_trials_to_sample]

        for trial in trials:
            self.update_trial_state(trial.uid, None, TrialState.RUNNING)
        return trials

    def __append_trial(
        self,
        trial_kwargs: Dict[str, Any],
        optuna_trial_num: int,
        trial_state: TrialState,
    ) -> bool:
        if trial_state == TrialState.PRUNED_INVALID:
            self._update_optuna_trial_state(
                optuna_trial_num, None, OptunaTrialState.PRUNED
            )
            return False
        else:
            trial_config = type(self.config)(**trial_kwargs)

            if trial_config.uid in self.all_trials_uid:
                trial_state = TrialState.PRUNED_DUPLICATE
                self.logger.warn(
                    f"Sampled a duplicate trial. Skipping: \n{self.tune_trial_str(trial_config)}"
                )
                return False
            self.__append_trial_internal(
                trial_config.uid, trial_kwargs, optuna_trial_num, trial_state
            )
            return True

    def __sample_trials(
        self,
        n_trials: int,
        prev_trials: Optional[List[ParallelConfig]] = None,
        ignore_errors=False,
    ) -> List[ParallelConfig]:

        error_upper_bound = n_trials * 10
        sampled_trials: List[ParallelConfig] = [] if prev_trials is None else prev_trials
        while len(sampled_trials) < n_trials:

            if (
                len(self.pruned_errored_trials) + len(self.duplicate_trials)
                > error_upper_bound
            ):
                raise RuntimeError(
                    f"Reached maximum limit of misconfigured trials. {error_upper_bound}\nFound {len(self.duplicate_trials)} duplicate and {len(self.pruned_errored_trials)}."
                )

            optuna_trial = self.optuna_study.ask()
            trial_kwargs = self._sample_trial_params(
                optuna_trial, self.search_space, self.config
            )
            trial_state = TrialState.WAITING

            try:
                trial_config = type(self.config)(**trial_kwargs)
                prev_uid = copy.deepcopy(trial_config.uid)
                model_dir = self.experiment_dir.joinpath(trial_config.uid)
                trial_config.model_dir = model_dir.as_posix()
                trial_config.train_config.resume = False
                # Updating the model_dir should not change the uid.
                assert prev_uid == trial_config.uid, "Misconfigured experiment."
                trial_kwargs["model_dir"] = model_dir.as_posix()
                trial_kwargs["train_config"]["resume"] = False

            except TypeError as e:
                if ignore_errors:
                    trial_state = TrialState.PRUNED_INVALID
                    self.logger.warn(str(e))
                else:
                    raise e

            if self.__append_trial(trial_kwargs, optuna_trial.number, trial_state):
                sampled_trials.append(trial_config)
        return sampled_trials

    def update_trial_state(
        self,
        config_uid: str,
        metrics: Optional[Metrics],
        state: TrialState,
    ):
        if state == TrialState.WAITING:
            return self._reset_trial_state(config_uid)
        elif state == TrialState.RECOVERABLE_ERROR:
            return self._inc_error_count(config_uid, state)


        trial_num = self._get_optuna_trial_num(config_uid)
        self._update_optuna_trial_state(trial_num, metrics, state)
        self._update_internal_trial_state(config_uid, metrics, state)

    def _internal_optim_values(self, metrics: Metrics) -> Dict[str, float]:
        if self.config.optim_directions is None:
            return {}
        vals = {}
        for k, v in self.config.optim_directions:
            val = getattr(metrics, k, None)
            if val is None:
                val = float("-inf") if Optim(v) == Optim.max else float("inf")
            vals[k] = val

        return vals

    def _get_optuna_trial_num(self, config_uid: str) -> int:
        with Session(self.engine) as session:
            stmt = select(Trial).where(Trial.config_uid == config_uid)
            res = session.scalar(stmt)
        return res.optuna_trial_num

    def _update_internal_trial_state(
        self, config_uid: str, metrics: Optional[Metrics], state: TrialState
    ):
        if metrics is not None:
            internal_metrics = self._internal_optim_values(metrics)
        else:
            internal_metrics = None
        with Session(self.engine) as session:
            stmt = select(Trial).where(Trial.config_uid == config_uid)
            res = session.execute(stmt).scalar_one()
            res.metrics.append(internal_metrics)
            res.state = state
            session.commit()
            session.flush()

        return True

    def _inc_error_count(self, config_uid: str, state: TrialState):
        with Session(self.engine) as session:
            stmt = select(Trial).where(Trial.config_uid == config_uid)
            res = session.execute(stmt).scalar_one()
            assert state == TrialState.RECOVERABLE_ERROR
            runtime_errors = copy.deepcopy(res.runtime_errors)
            res.runtime_errors = Trial.runtime_errors + 1

            session.commit()
            session.flush()

        if runtime_errors < 10:
            self.logger.warn(
                f"{config_uid} failed {runtime_errors} times."
            )
            self._reset_trial_state(config_uid)
        else:
            self.logger.error(
                f"{config_uid} failed {runtime_errors} times. Skipping."
            )
            self.update_trial_state(config_uid, None, TrialState.FAIL)

    def _reset_trial_state(self, config_uid: str):
        with Session(self.engine) as session:
            stmt = select(Trial).where(Trial.config_uid == config_uid)
            res = session.execute(stmt).scalar_one()
            assert res.state == TrialState.RUNNING
            res.state = TrialState.WAITING
            _config = copy.deepcopy(res.config_param)
            existing_checkpoints = len(
                list(
                    self.experiment_dir.joinpath(config_uid, "checkpoints").glob("*.pt")
                )
            )
            if existing_checkpoints > 0:
                _config["train_config"]["resume"] = True
            else:
                self.logger.warn(
                    f"{config_uid} was interupted but no checkpoint was found. Will re-start training."
                )
                _config["train_config"]["resume"] = False

            res.config_param = _config
            session.commit()
            session.flush()

        return True

    def _optuna_optim_values(self, metrics: Metrics) -> List[float]:
        if self.config.optim_directions is None:
            return []
        vals = []

        for k, v in self.config.optim_directions:
            val = getattr(metrics, k, None)
            if val is None:
                val = float("-inf") if Optim(v) == Optim.max else float("inf")
            vals.append(val)

        return vals

    def _update_optuna_trial_state(
        self,
        trial_num: int,
        metrics: Optional[Metrics],
        state: TrialState,
    ):

        optuna_state = state.to_optuna_state()
        if metrics is not None:
            optuna_metrics = self._optuna_optim_values(metrics)
        else:
            optuna_metrics = None

        if optuna_state == OptunaTrialState.RUNNING:
            return
        # TODO raises error for nan values in metrics. Fixme
        self.optuna_study.tell(
            trial_num,
            optuna_metrics,
            optuna_state,
        )

    @property
    def all_trials_uid(self) -> List[str]:
        return [c.uid for c in self.all_trials]

    @property
    def running_trials_uid(self) -> List[str]:
        return [c.uid for c in self.running_trials]

    @property
    def duplicate_trials_uid(self) -> List[str]:
        return [c.uid for c in self.duplicate_trials]

    @property
    def pending_trials_uid(self) -> List[str]:
        return [c.uid for c in self.pending_trials]

    @property
    def complete_trials_uid(self) -> List[str]:
        return [c.uid for c in self.complete_trials]

    def __append_trial_internal(
        self,
        config_uid: str,
        trial_kwargs: Dict[str, Any],
        optuna_trial_num: int,
        trial_state: TrialState,
    ):

        with Session(self.engine) as session:
            trial = Trial(
                config_uid=config_uid,
                config_param=trial_kwargs,
                optuna_trial_num=optuna_trial_num,
                state=trial_state,
                metrics=[],
            )
            session.add(trial)
            session.commit()

    def _get_trials_by_stmt(self, stmt) -> List[Trial]:
        with self.engine.connect() as conn:
            trials: List[Trial] = conn.execute(stmt).fetchall()
        return trials

    def _get_trial_configs_by_stmt(self, stmt) -> List[ParallelConfig]:
        trials = self._get_trials_by_stmt(stmt)
        configs = []
        for trial in trials:
            trial_config = type(self.config)(**trial.config_param)
            configs.append(trial_config)
            assert trial_config.uid == trial.config_uid
        return configs

    @property
    def all_trials(self) -> List[ParallelConfig]:
        stmt = select(Trial).where(
            (Trial.state != TrialState.WAITING)
            & (Trial.state != TrialState.PRUNED_DUPLICATE)
            & (Trial.state != TrialState.PRUNED_INVALID)
        )
        return self._get_trial_configs_by_stmt(stmt)

    @property
    def pruned_errored_trials(self) -> List[Dict[str, Any]]:
        """
        error trials can not be initialized to a configuration
        and such as return the kwargs parameters.
        """

        stmt = select(Trial).where((Trial.state == TrialState.PRUNED_INVALID))
        trials = self._get_trials_by_stmt(stmt)
        return [trial.config_param for trial in trials]

    @property
    def running_trials(self) -> List[ParallelConfig]:

        stmt = select(Trial).where((Trial.state == TrialState.RUNNING))
        return self._get_trial_configs_by_stmt(stmt)

    @property
    def duplicate_trials(self) -> List[ParallelConfig]:
        stmt = select(Trial).where((Trial.state == TrialState.PRUNED_DUPLICATE))
        return self._get_trial_configs_by_stmt(stmt)

    @property
    def pending_trials(self) -> List[ParallelConfig]:
        stmt = select(Trial).where((Trial.state == TrialState.WAITING))
        return self._get_trial_configs_by_stmt(stmt)

    @property
    def complete_trials(self) -> List[ParallelConfig]:
        stmt = select(Trial).where((Trial.state == TrialState.COMPLETE))
        return self._get_trial_configs_by_stmt(stmt)

    @property
    def failed_trials(self) -> List[ParallelConfig]:
        stmt = select(Trial).where((Trial.state == TrialState.FAIL))
        return self._get_trial_configs_by_stmt(stmt)

    @property
    def n_trials_remaining(self) -> int:
        """
        We get all trials as it can include, trials at different
        states. We exclude the unscheduled trials (pending), and
        the ones that are pruned during sampling.
        """

        return self.config.total_trials - len(self.all_trials)

import copy
import enum
import typing as ty
from collections import OrderedDict
from pathlib import Path

import numpy as np
import optuna
from optuna.trial import TrialState as OptunaTrialState
from sqlalchemy import Integer, PickleType, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

import ablator.utils.base as butils
from ablator.main.configs import (
    Optim,
    ParallelConfig,
    SearchAlgo,
    SearchSpace,
    SearchType,
)
from ablator.modules.loggers.file import FileLogger
from ablator.utils.file import nested_set


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

    def to_optuna_state(self) -> OptunaTrialState | None:
        if self in [
            TrialState.PRUNED,
            TrialState.PRUNED_INVALID,
            TrialState.PRUNED_DUPLICATE,
            TrialState.PRUNED_POOR_PERFORMANCE,
        ]:
            return OptunaTrialState.PRUNED
        if self in {TrialState.RUNNING}:
            return None

        return OptunaTrialState(self)


def augment_trial_kwargs(
    trial_kwargs: dict[str, ty.Any], augmentation: dict[str, ty.Any]
) -> dict[str, ty.Any]:
    trial_kwargs = copy.deepcopy(trial_kwargs)
    config_dot_path: str
    dot_paths = list(augmentation.keys())

    assert len(set(dot_paths)) == len(
        dot_paths
    ), f"Duplicate tune paths: {set(dot_paths).difference(dot_paths)}"
    for config_dot_path, val in augmentation.items():
        path: list[str] = config_dot_path.split(".")
        trial_kwargs = nested_set(trial_kwargs, path, val)
    return trial_kwargs


def parse_metrics(
    metric_directions: OrderedDict, metrics: dict[str, float]
) -> dict[str, float]:
    vals = OrderedDict()
    metric_keys = set(metric_directions)
    user_metrics = set(metrics)
    assert (
        user_metrics == metric_keys
    ), f"Different specified metric directions `{metric_keys}` and `{user_metrics}`"
    for k, v in metric_directions.items():
        val = metrics[k]
        if val is None or not np.isfinite(val):
            val = float("-inf") if Optim(v) == Optim.max else float("inf")
        vals[k] = val
    return vals


def sample_trial_params(
    optuna_trial: optuna.Trial,
    search_space: dict[str, SearchSpace],
) -> dict[str, ty.Any]:
    parameter: dict[str, ty.Any] = {}

    for k, v in search_space.items():
        # TODO conditional sampling
        if v.value_range is not None and v.value_type == SearchType.integer:
            low, high = v.value_range
            low = int(low)
            high = int(high)
            assert (
                min(low, high) == low
            ), "`value_range` must be in the format of (min,max)"
            parameter[k] = optuna_trial.suggest_int(k, low, high)
        elif v.value_range is not None and v.value_type == SearchType.numerical:
            low, high = v.value_range
            low = float(low)
            high = float(high)
            assert (
                min(low, high) == low
            ), "`value_range` must be in the format of (min,max)"
            parameter[k] = optuna_trial.suggest_float(k, low, high)
        elif v.categorical_values is not None:
            parameter[k] = optuna_trial.suggest_categorical(k, v.categorical_values)
        else:
            raise ValueError(f"Invalid SearchSpace {v}.")

    return parameter


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


class OptunaState:
    def __init__(
        self,
        storage: str,
        study_name,
        optim_metrics: dict[str, Optim],
        search_algo,
        search_space: dict[str, SearchSpace],
    ) -> None:
        sampler: optuna.samplers.BaseSampler

        if search_algo == SearchAlgo.random:
            sampler = optuna.samplers.RandomSampler()
        elif search_algo == SearchAlgo.tpe:
            sampler = optuna.samplers.TPESampler()
        else:
            raise NotImplementedError
        if optim_metrics is None:
            raise ValueError("Must specify optim_metrics.")

        self.optim_metrics = OrderedDict(optim_metrics)
        self.search_space = search_space
        directions = [
            optuna.study.StudyDirection(
                2 if Optim(optim_metrics[k]) == Optim.max else 1
            )
            for k in optim_metrics
        ]

        self.optuna_study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            directions=directions,
            load_if_exists=True,
            sampler=sampler,
        )

    def _optuna_optim_values(self, metrics: dict[str, float]) -> list[float]:
        return list(parse_metrics(self.optim_metrics, metrics).values())

    def update_trial(
        self,
        trial_num: int,
        metrics: dict[str, float] | None,
        state: TrialState,
    ):
        if metrics is None and state == TrialState.COMPLETE:
            raise RuntimeError(f"Missing metrics for complete trial {trial_num}.")
        if metrics is None or state != TrialState.COMPLETE:
            return
        optuna_state = state.to_optuna_state()
        optuna_metrics = self._optuna_optim_values(metrics)

        # TODO raises error for nan values in metrics. Fixme
        self.optuna_study.tell(trial_num, optuna_metrics, optuna_state)

    def sample_trial(self):
        optuna_trial = self.optuna_study.ask()
        return (
            optuna_trial.number,
            sample_trial_params(optuna_trial, self.search_space),
        )


class ExperimentState:
    def __init__(
        self,
        experiment_dir: Path,
        config: ParallelConfig,
        logger: FileLogger | None = None,
        resume: bool = False,
    ) -> None:
        self.optuna_trial_map: dict[str, optuna.Trial] = {}
        self.config = config
        self.logger: FileLogger = logger if logger is not None else butils.Dummy()
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        default_vals = self.config.make_dict(self.config.annotations, flatten=True)
        assert len(self.config.search_space), "Must specify a config.search_space."
        for k in self.config.search_space:
            if k not in default_vals:
                raise RuntimeError(
                    f"SearchSpace parameter {k} was not found in the configuration {sorted(list(default_vals.keys()))}."
                )
        study_name = config.uid
        self.experiment_dir = experiment_dir

        optuna_db_path = experiment_dir.joinpath(f"{study_name}_optuna.db")
        if optuna_db_path.exists() and not resume:
            raise RuntimeError(
                f"{optuna_db_path} exists. Please remove before starting a study."
            )

        self.optuna_state = OptunaState(
            f"sqlite:///{optuna_db_path}",
            study_name=config.uid,
            optim_metrics=config.optim_metrics,
            search_algo=config.search_algo,
            search_space=config.search_space,
        )
        experiment_state_db = experiment_dir.joinpath(f"{study_name}_state.db")

        self.engine = create_engine(f"sqlite:///{experiment_state_db}", echo=False)
        Trial.metadata.create_all(self.engine)

        self._init_trials(resume=resume)

    @staticmethod
    def search_space_dot_path(trial: ParallelConfig) -> dict[str, ty.Any]:
        return {
            dot_path: trial.get_val_with_dot_path(dot_path)
            for dot_path in trial.search_space.keys()
        }

    @staticmethod
    def tune_trial_str(trial: ParallelConfig) -> str:
        trial_map = ExperimentState.search_space_dot_path(trial)
        msg = f"\n{trial.uid}:\n\t"
        msg = "\n\t".join(
            [f"{dot_path} -> {val} " for dot_path, val in trial_map.items()]
        )
        return msg

    def _init_trials(self, resume: bool = False) -> list[ParallelConfig]:
        max_trials_conc = min(self.config.concurrent_trials, self.config.total_trials)

        if self.config.search_algo in [
            SearchAlgo.random,
        ]:
            trials_to_sample = self.n_trials_remaining
        else:
            trials_to_sample = max_trials_conc

        # if there are currently running trials, return those first and is resume. Otherwise
        # ERROR

        if len(self.running_trials) > 0 and not resume:
            raise RuntimeError(
                "Experiment exists. You need to use `resume = True` or use a different path."
            )
        running_trials = []
        for trial in self.running_trials:
            self.update_trial_state(trial.uid, None, TrialState.WAITING)
            running_trials.append(trial)

        trials = self.__sample_trials(
            trials_to_sample,
            running_trials + self.pending_trials,
            ignore_errors=self.config.ignore_invalid_params,
        )[:max_trials_conc]

        for trial in trials:
            self.update_trial_state(trial.uid, None, TrialState.RUNNING)
        assert len(self.running_trials) > 0, "No trials could be scheduled."
        return trials

    def sample_trials(self, n_trials_to_sample: int) -> list[ParallelConfig] | None:
        # Return pending trials when sampling first.
        assert n_trials_to_sample > 0
        n_trials_to_sample = min(self.n_trials_remaining, n_trials_to_sample)
        if self.n_trials_remaining == 0:
            self.logger.warn(
                f"Limit of trials to sample '{self.config.total_trials}' reached."
            )
            return None
        trials = self.__sample_trials(
            n_trials_to_sample,
            prev_trials=self.pending_trials,
            ignore_errors=self.config.ignore_invalid_params,
        )[:n_trials_to_sample]

        for trial in trials:
            self.update_trial_state(trial.uid, None, TrialState.RUNNING)
        return trials

    def __append_trial(
        self,
        trial_kwargs: dict[str, ty.Any],
        optuna_trial_num: int,
        trial_state: TrialState,
    ) -> bool:
        if trial_state in {TrialState.PRUNED_INVALID, TrialState.PRUNED_DUPLICATE}:
            # self.optuna_state.update_trial(optuna_trial_num, None, trial_state)
            self.__append_trial_internal(
                "none", trial_kwargs, optuna_trial_num, trial_state
            )
            return False
        trial_config = type(self.config)(**trial_kwargs)
        self.__append_trial_internal(
            trial_config.uid, trial_kwargs, optuna_trial_num, trial_state
        )
        return True

    def __sample_trials(
        self,
        n_trials: int,
        prev_trials: list[ParallelConfig] | None = None,
        ignore_errors=False,
    ) -> list[ParallelConfig]:
        error_upper_bound = n_trials * 10
        sampled_trials: list[ParallelConfig] = (
            [] if prev_trials is None else prev_trials
        )
        while len(sampled_trials) < n_trials:
            if (
                len(self.pruned_errored_trials) + len(self.pruned_duplicate_trials)
                > error_upper_bound
            ):
                raise RuntimeError(
                    f"Reached maximum limit of misconfigured trials. {error_upper_bound}\n"
                    f"Found {len(self.pruned_duplicate_trials)} duplicate and "
                    f"{len(self.pruned_errored_trials)} invalid trials."
                )

            trial_num, parameter = self.optuna_state.sample_trial()
            trial_kwargs = augment_trial_kwargs(
                trial_kwargs=self.config.to_dict(), augmentation=parameter
            )

            trial_state = TrialState.WAITING

            try:
                trial_config = type(self.config)(**trial_kwargs)
                if trial_config.uid in self.all_trials_uid:
                    trial_state = TrialState.PRUNED_DUPLICATE
            except Exception as e:
                if ignore_errors:
                    trial_state = TrialState.PRUNED_INVALID
                    self.logger.warn(f"ignoring: {parameter}. Error:{e}")
                else:
                    raise TypeError(f"Invalid trial parameters {parameter}") from e

            if self.__append_trial(trial_kwargs, trial_num, trial_state):
                sampled_trials.append(trial_config)
        return sampled_trials

    def update_trial_state(
        self,
        config_uid: str,
        metrics: dict[str, float] | None = None,
        state: TrialState = TrialState.RUNNING,
    ) -> None:
        if state == TrialState.RECOVERABLE_ERROR:
            self._inc_error_count(config_uid, state)
            return

        self._update_internal_trial_state(config_uid, metrics, state)
        # NOTE currently it is error prone to update the optuna state
        trial_num = self._get_optuna_trial_num(config_uid)
        self.optuna_state.update_trial(trial_num, metrics, state)

    def _get_optuna_trial_num(self, config_uid: str) -> int:
        with Session(self.engine) as session:
            stmt = select(Trial).where(Trial.config_uid == config_uid)
            res = session.scalar(stmt)
        return res.optuna_trial_num

    def _update_internal_trial_state(
        self, config_uid: str, metrics: dict[str, float] | None, state: TrialState
    ):
        if metrics is not None:
            internal_metrics = parse_metrics(self.config.optim_metrics, metrics)
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
            self.logger.warn(f"{config_uid} failed {runtime_errors} times.")
            self.update_trial_state(config_uid, None, TrialState.WAITING)
        else:
            self.logger.error(f"{config_uid} failed {runtime_errors} times. Skipping.")
            self.update_trial_state(config_uid, None, TrialState.FAIL)

    # def _reset_trial_state(self, config_uid: str):
    #     with Session(self.engine) as session:
    #         stmt = select(Trial).where(Trial.config_uid == config_uid)
    #         res = session.execute(stmt).scalar_one()
    #         assert res.state == TrialState.RUNNING
    #         res.state = TrialState.WAITING
    #         _config = copy.deepcopy(res.config_param)
    #         existing_checkpoints = len(
    #             list(
    #                 self.experiment_dir.joinpath(config_uid, "checkpoints").glob("*.pt")
    #             )
    #         )
    #         if existing_checkpoints > 0:
    #             _config["train_config"]["resume"] = True
    #         else:
    #             self.logger.warn(
    #                 f"{config_uid} was interupted but no checkpoint was found. Will re-start training."
    #             )
    #             _config["train_config"]["resume"] = False

    #         res.config_param = _config
    #         session.commit()
    #         session.flush()

    #     return True

    def __append_trial_internal(
        self,
        config_uid: str,
        trial_kwargs: dict[str, ty.Any],
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

    def _get_trials_by_stmt(self, stmt) -> list[Trial]:
        with self.engine.connect() as conn:
            trials: list[Trial] = conn.execute(stmt).fetchall()
        return trials

    def _get_trial_configs_by_stmt(self, stmt) -> list[ParallelConfig]:
        trials = self._get_trials_by_stmt(stmt)
        configs = []
        for trial in trials:
            trial_config = type(self.config)(**trial.config_param)
            configs.append(trial_config)
            assert trial_config.uid == trial.config_uid
        return configs

    @property
    def all_trials_uid(self) -> list[str]:
        return [c.uid for c in self.all_trials]

    @property
    def all_trials(self) -> list[ParallelConfig]:
        stmt = select(Trial).where(
            # (Trial.state != TrialState.WAITING)
            (Trial.state != TrialState.PRUNED_DUPLICATE)
            & (Trial.state != TrialState.PRUNED_INVALID)
        )
        return self._get_trial_configs_by_stmt(stmt)

    @property
    def pruned_errored_trials(self) -> list[dict[str, ty.Any]]:
        """
        error trials can not be initialized to a configuration
        and such as return the kwargs parameters.
        """

        stmt = select(Trial).where((Trial.state == TrialState.PRUNED_INVALID))
        trials = self._get_trials_by_stmt(stmt)
        return [dict(trial.config_param) for trial in trials]

    @property
    def pruned_duplicate_trials(self) -> list[dict[str, ty.Any]]:
        stmt = select(Trial).where((Trial.state == TrialState.PRUNED_DUPLICATE))
        trials = self._get_trials_by_stmt(stmt)
        return [dict(trial.config_param) for trial in trials]

    @property
    def running_trials(self) -> list[ParallelConfig]:
        stmt = select(Trial).where((Trial.state == TrialState.RUNNING))
        return self._get_trial_configs_by_stmt(stmt)

    @property
    def pending_trials(self) -> list[ParallelConfig]:
        stmt = select(Trial).where((Trial.state == TrialState.WAITING))
        return self._get_trial_configs_by_stmt(stmt)

    @property
    def complete_trials(self) -> list[ParallelConfig]:
        stmt = select(Trial).where((Trial.state == TrialState.COMPLETE))
        return self._get_trial_configs_by_stmt(stmt)

    @property
    def failed_trials(self) -> list[ParallelConfig]:
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

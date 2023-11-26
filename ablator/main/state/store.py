import enum

from sqlalchemy import Integer, PickleType, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TrialState(enum.IntEnum):
    """
    An enumeration of possible states for a trial with more pruned states.

    Attributes
    ----------
        RUNNING : int
            A trial that has been succesfully scheduled to run
        COMPLETE : int
            Succesfully completed trial
        PRUNED : int
            Trial pruned because of various reasons
        FAIL : int
            Trial that produced an unrecoverable error during execution
        WAITING : int
            Trial that is waiting to be scheduled to run
        PRUNED_INVALID : int
            Trial that was pruned during sampling as it was invalid
        PRUNED_DUPLICATE : int
            Trial that was sampled but was already present
        PRUNED_POOR_PERFORMANCE : int
            Trial that was pruned during execution for poor performance
        FAIL_RECOVERABLE : int
            Trial that was pruned during execution for poor performance

    """

    RUNNING: int = 0
    COMPLETE: int = 1
    PRUNED: int = 2
    FAIL: int = 3
    WAITING: int = 4
    PRUNED_INVALID: int = 5
    PRUNED_DUPLICATE: int = 6
    PRUNED_POOR_PERFORMANCE: int = 7
    FAIL_RECOVERABLE: int = 8


class Trial(Base):
    """
    Class to store adata about trial.

    Attributes
    ----------
    id: Mapped[int]
        The trial Id used for internal purposes
    config_uid: Mapped[str]
        The configuration identifier associated with the trial's unique attributes
    metrics: Mapped[PickleType]
        The performance metrics dictionary associated as reported by the trial.
        Dict[str,float] where str is the metric name and float is the metric value.
    config_param: Mapped[PickleType]
        The configuration parameters for the specific trial including the defaults.
    aug_config_param: Mapped[PickleType]
        The augmenting configuration as picked by the config sampler.
        It is the values only different from the default config (excl. Derived properties)
    trial_uid: Mapped[Integer]
        The trial_uid corresponding to the internal HPO sampler, used to communicate with the sampler.
    state: Mapped[PickleType]
        The ``TrialState``
    runtime_errors: Mapped[int]
        Total runtime errors that the trial encountered and are incremented
        every time the trial faces a recoverable error.
    """

    __tablename__ = "trial"
    id: Mapped[int] = mapped_column(primary_key=True)
    config_uid: Mapped[str] = mapped_column(String(30))
    metrics: Mapped[PickleType] = mapped_column(PickleType)
    config_param: Mapped[PickleType] = mapped_column(PickleType)
    aug_config_param: Mapped[PickleType] = mapped_column(PickleType)
    trial_uid: Mapped[Integer] = mapped_column(Integer)
    state: Mapped[PickleType] = mapped_column(PickleType, default=TrialState.WAITING)
    runtime_errors: Mapped[int] = mapped_column(Integer, default=0)
    # NOTE the following attributes are subject to be removed when optuna is decoupled.
    # they are for internal use ONLY
    _opt_distributions_kwargs: Mapped[PickleType] = mapped_column(
        PickleType, nullable=True
    )
    _opt_distributions_types: Mapped[PickleType] = mapped_column(
        PickleType, nullable=True
    )
    _opt_params: Mapped[PickleType] = mapped_column(PickleType, nullable=True)

    def __repr__(self) -> str:
        return (
            f"Trial(id={self.id!r}, config_uid={self.config_uid!r},"
            f" fullname={self.aug_config_param!r})"
        )

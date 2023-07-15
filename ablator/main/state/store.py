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

    RUNNING = 0
    COMPLETE = 1
    PRUNED = 2
    FAIL = 3
    WAITING = 4
    PRUNED_INVALID = 5
    PRUNED_DUPLICATE = 6
    PRUNED_POOR_PERFORMANCE = 7
    FAIL_RECOVERABLE = 8


class Trial(Base):
    __tablename__ = "trial"
    id: Mapped[int] = mapped_column(primary_key=True)
    config_uid: Mapped[str] = mapped_column(String(30))
    metrics: Mapped[PickleType] = mapped_column(PickleType)
    config_param: Mapped[PickleType] = mapped_column(PickleType)
    aug_config_param: Mapped[PickleType] = mapped_column(PickleType)
    trial_num: Mapped[Integer] = mapped_column(Integer)
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
        return f"Trial(id={self.id!r}, config_uid={self.config_uid!r}, fullname={self.aug_config_param!r})"

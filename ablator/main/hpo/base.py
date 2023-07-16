from abc import ABC, abstractmethod
import typing as ty

from ablator.main.state.store import TrialState


class BaseSampler(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._lock = False

    @abstractmethod
    def _eager_sample(self):
        """
        Please see documentation of `eager_sampler`
        """
        raise NotImplementedError

    @abstractmethod
    def update_trial(
        self, trial_id: int, metrics: dict[str, float] | None, state: TrialState
    ):
        """
        Update the trial state given the trial_id, the updated metrics, and the current trial state.

        Parameters
        ----------
        trial_id : int
            the trial_id which was returned when running ``eager_sampler``
        metrics : dict[str, float]
            a metric dictionary corresponding to the updated metrics.
        state : TrialState
            the updated trial state
        """
        raise NotImplementedError

    @abstractmethod
    def internal_repr(self, trial_id: int) -> None | dict[str, ty.Any]:
        """
        Return the internal representation of the trial if one is maintained by
        the sampler.

        Parameters
        ----------
        trial_id : int
            the trial_id which internal representation is retrieved.

        Returns
        -------
        None | dict[str, ty.Any]
            ``None`` when there is no internal representation maintained by the sampler.
            Otherwise a dictionary with keys as the internal configuration names and values, the
            corresponding values.

        """
        raise NotImplementedError

    @abstractmethod
    def _drop(self):
        """
        Should delete an eagerly sampled trial. Please see documentation of `unlock` for implimenetation details.
        """
        raise NotImplementedError

    def eager_sample(self) -> tuple[int, dict[str, ty.Any], None | dict[str, ty.Any]]:
        """
        eager_sample A sampled trial can be erroneous, for this reason we eagerly sample
        and lock the sampler until the user can verify the sampled configuration.

        Returns
        -------
        tuple[int | dict[str, ty.Any]]
            a tuple that contains the trial id and the sampled configuration from the search space.

        Raises
        ------
        StopIteration
            Can raise an error if there are no more trials to sample.
        """
        assert (
            not self._lock
        ), "Must call `unlock(drop=[True,False])` after `eager_sampler`."
        self._lock = True
        trial_id, config = self._eager_sample()
        kwargs = self.internal_repr(trial_id)
        return trial_id, config, kwargs

    def unlock(self, drop: bool):
        """
        unlock informs the sampler on whether the eagerely sampled trial was valid or should be dropped.

        Parameters
        ----------
        drop : bool
            whether to drop the trial
        """
        assert self._lock or not drop
        self._lock = False
        if drop:
            self._drop()

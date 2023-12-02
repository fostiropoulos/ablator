from abc import ABC, abstractmethod
import logging
import multiprocessing
import threading

import traceback


class Heart(ABC):
    """
    Base class for managing the execution of a background update process
    as a thread. Includes a fault-tolerant strategy, where we execute
    `heartbeat` function on an interval and in case of error, we repeat the process
    of up to `missed_heart_beats`.

    Parameters
    ----------
    missed_heart_beats : int, optional
        the number of missed updates after which it will raise an error, by default 3
    heartbeat_interval : int, optional
        the interval in seconds by which to perform a heart-beat, by default 10

    Attributes
    ----------
    missed_heart_beats : int
        the number of heart-beats after which the process will be considered
        dead
    heartbeat_interval : int
        the interval in seconds by which to perform a heart-beat
    """

    def __init__(self, missed_heart_beats: int = 3, heartbeat_interval: int = 10):
        # the first heart-beat is for error-diagnosing
        self.heartbeat()
        self.missed_heart_beats = missed_heart_beats
        self.heartbeat_interval = heartbeat_interval
        self._end_of_life = multiprocessing.Event()
        self._heart = threading.Thread(target=self._heartbeat)
        self._heart.daemon = True
        self._heart.start()

    # pylint: disable=broad-exception-caught
    def _heartbeat(self):
        missed_heartbeats = 0
        while True:
            try:
                self.heartbeat()
                missed_heartbeats = 0
            except Exception:
                exc = traceback.format_exc()
                logging.error("Error during heartbeat: %s", exc)
                missed_heartbeats += 1
            if missed_heartbeats > self.missed_heart_beats:
                raise RuntimeError("Missed too many heartbeats.")
            if self._end_of_life.wait(self.heartbeat_interval):
                return

    @abstractmethod
    def heartbeat(self, timeout: int | None = None):
        ...

    def __enter__(self):
        return self

    def stop(self):
        self._end_of_life.set()

    def __exit__(self, *args, **kwargs):
        self.stop()

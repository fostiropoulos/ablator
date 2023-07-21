import os
from datetime import datetime
from pathlib import Path

import ray


class FileLogger:
    """
    A logger that writes messages to a file and prints them to the console.

    Attributes
    ----------
    WARNING : str
        ANSI escape code for the warning text color.
    FAIL : str
        ANSI escape code for the error text color.
    ENDC : str
        ANSI escape code for resetting the text color.
    """

    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"

    def __init__(
        self,
        path: str | Path | None = None,
        verbose: bool = True,
        prefix: str | None = None,
    ):
        """
        Initialize a FileLogger.

        Parameters
        ----------
        path : str | Path | None, optional
            Path to the log file, by default ``None``.
        verbose : bool, optional
            Whether to print messages to the console, by default ``True``.
        prefix : str | None, optional
            A prefix to add to each logged message, by default ``None``.
        """
        self.path = path
        if path is not None:
            self.set_path(path)
        self.verbose = verbose
        self.set_prefix(prefix)

    def _write(self, msg: str):
        """
        Write a message to the log file.

        Parameters
        ----------
        msg : str
            The message to write.
        """

        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"{msg}\n")

    def _print(self, msg: str, verbose=False):
        """Print a message to the console.

        Parameters
        ----------
        msg : str
            The message to print.
        verbose : bool, optional
            Whether to print messages to the console, by default True.
        """

        if self.verbose or verbose:
            print(msg)

    def info(self, msg: str, verbose=False) -> str:
        """Log an info message.

        Parameters
        ----------
        msg : str
            The message to log.
        verbose : bool, optional
            Whether to print messages to the console, by default ``False``.

        Returns
        -------
        str
            the formated string message
        """
        return self(msg, verbose)

    def warn(self, msg: str, verbose=True) -> str:
        """Log a warning message.

        Parameters
        ----------
        msg : str
            The message to log.
        verbose : bool, optional
            Whether to print messages to the console, by default ``True``.

        Returns
        -------
        str
            the formated string message
        """
        msg = f"{FileLogger.WARNING}{msg}{FileLogger.ENDC}"
        return self(msg, verbose)

    def error(self, msg: str) -> str:
        """Log an error message.

        Parameters
        ----------
        msg : str
            The message to log.

        Returns
        -------
        str
            the formated string message
        """
        msg = f"{FileLogger.FAIL}{msg}{FileLogger.ENDC}"
        return self(msg, True)

    def __call__(self, msg: str, verbose=True) -> str:
        """Log a message.

        Parameters
        ----------
        msg : str
            The message to log.
        verbose : bool, optional
            Whether to print messages to the console, by default True.


        Returns
        -------
        str
            the formated string message
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"{now}: {self.prefix}{msg}"
        self._write(msg)
        self._print(msg, verbose)
        return msg

    def set_prefix(self, prefix: str | None = None):
        """Set the prefix for the logger.

        Parameters
        ----------
        prefix : str | None, optional
            The prefix to add to each logged message, by default ``None``.
        """
        if prefix is not None:
            self.prefix = f"{prefix} - "
        else:
            self.prefix = ""

    def set_path(self, path: str | Path):
        """Set the path to the log file.

        Parameters
        ----------
        path : str | Path
            The path to the log file.
        """
        self.path = Path(path)
        parent_dir = self.path.parent
        parent_dir.mkdir(exist_ok=True, parents=True)
        mode = "a" if os.path.exists(path) else "w"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(path, mode, encoding="utf-8") as f:
            f.write(f"Starting Logger {now} \n")


class RemoteFileLogger(FileLogger):
    def __init__(
        self,
        path: str | Path | None = None,
        verbose: bool = True,
        prefix: str | None = None,
    ):
        super().__init__(path, verbose, prefix)
        self._file_logger: ray.RemoteFunction[FileLogger]

    def to_remote(self, address: str | None = None):
        if address is None:
            address, _ = ray.get_runtime_context().gcs_address.split(":")
        self._file_logger = (
            ray.remote(FileLogger)
            .options(resources={f"node:{address}": 0.001})  # type: ignore
            .remote(
                self.path,
                self.verbose,
                self.prefix[: -len(" - ")] if self.prefix is not None else None,
            )
        )

    def __call__(self, msg: str, verbose=True):
        if hasattr(self, "_file_logger"):
            msg = ray.get(self._file_logger.__call__.remote(msg, False))
            self._print(msg, verbose)
        else:
            super().__call__(msg, verbose=verbose)

    def set_path(self, path: str | Path):
        if hasattr(self, "_file_logger"):
            ray.get(self._file_logger.set_path.remote(path))
        else:
            super().set_path(path)

    def set_prefix(self, prefix: str | None = None):
        if hasattr(self, "_file_logger"):
            ray.get(self._file_logger.set_prefix.remote(prefix))
        else:
            super().set_prefix(prefix)

from datetime import datetime
import os

from pathlib import Path


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
        """ Print a message to the console.

        Parameters
        ----------
        msg : str
            The message to print.
        verbose : bool, optional
            Whether to print messages to the console, by default True.
        """

        if self.verbose or verbose:
            print(msg)

    def info(self, msg: str, verbose=False):
        """Log an info message.

        Parameters
        ----------
        msg : str
            The message to log.
        verbose : bool, optional
            Whether to print messages to the console, by default ``False``.
        """
        self(msg, verbose)

    def warn(self, msg: str, verbose=True):
        """Log a warning message.

        Parameters
        ----------
        msg : str
            The message to log.
        verbose : bool, optional
            Whether to print messages to the console, by default ``True``.
        """
        msg = f"{FileLogger.WARNING}{msg}{FileLogger.ENDC}"
        self(msg, verbose)

    def error(self, msg: str):
        """Log an error message.

        Parameters
        ----------
        msg : str
            The message to log.
        """
        msg = f"{FileLogger.FAIL}{msg}{FileLogger.ENDC}"
        self(msg, True)

    def __call__(self, msg: str, verbose=True):
        """Log a message.

        Parameters
        ----------
        msg : str
            The message to log.
        verbose : bool, optional
            Whether to print messages to the console, by default True.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"{now}: {self.prefix}{msg}"
        self._write(msg)
        self._print(msg, verbose)

    def set_prefix(self, prefix: str | None = None):
        """ Set the prefix for the logger.

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
        """ Set the path to the log file.

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

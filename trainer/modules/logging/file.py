from datetime import datetime
import os

from pathlib import Path
import typing as ty


class FileLogger:
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"

    def __init__(
        self, path: ty.Optional[str] = None, verbose: bool = True, prefix: str = ""
    ):

        self.path = path
        if path is not None:
            self.set_path(path)
        self.verbose = verbose
        self.prefix = prefix

    def _write(self, msg: str):
        if self.path is not None:
            with open(self.path, "a") as f:
                f.write(f"{msg}\n")

    def _print(self, msg: str, force_print=False):
        if self.verbose or force_print:
            print(msg)

    def info(self, msg: str, force_print=False, to_console=True):
        self(msg, force_print=force_print, to_console=to_console)

    def warn(self, msg: str, force_print=True):
        msg = f"{FileLogger.WARNING}{msg}{FileLogger.ENDC}"
        self(msg, force_print)

    def error(self, msg: str, force_print=True):
        msg = f"{FileLogger.FAIL}{msg}{FileLogger.ENDC}"
        self(msg, force_print)

    def __call__(self, msg: str, force_print=True, to_console=True):
        now = datetime.now()
        msg = f"{now}: {self.prefix}{msg}"
        self._write(msg)
        if to_console:
            self._print(msg, force_print)

    def set_prefix(self, prefix):
        self.prefix = prefix

    def set_path(self, path):
        self.path = path
        parent_dir = Path(path).parent
        parent_dir.mkdir(exist_ok=True, parents=True)
        mode = "a" if os.path.exists(path) else "w"
        now = datetime.now()

        with open(path, mode) as f:
            f.write(f"Starting Logger {now} \n")

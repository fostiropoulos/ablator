from datetime import datetime
import os

from pathlib import Path


class FileLogger:
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"

    def __init__(
        self,
        path: str | Path | None = None,
        verbose: bool = True,
        prefix: str | None = None,
    ):
        self.path = path
        if path is not None:
            self.set_path(path)
        self.verbose = verbose
        self.set_prefix(prefix)

    def _write(self, msg: str):
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"{msg}\n")

    def _print(self, msg: str, verbose=False):
        if self.verbose or verbose:
            print(msg)

    def info(self, msg: str, verbose=False):
        self(msg, verbose)

    def warn(self, msg: str, verbose=True):
        msg = f"{FileLogger.WARNING}{msg}{FileLogger.ENDC}"
        self(msg, verbose)

    def error(self, msg: str):
        msg = f"{FileLogger.FAIL}{msg}{FileLogger.ENDC}"
        self(msg, True)

    def __call__(self, msg: str, verbose=True):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"{now}: {self.prefix}{msg}"
        self._write(msg)
        self._print(msg, verbose)

    def set_prefix(self, prefix: str | None = None):
        if prefix is not None:
            self.prefix = f"{prefix} - "
        else:
            self.prefix = ""

    def set_path(self, path: str | Path):
        self.path = Path(path)
        parent_dir = self.path.parent
        parent_dir.mkdir(exist_ok=True, parents=True)
        mode = "a" if os.path.exists(path) else "w"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(path, mode, encoding="utf-8") as f:
            f.write(f"Starting Logger {now} \n")

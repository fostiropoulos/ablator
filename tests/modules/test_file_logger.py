from pathlib import Path
from ablator.modules.loggers.file import FileLogger
from ablator.modules.loggers.main import SummaryLogger
from ablator.modules.loggers.tensor import TensorboardLogger
import sys
from contextlib import redirect_stdout
import io
import os


def assert_console_output(fn,assert_fn):
    f = io.StringIO()
    with redirect_stdout(f):
        fn()
    s = f.getvalue()
    assert assert_fn(s)


def test_file_logger(tmp_path: Path):
    logpath = tmp_path.joinpath("test.log")
    l = FileLogger(logpath, verbose=True, prefix="1")
    assert_console_output(lambda: l.info("hello"), lambda s: s.endswith("1 - hello\n"))
    lines = logpath.read_text().split("\n")
    assert len(lines) == 3
    assert lines[0].startswith("Starting Logger")
    assert lines[1].endswith("hello")
    
    assert_console_output(lambda: l.warn("hello"), lambda s: s.endswith("1 - \x1b[93mhello\x1b[0m\n"))
    lines = logpath.read_text().split("\n")
    assert len(lines) == 4
    assert lines[0].startswith("Starting Logger")
    assert lines[1].endswith("hello")
    
    assert_console_output(lambda: l.error("hello"), lambda s: s.endswith("1 - \x1b[91mhello\x1b[0m\n"))
    lines = logpath.read_text().split("\n")
    assert len(lines) == 5
    assert lines[0].startswith("Starting Logger")
    assert lines[1].endswith("hello")

    l.verbose = False
    assert_console_output(lambda: l.info("hello"), lambda s: len(s) == 0)
    assert_console_output(lambda: l.info("hello", verbose=True), lambda s: s.endswith("hello\n"))
    assert_console_output(lambda: l.warn("hello"), lambda s: s.endswith("1 - \x1b[93mhello\x1b[0m\n"))
    assert_console_output(lambda: l.warn("hello", verbose=False), lambda s: len(s)==0)
    assert_console_output(lambda: l.error("hello"), lambda s: s.endswith("\x1b[91mhello\x1b[0m\n"))



if __name__ == "__main__":

    tmp_path = Path("/tmp/")
    logfile = tmp_path.joinpath("test.log")
    if os.path.isfile(logfile):
        os.remove(logfile)
    test_file_logger(Path("/tmp/"))

    pass

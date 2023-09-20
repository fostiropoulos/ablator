import sys
import unittest
from unittest.mock import MagicMock

import pytest
from mock.mock import patch, Mock

from ablator.mp.utils import ray_init
from ablator.utils.progress_bar import (
    num_format,
    ProgressBar,
    RemoteProgressBar,
    RemoteDisplay,
    Display,
    in_notebook,
    get_last_line
)
import numpy as np
from pathlib import Path
import time
from collections import defaultdict
import ray
from multiprocessing import Process
import mock


def _assert_num_format(num, width):
    str_num = num_format(num, width)
    float_num = float(str_num)
    relative_error = (num - float_num) / (num + 1e-5)
    assert len(str_num) == width
    assert relative_error < 1e-5
    print(str_num)


def test_num_format():
    nums = [
        100000000000000000000000000000,
        100000000,
        100,
        100.000,
        0,
        0.0000,
        0.001,
        0.00000001,
        0.00000000001,
        1.0000000000001,
        10000000001.0010000000,
        10000000001.002000000,
        10000000001.00000000000,
        10000000001.0000000000001,
        10000000001.0000000000,
        10000000001.00000000000010000000000,
    ]
    widths = [8, 10, 12, 15, 200]
    for width in widths[::-1]:
        for i, num in enumerate(nums):
            for sign in [-1, 1]:
                _num = sign * num
                _assert_num_format(_num, width)
                _assert_num_format(np.float32(_num), width)
                try:
                    _assert_num_format(np.longlong(_num), width)
                except OverflowError:
                    pass


def _measure_dur(num):
    start = time.time()
    num_format(num)
    end = time.time()
    return end - start


def _measure_dur_naive(num):
    start = time.time()
    s = f"{num:.5e}"
    end = time.time()
    return end - start


def test_perf():
    durs = defaultdict(list)
    for i in range(10000):
        i = np.random.randint(0, 10**8)
        durs["int"].append(_measure_dur(i))
        durs["int_naive"].append(_measure_dur_naive(i))
        i = np.random.rand() * i
        durs["float"].append(_measure_dur(i))
        durs["float_naive"].append(_measure_dur_naive(i))


def _rand_str(size=10):
    return "".join([chr(ord("a") + np.random.randint(26)) for i in range(size)])


def write_text(fp: Path, n=1000):
    with open(fp, "a") as f:
        for i in range(n):
            f.write(f"Some rand Log: {_rand_str(100)}\n")
            f.flush()
            time.sleep(0.5)
        pass


def _test_tui(tmp_path: Path, progress_bar=None):
    uid = _rand_str(10)
    fp = tmp_path.joinpath(uid)
    fp.write_text("")
    p = Process(target=write_text, args=(fp,))
    p.start()

    b = ProgressBar(100000, logfile=fp, remote_display=progress_bar, uid=uid)

    s = {_rand_str(): np.random.random() for i in range(100)}
    for i in b:
        for k, v in s.items():
            s[k] = np.random.random()
        b.update_metrics(s, i)
        time.sleep(0.1)
    return


def _test_tui_remote(tmp_path: Path):
    if not ray.is_initialized():
        ray_init()
    import random

    @ray.remote
    def test_remote(i, progress_bar: RemoteProgressBar):
        _test_tui(tmp_path, progress_bar)

    trials = 20
    progress_bar = RemoteProgressBar.remote(trials)

    dis = RemoteDisplay(progress_bar)

    remotes = []
    for i in range(trials):
        remotes.append(test_remote.remote(i, progress_bar))
        time.sleep(random.random())
        dis.refresh(force=True)
    while len(remotes) > 0:
        done_id, remotes = ray.wait(remotes, num_returns=1, timeout=0.1)
        dis.refresh()
        time.sleep(random.random() / 10)

def test_in_notebook():
    # test case of simulating jupyter notebook environment
    ipython_mock = MagicMock()
    ipython_mock.config = {"IPKernelApp": True}
    with pytest.MonkeyPatch.context():
        pytest.MonkeyPatch().setattr("IPython.get_ipython", lambda: ipython_mock)
    assert in_notebook() is True

def test_in_notebook_with_no_ipkernel_attr():
    # test case of standard python env: IPython module could not be imported
    ipython_mock = MagicMock()
    ipython_mock.config = {}
    with pytest.MonkeyPatch.context():
        pytest.MonkeyPatch().setattr("IPython.get_ipython", lambda: ipython_mock)
    assert in_notebook() is False

def test_in_notebook_with_import_error():
    with patch('IPython.get_ipython', side_effect=ImportError("Cannot import get_ipython")):
        assert in_notebook() is False

def test_in_notebook_with_attribute_error():
    with patch('IPython.get_ipython', return_value=None):
        assert in_notebook() is False

def test_get_last_line(tmpdir):
    # create a temporary file
    test_file=Path(tmpdir,"test.txt")
    # create test cases
    lines=["This is the first line\n","This is the second line\n","This is the last line"]
    # write test cases into the temporary file
    with open(test_file,"w") as test_file:
        test_file.writelines(lines)
    # test if the function could return the last line of file correctly
    assert get_last_line(test_file)=="This is the last line"

def test_get_last_line_with_edge_cases(tmpdir):
    # edge test case: filename is None
    assert get_last_line(None)==None

    # edge test case: file does not exist
    temp_path=Path(tmpdir,"not_exist.txt")
    assert get_last_line(temp_path) == None

    # edge test case: only one line exists in the file.
    one_line_file=Path(tmpdir,"one_line.txt")
    with open(one_line_file,"w") as file:
        file.writelines("This is the last line without the new line character")
    assert get_last_line(one_line_file)=="This is the last line without the new line character"

    # edge test case: empty file
    empty_file=Path(tmpdir,"empty.txt")
    empty_file.touch()
    assert get_last_line(empty_file)==""

def test_display_class_init_function():
    # test __init__ function of Display class
    display=Display()
    assert hasattr(display,"nrows") and hasattr(display,"ncols") and not hasattr(display,"html_value")

def test_display_class_display_function():
    mock_display_instance = Display()
    mock_display_instance.ncols = None
    mock_display_instance._display("12345",0)
    last_line = mock_display_instance.stdscr.instr(0, 0, 5)
    assert last_line.decode('utf-8')=="     "

    display=Display()
    display._display("12345",0)
    last_line=display.stdscr.instr(0,0,5)
    assert last_line.decode('utf-8')=="12345"

def test_display_class_refresh_function():
    # test _refresh function of Display class
    display = Display()
    display._display("12345", 0)
    last_line = display.stdscr.instr(0, 0, 5)
    assert last_line.decode('utf-8') == "12345"
    display._refresh()
    last_line = display.stdscr.instr(0, 0, 5)
    assert last_line.decode('utf-8')=="     "

def test_display_class_update_screen_dims_function():
    # test _update_screen_dims function of Display class
    display=Display()
    nrows=display.nrows
    ncols=display.ncols
    display.stdscr.resize(nrows+1, ncols+1)
    display._update_screen_dims()
    nrows_update=display.nrows
    ncols_update=display.ncols
    assert nrows_update!=nrows and ncols_update!=ncols

def test_display_class_print_texts_function():
    # test print_texts function of Display class
    # mainly to test the function of print_texts() could work well
    display=Display()
    texts = ["hello", "world", "!"]
    display.print_texts(texts)
    assert display.stdscr.instr(0, 0, 5) == b"     "

if __name__ == "__main__":
    tmp_path = Path("/tmp/")
    # _test_tui(tmp_path)
    # _test_tui_remote(tmp_path)

import curses
import os
import select
import sys
import unittest
import pyte
from unittest.mock import MagicMock

import pytest
from mock.mock import patch, Mock
import re

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
    # test case of standard python env:
    # when there is an import error which means it's not in notebook, the return value should be False
    with patch('IPython.get_ipython', side_effect=ImportError("Cannot import get_ipython")):
        assert in_notebook() is False

def test_in_notebook_with_attribute_error():
    # test case of standard python env:
    # when there is an
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
    # test _display function when display instance's nclos or nrows is None
    mock_display_instance = Display()
    mock_display_instance.ncols = None
    mock_display_instance._display("12345",0)
    last_line = mock_display_instance.stdscr.instr(0, 0, 5)
    assert last_line.decode('utf-8')=="     "

    # test _display function when display instance's nclos and nrows is not None
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
    # test case of display class's print_texts function
    # since the texts are displayed in the terminal by using this function
    #   need to use pseudo-terminal to test if the result is correct
    def child_process_print_texts_function():
        display = Display()
        texts = ["hello", "world", "!"]
        display.print_texts(texts)
    keys={"hello":0,"world":1,"!":2}
    tui(child_process_print_texts_function,keys)

#
def display_class_close_function():
    display = Display()
    assert display.is_terminal==True
    display.close()
    assert display.is_terminal==False


def tui(test_func,assertion_func):
    pid, f_d = os.forkpty()
    if pid == 0:
        test_func()
    else:
        screen = pyte.Screen(80, 10)
        stream = pyte.ByteStream(screen)
        try:
            [f_d], _, _ = select.select(
                [f_d], [], [], 1)
        except (KeyboardInterrupt, ValueError):
            print("ValueError")
        except Exception:
            print("exception 1")
        else:
            print("read from screen")
            try:
                data = os.read(f_d, 10000)
                print("input stream")
                stream.feed(data)
                for line in screen.display:
                    print(line)
            except Exception:
                print("exception 2")
        print("Terminal Start")
        for line in screen.display:
            print(line)
        print("Terminal End")
        assertion_func(screen)

def test_progress_bar_class_init_function(tmpdir):
    # test case: if the attributes of an instance have all been initiated
    log_file = Path(tmpdir, "test.log")
    progress_bar = ProgressBar(10, 2, log_file, 1, None, "111111")
    assert progress_bar is not None
    assert progress_bar.total_steps == 10
    assert progress_bar.epoch_len == 2
    assert progress_bar.logfile == log_file
    assert progress_bar.update_interval == 1
    assert progress_bar.remote_display == None
    assert isinstance(progress_bar.display, Display)
    assert progress_bar.display.ncols != 0
    assert progress_bar.display.nrows != 0
    assert progress_bar.uid == "111111"

def test_progress_bar_class_init_function_display(tmpdir):
    # test case: if the texts of progress_bar have been correctly display after initiating( current_iteration=0)
    def child_process_progress_bar_init_function():
        log_file=Path(tmpdir,"test.log")
        progress_bar=ProgressBar(10,2,log_file,1,None,"111111")

    def assertion(screen):
        for key, value in keys.items():
            assert key in screen.display[value] or key==screen.display[value]

    keys={"111111":1}
    tui(child_process_progress_bar_init_function,assertion)

def test_progress_bar_class_reset_function(tmpdir):
    # test case: if the current_iteration attribute of instance is resetting to 0 correctly
    log_file = Path(tmpdir, "test.log")
    progress_bar = ProgressBar(10, 2, log_file, 1, None, "111111")
    progress_bar.current_iteration=3
    assert progress_bar.current_iteration==3
    progress_bar.reset()
    assert progress_bar.current_iteration==0

def test_progress_bar_class_make_bar_function():
    # test if the make_bar function return correctly when current_iteration>0
    bar = ProgressBar.make_bar(current_iteration=1, start_time=time.time() - 10, epoch_len=2,
                                total_steps=10, ncols=100)
    assert re.match(r'\s*50%\|█████     \| 1/2 \[00:10+<\d+:\d+, \d+\.\d+s/it, Remaining: \d+:\d+\]\s+',bar)

    # test if the make_bar function return correctly when current_iteration=0
    bar = ProgressBar.make_bar(current_iteration=0, start_time=time.time() - 10, epoch_len=2,
                                total_steps=10, ncols=100)
    assert re.match(r'\s*0%\|\s*\| 0/2 \[00:10<\?, \?it/s, Remaining: \?\?\]\s*',bar)

def progress_bar_class_make_metrics_message():
    # test case: if metrics could be converted to correct format when nrows and ncols are None
    metrics = {
        "metric1":123,
        "metric2":456.789,
        "metric3":"value",
    }
    expected_result = 'metric1:  00000123 | metric2:  456.7890 | metric3:      value'
    result = ProgressBar.make_metrics_message(metrics)
    assert re.match(result[0],expected_result)

    # test case: if metrics could be converted to separate lines when ncols attribute is given
    metrics={
        "metric1":123,
        "metric2":456.789,
        "metric3":"value1",
        "metric4":"value2"
    }
    expected_result1="metric1:  00000123 | metric2:  456.7890"
    expected_result2="metric3:    value1 | metric4:    value2"
    result=ProgressBar.make_metrics_message(metrics,ncols=50)
    assert re.match(result[0],expected_result1)
    assert re.match(result[1],expected_result2)

    # test case: if metrics could be cut off by the given nrows attribute
    metrics = {
        "metric1": 123,
        "metric2": 456.789,
        "metric3": "value1",
        "metric4": "value2"
    }
    expected_result = "metric1:  00000123 | metric2:  456.7890"
    result = ProgressBar.make_metrics_message(metrics, ncols=50, nrows=1)
    assert re.match(result[0],expected_result) and len(result)==1

def test_progress_bar_class_ncols():
    # test case: if ncols is returned correctly when display is not None
    progress_bar=ProgressBar(10)
    assert progress_bar.display is not None
    assert progress_bar.ncols==progress_bar.display.ncols

    # test case: if ncols is returned as None when display is None
    progress_bar.display=None
    assert progress_bar.ncols==None

def test_progress_bar_class_nrows():
    # test case: if nrows is returned correctly when display is not None
    progress_bar=ProgressBar(10)
    assert progress_bar.display is not None
    assert progress_bar.nrows==progress_bar.display.nrows

    # test case: if nrows is returned as None when display is None
    progress_bar.display=None
    assert progress_bar.nrows==None

def test_progress_bar_class_make_print_message(tmpdir):
    # test case of make_print_message when uid is not None
    keys_uid={"":0,r'111111:\s*0%\|\s*\| 0/2 \[00:00<\?, \?it/s, Remaining: \?\?\]\s*':1}
    def child_process_uid():
        log_file = Path(tmpdir, "test.log")
        progress_bar = ProgressBar(10, 2, log_file, 1, None, "111111")
    def assertion_uid(screen):
        for key,value in keys_uid.items():
            if value==0:
                assert screen.display[0].isspace()
            elif value==1:
                assert re.match(key,screen.display[value])

    tui(child_process_uid,assertion_uid)

    # test case of make_print_message when uid is None
    keys={r'\s*0%\|\s*\| 0/2 \[00:00<\?, \?it/s, Remaining: \?\?\]\s*':1}
    def child_process():
        log_file = Path(tmpdir, "test.log")
        progress_bar = ProgressBar(10, 2, log_file, 1, None)
    def assertion(screen):
        for key,value in keys.items():
            if value==0:
                assert screen.display[0].isspace()
            elif value==1:
                assert re.match(key,screen.display[value])
    tui(child_process,assertion)


def test_progress_bar_class_update(tmpdir):
    # test case: if the texts have been displayed after running _update function
    def child_process():
        log_file = Path(tmpdir, "test.log")
        progress_bar = ProgressBar(10, 2, log_file, 1, None, "111111")
    def assertion(screen):
        assert screen.display[1].isspace() is False

    tui(child_process, assertion)

def test_progress_bar_class_update_metrics(tmpdir):
    # test case: if the _update function is executed when current_iteration is 0
    keys={r'metric:\s+value':0,r'111111:\s+0%\|█*\s*\| 0/2 \[\d+:\d+<\?, \?it/s, Remaining: \?\?\]\s*':1}
    log_file = Path(tmpdir, "test.log")
    progress_bar = ProgressBar(10, 2, log_file, 1, None, "111111")
    def child_process_update_metrics_1():
        progress_bar.update_metrics({"metric":"value"},current_iteration=0)
    def assertion_update_metrics_1(screen):
        for key, value in keys.items():
            assert re.match(key, screen.display[value])
    tui(child_process_update_metrics_1,assertion_update_metrics_1)

def test_progress_bar_class_update_metrics_interval(tmpdir):
    # test case: if the _update function is executed when time.time() - self._prev_update_time > self.update_interval
    log_file = Path(tmpdir, "test.log")
    keys_interval={r'metric:\s+value':0,r'111111:\s+100%\|██████████\| 2/2 \[\d+:\d+<\d+:\d+, \d+\.\d+it/s, Remaining: \d+:\d+\]\s*':1}
    def child_process_update_metrics_2():
        progress_bar_interval.update_metrics({"metric":"value"},current_iteration=2)
    def assertion_update_metrics_2(screen):
        for key, value in keys_interval.items():
            print("key")
            print(key)
            assert re.match(key, screen.display[value])
    progress_bar_interval = ProgressBar(10, 2, log_file, 0, None, "111111")
    tui(child_process_update_metrics_2, assertion_update_metrics_2)

def test_progress_bar_class_update_metrics_current_iteration(tmpdir):
    # test case: if the _update function is executed when current_iteration equals epoch_len-1
    log_file = Path(tmpdir, "test.log")
    keys={r'metric:\s+value':0,r'111111:\s+50%\|█████     \| 1/2 \[\d+:\d+<\d+:\d+, \d+\.\d+it/s, Remaining: \d+:\d+\]\s*':1}
    def child_process_update_metrics_2():
        progress_bar.update_metrics({"metric":"value"},current_iteration=progress_bar.epoch_len-1)
    def assertion_update_metrics_2(screen):
        for key, value in keys.items():
            print("key")
            print(key)
            print(screen.display[value])
            assert re.match(key, screen.display[value])
    progress_bar = ProgressBar(10, 2, log_file, 1, None, "111111")
    tui(child_process_update_metrics_2, assertion_update_metrics_2)

if __name__ == "__main__":
    tmp_path = Path("/tmp/")
    # _test_tui(tmp_path)
    # _test_tui_remote(tmp_path)

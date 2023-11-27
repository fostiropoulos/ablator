import re
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock
import os
import numpy as np
import pytest
from mock.mock import patch

from ablator.utils.progress_bar import (
    Display,
    ProgressBar,
    get_last_line,
    in_notebook,
    num_format,
)


@pytest.fixture(autouse=True)
def set_custom_env_variable():
    os.environ["TERM"] = "linux"


def test_num_format():
    def _assert_num_format(num, width):
        str_num = num_format(num, width)
        float_num = float(str_num)
        relative_error = (num - float_num) / (num + 1e-5)
        assert len(str_num) == width
        assert relative_error < 1e-5

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


def test_perf():
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

    durs = defaultdict(list)
    for i in range(10000):
        i = np.random.randint(0, 10**8)
        durs["int"].append(_measure_dur(i))
        durs["int_naive"].append(_measure_dur_naive(i))
        i = np.random.rand() * i
        durs["float"].append(_measure_dur(i))
        durs["float_naive"].append(_measure_dur_naive(i))
    # Make sure that smart formatting is within a factor of 10 to naive formatting
    assert np.mean(durs["int"]) < np.mean(durs["int_naive"]) * 10
    assert np.mean(durs["float"]) < np.mean(durs["float_naive"]) * 10


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
    with patch(
        "IPython.get_ipython", side_effect=ImportError("Cannot import get_ipython")
    ):
        assert in_notebook() is False


def test_in_notebook_with_attribute_error():
    # test case of standard python env:
    # there is no return value when call IPython.get_ipython
    with patch("IPython.get_ipython", return_value=None):
        assert in_notebook() is False


def test_get_last_line(tmp_path):
    # create a temporary file
    test_file = Path(tmp_path, "test.txt")
    # create test cases
    lines = [
        "This is the first line\n",
        "This is the second line\n",
        "This is the last line",
    ]
    # write test cases into the temporary file
    with open(test_file, "w") as file:
        file.writelines(lines)
    # test if the function could return the last line of file correctly
    result = get_last_line(test_file)
    assert result == "This is the last line"


def test_get_last_line_with_edge_cases(tmp_path):
    # edge test case: filename is None
    assert get_last_line(None) is None

    # edge test case: file does not exist
    temp_path = Path(tmp_path, "not_exist.txt")
    assert get_last_line(temp_path) is None

    # edge test case: only one line exists in the file.
    one_line_file = Path(tmp_path, "one_line.txt")
    with open(one_line_file, "w") as file:
        file.writelines("This is the last line without the new line character")
    assert (
        get_last_line(one_line_file)
        == "This is the last line without the new line character"
    )

    # edge test case: empty file
    empty_file = Path(tmp_path, "empty.txt")
    empty_file.touch()
    assert get_last_line(empty_file) == ""


def test_display_class_init_function():
    # test __init__ function of Display class
    display = Display()
    assert (
        hasattr(display, "nrows")
        and hasattr(display, "ncols")
        and not hasattr(display, "html_value")
    )


def test_display_class_display_function():
    # test _display function when display instance's nclos or nrows is None
    mock_display_instance = Display()
    mock_display_instance.ncols = None
    mock_display_instance._display("12345", 0)
    last_line = mock_display_instance.stdscr.instr(0, 0, 5)
    assert last_line.decode("utf-8") == "     "

    # test _display function when display instance's nclos and nrows is not None
    display = Display()
    display._display("12345", 0)
    last_line = display.stdscr.instr(0, 0, 5)
    assert last_line.decode("utf-8") == "12345"


def test_display_class_refresh_function():
    # test _refresh function of Display class
    display = Display()
    display._display("12345", 0)
    last_line = display.stdscr.instr(0, 0, 5)
    assert last_line.decode("utf-8") == "12345"
    display._refresh()
    last_line = display.stdscr.instr(0, 0, 5)
    assert last_line.decode("utf-8") == "     "


def test_display_class_update_screen_dims_function():
    # test _update_screen_dims function of Display class
    display = Display()
    nrows = display.nrows
    ncols = display.ncols
    display.stdscr.resize(nrows + 1, ncols + 1)
    display._update_screen_dims()
    nrows_update = display.nrows
    ncols_update = display.ncols
    assert nrows_update != nrows and ncols_update != ncols


def test_display_class_close_function():
    display = Display()
    assert display.is_terminal
    display.close()
    assert not display.is_terminal


def test_progress_bar_class_init_function(tmp_path):
    # test case: if the attributes of an instance have all been initiated
    log_file = Path(tmp_path, "test.log")
    progress_bar = ProgressBar(10, 2, log_file, 1, None, "111111")
    assert progress_bar is not None
    assert progress_bar.total_steps == 10
    assert progress_bar.epoch_len == 2
    assert progress_bar.logfile == log_file
    assert progress_bar.update_interval == 1
    assert progress_bar.remote_display is None
    assert isinstance(progress_bar.display, Display)
    assert progress_bar.display.ncols != 0
    assert progress_bar.display.nrows != 0
    assert progress_bar.uid == "111111"


def test_progress_bar_class_reset_function(tmp_path):
    # test case: if the current_iteration attribute of instance is resetting to 0 correctly
    log_file = Path(tmp_path, "test.log")
    progress_bar = ProgressBar(10, 2, log_file, 1, None, "111111")
    progress_bar.current_iteration = 3
    assert progress_bar.current_iteration == 3
    progress_bar.reset()
    assert progress_bar.current_iteration == 0


def test_progress_bar_class_make_bar_function():
    # test if the make_bar function return correctly when current_iteration>0
    bar = ProgressBar.make_bar(
        current_iteration=1,
        start_time=time.time() - 10,
        epoch_len=2,
        total_steps=10,
        ncols=100,
    )
    assert re.match(
        (
            r"\s*50%\|█████     \| 1/2 \[00:10+<\d+:\d+, \d+\.\d+s/it, Remaining:"
            r" \d+:\d+\]\s+"
        ),
        bar,
    )

    # test if the make_bar function return correctly when current_iteration=0
    bar = ProgressBar.make_bar(
        current_iteration=0,
        start_time=time.time() - 10,
        epoch_len=2,
        total_steps=10,
        ncols=100,
    )
    assert re.match(r"\s*0%\|\s*\| 0/2 \[00:10<\?, \?it/s, Remaining: \?\?\]\s*", bar)


def test_progress_bar_class_make_metrics_message():
    # test case: if metrics could be converted to correct format when nrows and ncols are None
    metrics = {
        "metric1": 123,
        "metric2": 456.789,
        "metric3": "value",
    }
    expected_result = "metric1:  00000123 | metric2:  456.7890 | metric3:      value"
    result = ProgressBar.make_metrics_message(metrics)
    assert re.match(result[0], expected_result)

    # test case: if metrics could be converted to separate lines when ncols attribute is given
    metrics = {
        "metric1": 123,
        "metric2": 456.789,
        "metric3": "value1",
        "metric4": "value2",
    }
    expected_result1 = "metric1:  00000123 | metric2:  456.7890"
    expected_result2 = "metric3:    value1 | metric4:    value2"
    result = ProgressBar.make_metrics_message(metrics, ncols=50)
    assert re.match(result[0], expected_result1)
    assert re.match(result[1], expected_result2)

    # test case: if metrics could be cut off by the given nrows attribute
    metrics = {
        "metric1": 123,
        "metric2": 456.789,
        "metric3": "value1",
        "metric4": "value2",
        "metric5": "value1",
        "metric6": "value2",
    }
    expected_result = "metric1:  00000123 | metric2:  456.7890"
    result = ProgressBar.make_metrics_message(metrics, nrows=1)
    assert re.match(result[0], expected_result) and len(result) == 1


def test_progress_bar_class_ncols():
    # test case: if ncols is returned correctly when display is not None
    progress_bar = ProgressBar(10)
    assert progress_bar.display is not None
    assert progress_bar.ncols == progress_bar.display.ncols

    # test case: if ncols is returned as None when display is None
    progress_bar.display = None
    assert progress_bar.ncols is None


def test_progress_bar_class_nrows():
    # test case: if nrows is returned correctly when display is not None
    progress_bar = ProgressBar(10)
    assert progress_bar.display is not None
    assert progress_bar.nrows == progress_bar.display.nrows - 5

    # test case: if nrows is returned as None when display is None
    progress_bar.display = None
    assert progress_bar.nrows is None


def test_progress_bar_class_make_print_message(tmp_path):
    # test case of make_print_message when uid is not None
    re_uid = r"111111:\s*0%\|\s*\| 0/2 \[00:00<\?, \?it/s, Remaining: \?\?\]\s*"
    log_file = Path(tmp_path, "test.log")
    progress_bar = ProgressBar(10, 2, log_file, 1, None, "111111")
    texts = progress_bar.make_print_message()
    assert texts[0] == ""
    assert re.match(re_uid, texts[1])

    # test case of make_print_message when uid is None
    re_no_uid = r"\s*0%\|\s*\| 0/2 \[00:00<\?, \?it/s, Remaining: \?\?\]\s*"
    progress_bar = ProgressBar(10, 2, log_file, 1, None)
    texts = progress_bar.make_print_message()
    assert texts[0] == ""
    assert re.match(re_no_uid, texts[1])


if __name__ == "__main__":
    from tests.conftest import run_tests_local

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    fn_names = ["test_perf"]
    test_fns = [_locals[fn] for fn in fn_names]
    run_tests_local(test_fns)

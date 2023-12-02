import os
import re
from pathlib import Path

import pytest
import ray

from ablator.utils.progress_bar import ProgressBar, RemoteDisplay, RemoteProgressBar

# All tests in this file are marked as mp
pytestmark = pytest.mark.mp()


@pytest.fixture(autouse=True)
def set_custom_env_variable():
    os.environ["TERM"] = "linux"


def test_remote_progress_bar_make_bar_function(ray_cluster):
    # test case: test make_bar function of RemoteProgressBar class.
    remote_progress_bar = RemoteProgressBar.remote(10)
    bar = ray.get(remote_progress_bar.make_bar.remote())
    assert re.match(r"\s*0%\|\s*\| 0/10 \[00:00<\?, \?it/s, Remaining: \?\?\]\s*", bar)


def test_remote_progress_bar_make_print_texts(ray_cluster):
    # test case: test make_print_texts function of RemoteProgressBar class
    # test if the output message includes the metrics message of uid="111111"
    remote_progress_bar = RemoteProgressBar.remote(10)
    remote_progress_bar.update_status.remote("111111", ["status", "good"])
    texts = ray.get(remote_progress_bar.make_print_texts.remote())
    assert len(texts) == 2
    assert re.match(
        r"\s*0%\|\s*\| 0/10 \[00:00<\?, \?it/s, Remaining: \?\?\]\s*", texts[0]
    )
    assert texts[1] == "good | status"

    # test if the output message includes the metrics message of both uid="111111" and uid="222222"
    remote_progress_bar.update_status.remote("222222", ["status", "bad", "really"])
    texts = ray.get(remote_progress_bar.make_print_texts.remote())
    assert len(texts) == 4
    assert re.match(
        r"\s*0%\|\s*\| 0/10 \[00:00<\?, \?it/s, Remaining: \?\?\]\s*", texts[0]
    )
    assert texts[1] == "good | status"
    assert texts[2] == "bad | status"
    assert texts[3] == "     really"


def test_remote_progress_bar_close_function(ray_cluster):
    # test case: test close funtion of RemoteProgressBar class
    remote_progress_bar = RemoteProgressBar.remote(10)
    remote_progress_bar.update_status.remote("111111", ["status", "good"])
    texts = ray.get(remote_progress_bar.make_print_texts.remote())
    assert len(texts) == 2
    assert re.match(
        r"\s*0%\|\s*\| 0/10 \[00:00<\?, \?it/s, Remaining: \?\?\]\s*", texts[0]
    )
    assert texts[1] == "good | status"
    remote_progress_bar.close.remote("111111")
    texts = ray.get(remote_progress_bar.make_print_texts.remote())
    assert len(texts) == 1
    assert re.match(
        (
            r"\s*10%\|█         \| 1/10 \[00:00<\d+:\d+, \d+\.\d+it/s, Remaining:"
            r" \d+:\d+\]\s*"
        ),
        texts[0],
    )


def test_progress_bar_close_remote_progress_bar(tmp_path, ray_cluster):
    # test case: test if the close function of RemoteProgressBar class could be correctly used in Progress class
    log_file = Path(tmp_path, "test.log")
    remote_display = RemoteProgressBar.remote(10)
    progress_bar = ProgressBar(10, 2, log_file, 1, remote_display, "111111")
    remote_display.update_status.remote("111111", ["status", "good"])
    texts = ray.get(remote_display.make_print_texts.remote())
    assert len(texts) == 2
    assert re.match(
        r"\s*0%\|\s*\| 0/10 \[00:00<\?, \?it/s, Remaining: \?\?\]\s*", texts[0]
    )
    assert texts[1] == "good | status"
    progress_bar.close()
    texts = ray.get(remote_display.make_print_texts.remote())
    assert len(texts) == 1
    assert re.match(
        (
            r"\s*10%\|█         \| 1/10 \[00:00<\d+:\d+, \d+\.\d+it/s, Remaining:"
            r" \d+:\d+\]\s*"
        ),
        texts[0],
    )


def test_progress_bar_update_remote_progress_bar(tmp_path, ray_cluster):
    # test case: test if the update_status function of RemoteProgressBar class could be correctly used in _update function of Progress class
    log_file = Path(tmp_path, "test.log")
    lines = [
        "This is the first line\n",
        "This is the second line\n",
        "This is the last line",
    ]
    with open(log_file, "w") as file:
        file.writelines(lines)
    remote_progress_bar = RemoteProgressBar.remote(10)
    progress_bar = ProgressBar(10, 2, log_file, 1, remote_progress_bar, "111111")
    progress_bar.update_metrics({"metric": "value"}, 0)
    remote_display = RemoteDisplay(remote_progress_bar, 1)
    remote_display.refresh(True)


if __name__ == "__main__":
    from tests.conftest import run_tests_local

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]
    run_tests_local(test_fns)

import os
import re
import select
from pathlib import Path

import pytest

from ablator.utils.progress_bar import Display, ProgressBar

pytestmark = pytest.mark.skip(reason="TUI is currently unsupported.")


def tui(test_func, assertion_func):
    import pyte

    pid, f_d = os.forkpty()
    if pid == 0:
        test_func()
    else:
        screen = pyte.Screen(80, 10)
        stream = pyte.ByteStream(screen)
        try:
            [f_d], _, _ = select.select([f_d], [], [], 1)
        except (KeyboardInterrupt, ValueError):
            raise ValueError("Error occurred in the tui function.")
        else:
            try:
                data = os.read(f_d, 10000)
                stream.feed(data)
            except Exception:
                raise Exception
        assertion_func(screen)


def test_display_class_print_texts_function():
    # test case of display class's print_texts function
    # since the texts are displayed in the terminal by using this function
    #   need to use pseudo-terminal to test if the result is correct
    keys = {"hello": 0, "world": 1, "!": 2}
    display = Display()
    texts = ["hello", "world", "!"]

    def child_process_print_texts_function():
        display.print_texts(texts)

    def assertion_print_texts_function(screen):
        for key, value in keys.items():
            assert key in screen.display[value]

    tui(child_process_print_texts_function, assertion_print_texts_function)


def test_progress_bar_class_init_function_display(tmp_path):
    # test case: if the texts of progress_bar have been correctly display after initiating( current_iteration=0)
    def child_process_progress_bar_init_function():
        progress_bar_._update()

    def assertion_init_function_display(screen):
        for key, value in keys.items():
            assert re.match(key, screen.display[value])

    log_file = Path(tmp_path, "test.log")
    progress_bar_ = ProgressBar(10, 2, log_file, 1, None, "111111")
    keys = {r"111111:\s*0%\|\s*\| 0/2 \[00:00<\?, \?it/s, Remaining: \?\?\]\s*": 1}
    tui(child_process_progress_bar_init_function, assertion_init_function_display)


def test_progress_bar_class_update(tmp_path):
    # test case: if the texts have been displayed after running _update function
    log_file = Path(tmp_path, "test.log")
    progress_bar = ProgressBar(10, 2, log_file, 1, None, "111111")

    def child_process_update():
        progress_bar._update()

    def assertion_update(screen):
        assert screen.display[1].isspace() is False

    tui(child_process_update, assertion_update)


def test_progress_bar_class_update_metrics(tmp_path):
    # test case: if the _update function is executed when current_iteration is 0
    keys = {
        r"metric:\s+value": 0,
        r"111111:\s+0%\|\s*\| 0/2 \[\d+:\d+<\?, \?it/s, Remaining: \?\?\]\s*": 1,
    }
    log_file = Path(tmp_path, "test.log")
    progress_bar = ProgressBar(10, 2, log_file, 1, None, "111111")

    def child_process_update_metrics_1():
        progress_bar.update_metrics({"metric": "value"}, current_iteration=0)

    def assertion_update_metrics_1(screen):
        for key, value in keys.items():
            assert re.match(key, screen.display[value])

    tui(child_process_update_metrics_1, assertion_update_metrics_1)


def test_progress_bar_class_update_metrics_interval(tmp_path):
    # test case: if the _update function is executed when time.time() - self._prev_update_time > self.update_interval
    log_file = Path(tmp_path, "test.log")
    keys_interval = {
        r"metric:\s+value": 0,
        r"111111:\s+100%\|██████████\| 2/2 \[\d+:\d+<\d+:\d+, \d+\.\d+it/s, Remaining: \d+:\d+\]\s*": (
            1
        ),
    }

    def child_process_update_metrics_2():
        progress_bar_interval.update_metrics({"metric": "value"}, current_iteration=2)

    def assertion_update_metrics_2(screen):
        for key, value in keys_interval.items():
            assert re.match(key, screen.display[value])

    progress_bar_interval = ProgressBar(10, 2, log_file, 0, None, "111111")
    tui(child_process_update_metrics_2, assertion_update_metrics_2)


def test_progress_bar_class_update_metrics_current_iteration(tmp_path):
    # test case: if the _update function is executed when current_iteration equals epoch_len-1
    log_file = Path(tmp_path, "test.log")
    keys = {
        r"metric:\s+value": 0,
        r"111111:\s+50%\|█████     \| 1/2 \[\d+:\d+<\d+:\d+, \d+\.\d+it/s, Remaining: \d+:\d+\]\s*": (
            1
        ),
    }

    def child_process_update_metrics_2():
        progress_bar.update_metrics(
            {"metric": "value"}, current_iteration=progress_bar.epoch_len - 1
        )

    def assertion_update_metrics_2(screen):
        for key, value in keys.items():
            assert re.match(key, screen.display[value])

    progress_bar = ProgressBar(10, 2, log_file, 1, None, "111111")
    tui(child_process_update_metrics_2, assertion_update_metrics_2)


if __name__ == "__main__":
    # os.environ["TERM"] = "linux"

    from tests.conftest import run_tests_local

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]
    run_tests_local(test_fns)

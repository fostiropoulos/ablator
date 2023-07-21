import curses
import html
import os
import time
import typing as ty
from collections import defaultdict
from pathlib import Path

import ray
from tabulate import tabulate
from tqdm import tqdm

from ablator.utils.base import num_format

try:
    import ipywidgets as widgets
    from IPython.display import display
except ImportError:
    widgets = None


def in_notebook():
    try:
        # pylint: disable=import-outside-toplevel
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def get_last_line(filename: Path):
    if filename is None or not filename.exists():
        return None
    with open(filename, "rb") as f:
        try:  # catch OSError in case of a one line file
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
        return last_line


SEPERATOR = " | "


class Display:
    def __init__(self) -> None:
        self.is_terminal = not in_notebook()
        if self.is_terminal:
            # get existing stdout and redirect future stdout
            self._curses = curses
            self.stdscr = curses.initscr()
            self.stdscr.clear()
            self.nrows, self.ncols = self.stdscr.getmaxyx()
        else:
            assert (
                widgets is not None
            ), "Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html"
            self.ncols = int(1e3)
            self.nrows = int(1e3)
            self.html_widget = widgets.HTML(value="")
            self.html_value = ""
            display(self.html_widget)

    def _refresh(self):
        if self.is_terminal:
            self.stdscr.refresh()
            self.stdscr.clear()
        else:
            self.html_widget.value = self.html_value
            self.html_value = ""

    def _display(self, text, pos, is_last=False):
        if self.ncols is None or self.nrows is None:
            return

        # pylint: disable=import-outside-toplevel
        import _curses

        if self.is_terminal:
            try:
                _text = text[: self.ncols - 1] if is_last else text
                self.stdscr.addstr(pos, 0, _text)
            except _curses.error:
                pass
        else:
            self.html_value += html.escape(text) + "<br>"

    def close(self):
        if self.is_terminal:
            self._curses.nocbreak()
            self.stdscr.keypad(0)
            self._curses.echo()
            self._curses.endwin()
            self._curses.curs_set(1)  # Turn cursor back on
            self.is_terminal = False

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def _update_screen_dims(self):
        if self.is_terminal:
            self.nrows, self.ncols = self.stdscr.getmaxyx()

    def print_texts(self, texts):
        self._update_screen_dims()
        for i, text in enumerate(texts):
            self._display(text, i)
        self._refresh()


@ray.remote
class RemoteProgressBar:
    def __init__(self, total_trials: int | None):
        super().__init__()
        self.start_time: float = time.time()
        self.total_trials = total_trials if total_trials is not None else float("inf")
        self.closed: dict[str, bool] = defaultdict(lambda: False)
        self.texts: dict[str, list[str]] = defaultdict(lambda: [])
        self.finished_trials: int = 0

    def __iter__(self):
        for obj in range(self.total_trials):
            yield obj

    def close(self, uid):
        self.closed[uid] = True

    def make_bar(self):
        return ProgressBar.make_bar(
            current_iteration=self.current_iteration,
            start_time=self.start_time,
            total_steps=self.total_trials,
            epoch_len=self.total_trials,
            ncols=100,
        )

    @property
    def current_iteration(self):
        return sum(self.closed.values())

    def make_print_texts(self) -> list[str]:
        def _concat_texts(texts) -> list[str]:
            _texts = [f"{texts[1]}{SEPERATOR}{texts[0]}"]
            if len(texts) > 2:
                padding = " " * (len(texts[1].split(":")[0]) + 2)
                _texts.append(f"{padding}{texts[2]}")
            return _texts

        texts: list[str] = []
        texts.append(self.make_bar())

        for uid in sorted(self.texts):
            if not self.closed[uid]:
                texts += _concat_texts(self.texts[uid])

        return texts

    def update(self, finished_trials):
        self.finished_trials = finished_trials

    def update_status(self, uid: str, texts: list[str]):
        self.texts[uid] = texts


class RemoteDisplay(Display):
    def __init__(
        self, remote_progress_bar: RemoteProgressBar, update_interval: int = 1
    ) -> None:
        super().__init__()
        self._prev_update_time = time.time()
        self.update_interval = update_interval
        self.remote_progress_bar = remote_progress_bar

    def refresh(self, force=False):
        if time.time() - self._prev_update_time > self.update_interval or force:
            self._prev_update_time = time.time()
            self.print_texts(
                ray.get(self.remote_progress_bar.make_print_texts.remote())
            )


class ProgressBar:
    def __init__(
        self,
        total_steps,
        epoch_len: int | None = None,
        logfile: Path | None = None,
        update_interval: int = 1,
        remote_display: ty.Optional[RemoteProgressBar] = None,
        uid: str | None = None,
    ):
        if epoch_len is None:
            self.epoch_len = total_steps
        else:
            self.epoch_len = epoch_len
        self.total_steps = total_steps
        self.update_interval = update_interval
        self.start_time = time.time()
        self._prev_update_time = time.time()
        self.current_iteration = 0
        self.metrics: dict[str, ty.Any] = {}
        self.logfile = logfile
        self.display: Display | None = None
        self.remote_display: RemoteProgressBar | None = None
        if remote_display is None:
            self.display = Display()
        else:
            self.remote_display = remote_display
        self.uid = uid
        self._update()

    def __iter__(self):
        for obj in range(self.epoch_len):
            yield obj

    def reset(self) -> None:
        self.current_iteration = 0

    def close(self):
        if self.display is not None:
            self.display.close()
        else:
            self.remote_display.close.remote(self.uid)

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @classmethod
    def make_bar(
        cls,
        current_iteration: int,
        start_time: float,
        epoch_len: int,
        total_steps: int,
        ncols: int | None = None,
    ):
        if current_iteration > 0:
            rate = current_iteration / (time.time() - start_time)
            time_remaining = (total_steps - current_iteration) / rate
            ftime = tqdm.format_interval(time_remaining)
        else:
            ftime = "??"
        post_fix = f"Remaining: {ftime}"
        return tqdm.format_meter(
            current_iteration,
            epoch_len,
            elapsed=time.time() - start_time,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            postfix=post_fix,
            ncols=ncols,
        )

    @classmethod
    def make_metrics_message(
        cls,
        metrics: dict[str, ty.Any],
        nrows: int | None = None,
        ncols: int | None = None,
    ):
        rows = tabulate(
            [[k + ":", f"{num_format(v)}"] for k, v in metrics.items()],
            disable_numparse=True,
            tablefmt="plain",
            stralign="right",
        ).split("\n")
        text = ""
        texts = []
        for row in rows:
            row += SEPERATOR
            if ncols is not None and len(text) + len(row) > ncols:
                text = text[: -len(SEPERATOR)]
                texts.append(text)
                text = ""

            text += row
            if nrows is not None and len(texts) > nrows:
                break

        text = text[: -len(SEPERATOR)]
        texts.append(text)
        return texts

    @property
    def ncols(self):
        if self.display is not None:
            return self.display.ncols
        return None

    @property
    def nrows(self):
        if self.display is not None:
            return self.display.nrows - 5  # padding
        return None

    def make_print_message(self):
        texts = self.make_metrics_message(self.metrics, self.nrows, self.ncols)
        pbar = self.make_bar(
            current_iteration=self.current_iteration,
            start_time=self.start_time,
            total_steps=self.total_steps,
            epoch_len=self.epoch_len,
        )
        if self.uid is not None:
            texts.append(f"{self.uid}: {pbar}")
        else:
            texts.append(pbar)

        if (last_line := get_last_line(self.logfile)) is not None:
            texts.append(last_line)
        return texts

    def _update(self):
        texts = self.make_print_message()
        if self.display is not None:
            self.display.print_texts(texts)
        else:
            ray.get(self.remote_display.update_status.remote(self.uid, texts))
            if self.current_iteration + 1 == self.epoch_len:
                self.close()

    def update_metrics(self, metrics: dict[str, ty.Any], current_iteration: int):
        self.metrics = metrics
        self.current_iteration = current_iteration
        if (
            current_iteration == 0
            or time.time() - self._prev_update_time > self.update_interval
            or current_iteration + 1 == self.epoch_len
        ):
            self._prev_update_time = time.time()
            self._update()

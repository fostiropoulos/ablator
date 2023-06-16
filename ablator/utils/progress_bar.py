import curses
import os
import time
import typing as ty
from pathlib import Path

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

    def _display(self, text, pos):
        if self.ncols is None or self.nrows is None:
            return

        # pylint: disable=import-outside-toplevel
        import _curses

        if self.is_terminal:
            try:
                self.stdscr.addstr(pos, 0, text)
            except _curses.error:
                pass
        else:
            self.html_value += text + "<br>"

    def close(self):
        if self.is_terminal:
            curses.nocbreak()
            self.stdscr.keypad(0)
            curses.echo()
            curses.endwin()
            curses.curs_set(1)  # Turn cursor back on

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


class ProgressBar:
    def __init__(
        self,
        total,
        logfile: Path | None = None,
        update_interval: int = 1,
    ):
        self.total = total
        self.update_interval = update_interval
        self.start_time = time.time()
        self._prev_update_time = time.time()
        self.current_iteration = 0
        self.metrics: dict[str, ty.Any] = {}
        self.logfile = logfile
        self.display = Display()
        self._update()

    def __iter__(self):
        for obj in range(self.total):
            yield obj

    def reset(self) -> None:
        self.current_iteration = 0

    @classmethod
    def _make_bar(cls, current_iteration, start_time, total_steps):
        if current_iteration > 0:
            rate = current_iteration / (time.time() - start_time)
            time_remaining = (total_steps - current_iteration) / rate
            time_remaining = tqdm.format_interval(time_remaining)
        else:
            time_remaining = "??"
        post_fix = f"Remaining: {time_remaining}"
        return tqdm.format_meter(
            current_iteration,
            total_steps,
            elapsed=time.time() - start_time,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            postfix=post_fix,
        )

    @classmethod
    def _make_print_message(
        cls, metrics, nrows, ncols, logfile, current_iteration, start_time, total_steps
    ):
        rows = tabulate(
            [[k + ":", f"{num_format(v)}"] for k, v in metrics.items()],
            # maxcolwidths=[10, 10],
            disable_numparse=True,
            tablefmt="plain",
            stralign="right",
        ).split("\n")
        text = ""
        texts = []
        for row in rows:
            row += SEPERATOR
            if len(text) + len(row) > ncols:
                text = text[: -len(SEPERATOR)]
                texts.append(text)
                text = ""

            text += row
            if len(texts) > nrows - 5:
                break

        text = text[: -len(SEPERATOR)]
        texts.append(text)
        pbar = cls._make_bar(
            current_iteration=current_iteration,
            start_time=start_time,
            total_steps=total_steps,
        )
        texts.append(pbar)

        if logfile is not None and logfile.exists():
            last_line = get_last_line(logfile)
            texts.append(last_line)
        return texts

    def _update(self):
        texts = self._make_print_message(
            self.metrics,
            self.display.nrows,
            self.display.ncols,
            self.logfile,
            self.current_iteration,
            self.start_time,
            self.total,
        )
        self.display.print_texts(texts)

    def update_metrics(self, metrics: dict[str, ty.Any], current_iteration: int):
        self.metrics = metrics
        self.current_iteration = current_iteration
        if (
            current_iteration == 0
            or time.time() - self._prev_update_time > self.update_interval
            or current_iteration + 1 == self.total
        ):
            self._prev_update_time = time.time()
            self._update()

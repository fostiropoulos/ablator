import asyncio
import curses
import time
import typing as ty
from tabulate import tabulate
from tqdm import tqdm
from ablator.utils.base import num_format

try:
    import ipywidgets as widgets
    from IPython.display import display

except:
    widgets = None


def in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


SEPERATOR = " | "


class ProgressBar:
    def __init__(self, total, update_interval: int = 1):
        self.total = total
        self.update_interval = update_interval
        self.start_time = time.time()
        self._prev_update_time = time.time()
        self.total_steps = total
        self.current_iteration = 0
        self.is_terminal = not in_notebook()
        if self.is_terminal:
            self.stdscr = curses.initscr()
            self.stdscr.clear()
            self.nrows, self.ncols = self.stdscr.getmaxyx()
        else:
            assert (
                widgets is not None
            ), "Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html"
            self.ncols = float("inf")
            self.nrows = float("inf")
            self.html_widget = widgets.HTML(value="")
            self.html_value = ""
            display(self.html_widget)
        self._update([])

    def __iter__(self):
        for obj in range(self.total):
            yield obj
        return

    def __hash__(self):
        return id(self)

    def _refresh(self):
        if self.is_terminal:
            self.nrows, self.ncols = self.stdscr.getmaxyx()
            self.stdscr.refresh()
        else:
            self.html_widget.value = self.html_value
            self.html_value = ""

    def _update(self, rows):
        self._refresh()
        if self.ncols is None or self.nrows is None:
            return
        text = ""
        _pos = 0
        for row in rows:
            row += SEPERATOR
            if len(text) + len(row) > self.ncols:
                text = text[: -len(SEPERATOR)]
                self._display(text, pos=_pos)
                _pos += 1
                text = ""

            text += row
            if _pos > self.nrows - 5:
                return

        text = text[: -len(SEPERATOR)]
        self._display(text, pos=_pos)
        self._display(self.bar, pos=_pos + 1)
        self._refresh()

    def _display(self, text, pos):
        if self.is_terminal:
            self.stdscr.addstr(pos, 0, text)
        else:
            self.html_value += text + "<br>"

    def __exit__(self, exc_type, exc_value, traceback):
        if self.is_terminal:
            curses.endwin()
            curses.nocbreak()  # Turn off cbreak mode
            curses.echo()  # Turn echo back on
            curses.curs_set(1)  # Turn cursor back on

    def __del__(self):
        if self.is_terminal:
            curses.endwin()
            curses.nocbreak()  # Turn off cbreak mode
            curses.echo()  # Turn echo back on
            curses.curs_set(1)  # Turn cursor back on

    def reset(self) -> None:
        self.current_iteration = 0

    @property
    def bar(self):
        if self.current_iteration > 0:
            rate = self.current_iteration / (time.time() - self.start_time)
            time_remaining = (self.total_steps - self.current_iteration) / rate
            time_remaining = tqdm.format_interval(time_remaining)
        else:
            time_remaining = "??"
        post_fix = f"Remaining: {time_remaining}"
        return tqdm.format_meter(
            self.current_iteration,
            self.total,
            elapsed=time.time() - self.start_time,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            postfix=post_fix,
        )

    def _update_metrics(self, metrics, current_iteration):
        rows = tabulate(
            [[k + ":", f"{num_format(v)}"] for k, v in metrics.items()],
            # maxcolwidths=[10, 10],
            disable_numparse=True,
            tablefmt="plain",
            stralign="right",
        ).split("\n")
        self._update(rows)
        # self.update(current_iteration - self.current_iteration)
        self.current_iteration = current_iteration

    async def _async_update_metrics(self, metrics, current_iteration):
        return self._update_metrics(metrics, current_iteration)

    def update_metrics(self, metrics: dict[str, ty.Any], current_iteration: int):
        if time.time() - self._prev_update_time > self.update_interval:
            self._prev_update_time = time.time()
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # 'RuntimeError: There is no current event loop...'
                loop = None

            if loop:
                self._update_metrics(metrics, current_iteration)
            else:
                asyncio.run(self._async_update_metrics(metrics, current_iteration))

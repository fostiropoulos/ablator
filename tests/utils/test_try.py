# import pyte
# screen = pyte.Screen(40, 12)
# stream = pyte.Stream(screen)
# stream.feed("Hello World!")
# print(screen.display)
import curses
import os
import select
import time

import pyte
from ablator.utils.progress_bar import ProgressBar,Display
import pytest

def child_process():
    # display=Display()
    # display.print_texts(["hello world!"])
    progress_bar = ProgressBar(10, 2, None, 1, None, "111111")
    progress_bar.make_bar(current_iteration=3, start_time=time.time() - 10, epoch_len=progress_bar.epoch_len,
                                total_steps=progress_bar.total_steps, ncols=progress_bar.display.ncols)
    print("child process exit")

def test_tui():
    pid, f_d = os.forkpty()
    print(pid)
    print(f_d)
    if pid == 0:
        # child process spawns TUI
        # curses.wrapper(test_progress_bar_class_init_function())
        # curses.wrapper(ProgressBar)
        child_process()
        # os.kill(pid, 9)  # 发送 SIGKILL 信号强制终止进程
    else:
        screen = pyte.Screen(80, 10)
        stream = pyte.ByteStream(screen)
        # progress_bar = ProgressBar(10, 2, None, 1, None, "111111")
        # progress_bar.display.print_texts(["hello world"])
        print("xxxxxx")
        # parent process sets up virtual screen of
        # identical size
        # scrape pseudo-terminal's screen
        # while True:
        #     print("true while loop")
        try:
            [f_d], _, _ = select.select(
                [f_d], [], [], 1)
        except (KeyboardInterrupt, ValueError):
            # either test was interrupted or the
            # file descriptor of the child process
            # provides nothing to be read
            print("ValueError")
            # break
        else:
            try:
                # scrape screen of child process
                data = os.read(f_d, 1024)
                print("input stream")
                stream.feed(data)
                # for line in screen.display:
                #     print(line)
            except OSError:
                print("OSError")
                # reading empty
                # break
        print("Finally")
        for line in screen.display:
            print(line)
        print(screen.display[1])
        assert "111111" in screen.display[1]


def test_hhh():
    test_tui()
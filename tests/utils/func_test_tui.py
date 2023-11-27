import time
from multiprocessing import Process
from pathlib import Path

import numpy as np
import ray

from ablator.mp.utils import ray_init
from ablator.utils.progress_bar import ProgressBar, RemoteDisplay, RemoteProgressBar


def _rand_str(size=10):
    return "".join([chr(ord("a") + np.random.randint(26)) for i in range(size)])


def write_text(fp: Path, n=1000):
    with open(fp, "a") as f:
        for i in range(n):
            f.write(f"Some rand Log: {_rand_str(100)}\n")
            f.flush()
            time.sleep(0.5)
        pass


def test_tui(
    tmp_path: Path, progress_bar=None, n_metrics=20, duration: int | None = None
):
    """
    This is a functional test. Must be run directly.
    """
    uid = _rand_str(10)
    fp = tmp_path.joinpath(uid)
    fp.write_text("")
    p = Process(target=write_text, args=(fp,))
    p.start()

    b = ProgressBar(100000, logfile=fp, remote_display=progress_bar, uid=uid)

    s = {f"{i}{_rand_str()}": np.random.random() for i in range(n_metrics)}
    start_time = time.time()
    for i in b:
        for k, v in s.items():
            s[k] = np.random.random()
        b.update_metrics(s, i)
        time.sleep(0.1)
        if duration is not None and (time.time() - start_time) > duration:
            return
    return


def test_tui_remote(tmp_path: Path, trials=3, duration: int | None = None):
    if not ray.is_initialized():
        ray_init()
    import random

    @ray.remote
    def test_remote(i, progress_bar: RemoteProgressBar):
        test_tui(tmp_path, progress_bar, duration=duration)

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


if __name__ == "__main__":
    test_tui_remote(Path("/tmp/"))

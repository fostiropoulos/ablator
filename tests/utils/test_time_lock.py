import time
import pytest
import ray
from ablator.utils.base import Lock


@ray.remote(num_cpus=0.001)
def mock_remote(t: Lock):
    t.acquire()
    time.sleep(0.1)
    t.release()
    return True


def test_time_lock_ray():
    t = Lock(timeout=100)
    results = ray.get([mock_remote.remote(t) for i in range(10)])
    assert all(results)


def test_fail_lock_ray():
    t = Lock(timeout=1)
    t.acquire()
    with pytest.raises(TimeoutError, match="Could not obtain lock within 1.00 seconds"):
        ray.get([mock_remote.remote(t) for i in range(10)])
    assert True


def test_time_lock():
    t = Lock(timeout=1)
    t.acquire()
    with pytest.raises(TimeoutError, match="Could not obtain lock within 1.00 seconds"):
        with t:
            assert False
    with pytest.raises(TimeoutError, match="Could not obtain lock within 1.00 seconds"):
        with t:
            assert False
    t.release()
    with t:
        assert True
    with pytest.raises(ValueError, match="lock released too many times"):
        t.release()
    t.acquire()
    t.release()


if __name__ == "__main__":
    from tests.conftest import run_tests_local

    l = locals()
    fn_names = [fn for fn in l if fn.startswith("test_")]
    test_fns = [l[fn] for fn in fn_names]

    run_tests_local(test_fns)

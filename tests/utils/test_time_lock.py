import pytest
import ray
from ablator.utils.base import Lock


@pytest.mark.mp
def test_time_lock_ray(blocking_lock_remote):
    t = Lock(timeout=100)

    results = ray.get(
        [ray.remote(num_cpus=0.001)(blocking_lock_remote).remote(t) for i in range(10)],
        timeout=30,
    )
    assert all(results)


@pytest.mark.mp
def test_fail_lock_ray(blocking_lock_remote):
    t = Lock(timeout=1)
    t.acquire()
    with pytest.raises(TimeoutError, match="Could not obtain lock within 1.00 seconds"):
        ray.get(
            [
                ray.remote(num_cpus=0.001)(blocking_lock_remote).remote(t)
                for i in range(10)
            ]
        )
    assert True


@pytest.mark.mp
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

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]

    run_tests_local(test_fns)

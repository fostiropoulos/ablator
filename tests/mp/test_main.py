import copy
import functools
import shutil
import uuid
from pathlib import Path

import mock
import numpy as np
import pytest
import ray
import torch

from ablator.main.mp import ParallelTrainer
from ablator.main.state.store import TrialState
from ablator.modules.loggers.file import FileLogger
from ablator.mp.cluster import ClusterManager
from ablator.mp.gpu import GPUError
from ablator.mp.train_remote import (
    _handle_exception,
    _raise_or_ignore,
    train_main_remote,
)
from ablator.mp.utils import Resource, get_node_ip
from ablator.utils.file import expand_path

N_MOCK_NODES = 10

GPU_UTIL = 100  # mb


def available_resources(
    self: ClusterManager,
    mem_bottleneck_step=4,
    cpu_bottleneck_step=4,
    gpu_bottleneck_step=4,
    gpu_util=GPU_UTIL,
    mock_nodes=N_MOCK_NODES,
    incremental_running_tasks=True,
):
    self.cntr = getattr(self, "cntr", 1)
    mem = 100 if self.cntr > mem_bottleneck_step else 0
    cpu = 100 if self.cntr > cpu_bottleneck_step else 0
    n_remotes = self.cntr + 1
    free_mem = gpu_util - 1 if self.cntr > gpu_bottleneck_step else gpu_util + 1
    self.cntr = self.cntr + 1
    return {
        str(i): Resource(
            cpu_count=6,
            gpu_free_mem=[free_mem],
            mem=mem,
            cpu_usage=cpu,
            running_tasks=(
                [str(i) for i in range(n_remotes)] if incremental_running_tasks else []
            ),
        )
        for i in range(mock_nodes)
    }


def assert_bottleneck(trainer: ParallelTrainer, fn, bottleneck_step, soft_limit=1):
    trainer.cluster_manager.cntr = 1
    start_futures = len(trainer._make_futures(soft_limit=0))
    with (
        mock.patch(
            "ablator.mp.cluster.ClusterManager.available_resources",
            property(lambda self: fn(self)),
        ),
        mock.patch(
            "ablator.ParallelTrainer._gpu",
            new_callable=mock.PropertyMock,
            return_value=0.001,
        ),
    ):
        for i in range(bottleneck_step):
            trainer._heartbeat()
            futures = trainer._make_futures(soft_limit=soft_limit)
            assert len(futures) == start_futures + soft_limit * (i + 1)

        trainer._heartbeat()
        futures = trainer._make_futures(soft_limit=soft_limit)
        assert len(futures) == start_futures + soft_limit * (i + 1)


# stop iteration on sample trial
def error_resource(self, trial_id, trial, node_ip):
    if node_ip == "2":
        raise GPUError
    else:
        return trial_id


@pytest.mark.mp
def test_make_futures_resource_limits(
    tmp_path: Path, error_wrapper, make_config, working_dir
):
    tmp_path = tmp_path.joinpath("mock_dir")
    search_space_limit = 20
    # -1 because it is zero indexed
    config = make_config(
        tmp_path, search_space_limit=search_space_limit - 1, gpu_util=GPU_UTIL
    )
    n_nodes = 5
    config.optim_metrics = None
    # Starts a head-node for the cluster.
    config.total_trials = 100
    config.concurrent_trials = 30 // 5
    # Default limit to `_make_futures`
    soft_limit = 10
    # Make sure the concurrent limits are respected for each node.
    # _futures_scheduled_node
    with (
        mock.patch(
            "ablator.main.mp.ParallelTrainer._make_remote",
            lambda self, trial_id, trial, node_ip: trial_id,
        ),
        mock.patch(
            "ablator.mp.cluster.ClusterManager.available_resources",
            property(
                lambda self: available_resources(
                    self,
                    mem_bottleneck_step=1000,
                    cpu_bottleneck_step=1000,
                    gpu_bottleneck_step=1000,
                    mock_nodes=n_nodes,
                    gpu_util=100000,
                    incremental_running_tasks=False,
                )
            ),
        ),
    ):
        trainer = ParallelTrainer(wrapper=error_wrapper, run_config=config)
        trainer._init_state(working_dir)
        # This test did not catch an error with _is_limit and node specific
        # concourrent_trial_limit
        for i in range(1, 4):
            futures = trainer._make_futures(soft_limit=soft_limit)
            # should have sampled 0,1,2,0,1,2 ...
            assert futures == list(range(soft_limit * i))
            assert len(futures) == soft_limit * i
        futures = trainer._make_futures(soft_limit=soft_limit)
        assert list(range(soft_limit * i)) == futures
        trainer.run_config.concurrent_trials += 1
        sample_upper_limit = n_nodes * trainer.run_config.concurrent_trials

        futures = trainer._make_futures(soft_limit=soft_limit)
        assert list(range(sample_upper_limit)) == futures
        assert all(
            len(v) == trainer.run_config.concurrent_trials
            for v in trainer.running_futures.values()
        )

        trainer.run_config.concurrent_trials += 1
        for node in trainer.running_futures:
            assert (
                len(trainer.running_futures[node])
                == trainer.run_config.concurrent_trials - 1
            )
            futures = trainer._make_futures(soft_limit=1)
            assert (
                len(trainer.running_futures[node])
                == trainer.run_config.concurrent_trials
            )

        futures = trainer._make_futures(soft_limit=1)
        futures = trainer._make_futures(soft_limit=1)
        assert all(
            len(v) == trainer.run_config.concurrent_trials
            for v in trainer.running_futures.values()
        )

        sample_upper_limit = n_nodes * trainer.run_config.concurrent_trials
        futures = trainer._make_futures(soft_limit=soft_limit)
        assert list(range(sample_upper_limit)) == futures
        trainer.total_trials = 100
        trainer.run_config.concurrent_trials = 100000
        futures = trainer._make_futures(soft_limit=10)
        assert list(range(sample_upper_limit + 10)) == futures
        futures = trainer._make_futures(soft_limit=10000)
        assert list(range(trainer.total_trials)) == futures
        futures = trainer._make_futures(soft_limit=10000)
        assert list(range(trainer.total_trials)) == futures

        trainer.total_trials = 200
        futures = trainer._make_futures(soft_limit=10000)
        assert list(range(trainer.total_trials)) == futures
        # test what happens when no additional valid trials can be added, the total_trial limit is ignored.
        soft_limit = 10
        with mock.patch.object(
            trainer.experiment_state, "sample_trial", lambda: (0, config)
        ):
            trainer.total_trials = 205
            futures = trainer._make_futures(soft_limit=soft_limit)
            assert (
                list(range(trainer.total_trials - 5))
                == futures[: trainer.total_trials - 5]
            )
            lower = trainer.total_trials - 5
            assert [0] * soft_limit == futures[lower:]

        # stop iteration on sample trial
        def stop_iter():
            raise StopIteration

        current_futures = len(futures)
        with mock.patch.object(trainer.experiment_state, "sample_trial", stop_iter):
            trainer.total_trials = 1000
            futures = trainer._make_futures(soft_limit=soft_limit)
            assert len(futures) == current_futures

        # stop iteration on sample trial
        def error_trial():
            raise RuntimeError

        with (
            mock.patch.object(trainer.experiment_state, "sample_trial", error_trial),
            pytest.raises(RuntimeError),
        ):
            futures = trainer._make_futures(soft_limit=soft_limit)

    with (
        mock.patch(
            "ablator.main.mp.ParallelTrainer._make_remote",
            error_resource,
        ),
        mock.patch(
            "ablator.mp.cluster.ClusterManager.available_resources",
            property(
                lambda self: available_resources(
                    self,
                    mem_bottleneck_step=1000,
                    cpu_bottleneck_step=1000,
                    gpu_bottleneck_step=1000,
                    mock_nodes=n_nodes,
                    gpu_util=100000,
                    incremental_running_tasks=False,
                )
            ),
        ),
    ):
        prev_2_futures = len(trainer.running_futures["2"])

        prev_futures = len(futures)

        futures = trainer._make_futures(soft_limit=soft_limit)
        assert len(futures) == prev_futures + soft_limit
        assert len(trainer.running_futures["2"]) == prev_2_futures
        assert (
            len(trainer.running_futures["2"])
            < np.array(
                [len(trainer.running_futures[str(i)]) for i in range(n_nodes) if i != 2]
            )
        ).all()
    trainer.stop()


@pytest.mark.mp
def test_ray_init(
    tmp_path: Path, capture_output, error_wrapper, make_config, working_dir
):
    tmp_path = tmp_path.joinpath("mock_dir")
    config = make_config(tmp_path, search_space_limit=None)
    trainer = ParallelTrainer(wrapper=error_wrapper, run_config=config)
    trainer._init_state(working_dir)

    out, err = capture_output(lambda: trainer._init_state(working_dir, resume=True))
    assert "Ray is already initialized." in out
    trainer.stop()


@mock.patch(
    "ablator.ParallelTrainer._make_remote", lambda *args, **kwargs: (args, kwargs)
)
@pytest.mark.mp
def test_make_futures(
    tmp_path: Path, capture_output, error_wrapper, make_config, working_dir
):
    tmp_path = tmp_path.joinpath("mock_dir")
    config = make_config(tmp_path)
    # We remove sampling limits to test the limits in sampling
    # imposed by the resource allocation.
    config.concurrent_trials = None
    config.total_trials = None
    trainer = ParallelTrainer(wrapper=error_wrapper, run_config=config)
    trainer._init_state(working_dir)
    trainer._cpu
    del trainer._cpu
    out, err = capture_output(lambda: trainer._cpu)
    assert "Consider adjusting `concurrent_trials`." in out
    out, err = capture_output(lambda: trainer._cpu)
    assert len(out) == 0

    assert_bottleneck(trainer, available_resources, 4)

    # Test internal estimation of GPU usage.

    assert_bottleneck(
        trainer,
        lambda x: available_resources(
            x, mem_bottleneck_step=10, cpu_bottleneck_step=10, gpu_bottleneck_step=5
        ),
        5,
    )
    assert_bottleneck(
        trainer,
        lambda x: available_resources(
            x, mem_bottleneck_step=6, cpu_bottleneck_step=10, gpu_bottleneck_step=10
        ),
        6,
    )
    assert_bottleneck(
        trainer,
        lambda x: available_resources(
            x, mem_bottleneck_step=10, cpu_bottleneck_step=7, gpu_bottleneck_step=10
        ),
        7,
    )

    trainer.cluster_manager.cntr = 1
    futures_before = len(trainer._make_futures(0))
    with (
        mock.patch(
            "ablator.mp.cluster.ClusterManager.available_resources",
            property(lambda x: available_resources(x, gpu_util=GPU_UTIL - 1)),
        ),
        mock.patch(
            "ablator.ParallelTrainer._gpu",
            new_callable=mock.PropertyMock,
            return_value=0.001,
        ),
    ):
        futures = trainer._make_futures(soft_limit=1)
        assert len(futures) == futures_before
    trainer.stop()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason=(
        "The test is meant for left-over cuda memory. Not possible to evaluate without"
        " cuda support."
    ),
)
@pytest.mark.mp
def test_zombie_remotes(tmp_path: Path, wrapper, make_config, working_dir):
    tmp_path = tmp_path.joinpath("mock_dir")
    config = make_config(tmp_path)
    config.device = "cuda"
    ablator = ParallelTrainer(wrapper=wrapper, run_config=config)
    prev_memory_allocated = torch.cuda.memory_allocated()
    ablator._init_state(working_dir)
    assert prev_memory_allocated == torch.cuda.memory_allocated()


def _make_futures_clean(self):
    self.n_reps = getattr(self, "n_reps", 0)
    self.n_trials = getattr(self, "n_trials", 0)

    def _test(trial_id):
        return trial_id, None, None

    if self.n_reps >= 10:
        raise StopIteration

    self.n_reps += 1
    futures = []
    for i in range(10):
        futures.append(
            ray.remote(
                num_cpus=0.001,
                max_calls=1,
            )(
                _test
            ).remote(self.n_trials)
        )
        self.n_trials += 1
    return futures


def update_trial_state(self, trial_id, *args, **kwargs):
    self.last_trial_id = trial_id


def _make_futures_errors(self):
    self.n_reps = getattr(self, "n_reps", 0)
    self.n_trials = getattr(self, "n_trials", 0)
    self.futures = getattr(self, "futures", [])

    def _test(trial_id):
        if trial_id >= 6:
            raise RuntimeError
        return trial_id, None, None

    if len(self.futures) == 0:
        for i in range(10):
            self.futures.append(
                ray.remote(
                    num_cpus=0.001,
                    max_calls=1,
                )(
                    _test
                ).remote(self.n_trials)
            )
            self.n_trials += 1
        return self.futures
    else:
        self.futures.pop(0)
    self.n_reps += 1
    return self.futures


@pytest.mark.mp
def test_launch(tmp_path: Path, wrapper, make_config, working_dir):
    with (
        mock.patch(
            "ablator.ParallelTrainer._make_futures",
            _make_futures_clean,
        ),
        mock.patch(
            "ablator.main.state.state.ExperimentState.update_trial_state",
            update_trial_state,
        ),
    ):
        tmp_path = tmp_path.joinpath("mock_dir")
        config = make_config(tmp_path)
        config.device = "cuda"
        ablator = ParallelTrainer(wrapper=wrapper, run_config=config)
        ablator.launch(working_dir)
        assert ablator.n_reps == 10
        assert ablator.n_trials == ablator.n_reps * 10

    with (
        mock.patch(
            "ablator.ParallelTrainer._make_futures",
            _make_futures_errors,
        ),
        mock.patch(
            "ablator.main.state.state.ExperimentState.update_trial_state",
            update_trial_state,
        ),
    ):
        tmp_path = tmp_path.joinpath("mock_dir")
        config = make_config(tmp_path)
        config.device = "cuda"
        ablator = ParallelTrainer(wrapper=wrapper, run_config=config)
        ablator.launch(working_dir)
        assert ablator.n_reps == 6
    ablator.stop()


@pytest.mark.mp
def test_make_remote(tmp_path: Path, wrapper, make_config, working_dir):
    head_ip = get_node_ip()
    tmp_path = tmp_path.joinpath("mock_dir")
    config = make_config(tmp_path)
    config.device = "cpu"
    ablator = ParallelTrainer(wrapper=wrapper, run_config=config)
    ablator._init_state(working_dir)
    ablator.pre_train_setup()
    trial_id, trial = ablator.experiment_state.sample_trial()
    remote = ablator._make_remote(trial_id, config, node_ip=head_ip)
    uid, metrics, trial_state = ray.get(remote)
    ablator.experiment_state.update_trial_state(uid, metrics, trial_state)
    ablator.stop()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda focused test.")
@pytest.mark.mp
def test_make_remote_cuda(tmp_path: Path, wrapper, make_config, working_dir):
    head_ip = get_node_ip()
    tmp_path = tmp_path.joinpath("mock_dir")
    config = make_config(tmp_path)
    config.device = "cuda"
    ablator = ParallelTrainer(wrapper=wrapper, run_config=config)
    ablator._init_state(working_dir)
    with mock.patch(
        "ablator.mp.train_remote.train_main_remote", lambda: torch.cuda.is_available()
    ):
        trial_id, trial = ablator.experiment_state.sample_trial()
        remote = ablator._make_remote(trial_id, config, node_ip=head_ip)
        assert ray.get(remote)

    ablator.stop()


def test_train_main_remote(
    tmp_path: Path,
    assert_error_msg,
    capture_output,
    error_wrapper,
    wrapper,
    make_config,
    divergent_wrapper,
    custom_exception_class,
):
    config = make_config(tmp_path)
    experiment_dir = tmp_path.joinpath("mock_dir")
    config.experiment_dir = experiment_dir

    uid = 1029
    fault_tollerant = True
    crash_exceptions_types = None
    resume = False
    clean_reset = False
    progress_bar = None
    logger_path = tmp_path.joinpath("log.log")
    mp_logger = FileLogger(logger_path, verbose=True)
    resource_manager = None
    gpu = None
    remote_fn = functools.partial(
        train_main_remote,
        run_config=copy.deepcopy(config),
        mp_logger=mp_logger,
        resource_manager=resource_manager,
        gpu=gpu,
        uid=uid,
        fault_tollerant=fault_tollerant,
        crash_exceptions_types=crash_exceptions_types,
        resume=resume,
        clean_reset=clean_reset,
        progress_bar=progress_bar,
    )
    _wrapper = copy.deepcopy(wrapper)
    _new_uid, metrics, state = remote_fn(
        model=_wrapper,
    )
    assert _new_uid == uid and state == TrialState.COMPLETE
    assert (
        "val_loss" in metrics and metrics["current_epoch"] == config.train_config.epochs
    )
    _wrapper = copy.deepcopy(wrapper)
    out, err = capture_output(lambda: remote_fn(model=_wrapper))
    assert "Resume is set to False" in out
    config.train_config.epochs = 10
    resume = True
    uid = "xxxx"

    _wrapper = copy.deepcopy(wrapper)
    _new_uid, metrics, state = remote_fn(
        model=_wrapper, run_config=config, uid=uid, resume=resume
    )
    assert _new_uid == uid and state == TrialState.COMPLETE
    assert (
        "val_loss" in metrics and metrics["current_epoch"] == config.train_config.epochs
    )
    shutil.rmtree(experiment_dir)
    # test specifying and not specifying resource_manager
    with pytest.raises(ValueError, match="Must specify or leave unspecified"):
        _new_uid, metrics, state = remote_fn(
            run_config=config,
            uid=uid,
            model=copy.deepcopy(wrapper),
            resource_manager="X",
            gpu=None,
        )

    # this lr causes an error
    config.train_config.optimizer_config.arguments.lr = 11.0
    uid = str(uuid.uuid4())

    _new_uid, metrics, state = remote_fn(
        run_config=config, uid=uid, model=copy.deepcopy(error_wrapper)
    )
    assert state == TrialState.FAIL and metrics is None and _new_uid == uid

    shutil.rmtree(experiment_dir)

    msg = assert_error_msg(
        lambda: remote_fn(
            run_config=config,
            uid=uid,
            fault_tollerant=False,
            model=copy.deepcopy(error_wrapper),
        )
    )
    assert msg == "large lr."

    shutil.rmtree(experiment_dir)
    msg = assert_error_msg(
        lambda: remote_fn(
            run_config=config,
            uid=uid,
            crash_exceptions_types=[custom_exception_class],
            model=copy.deepcopy(error_wrapper),
        )
    )
    assert "large lr." in msg and "MyCustomException" in msg

    msg = assert_error_msg(
        lambda: remote_fn(
            model=copy.deepcopy(error_wrapper),
            run_config=config,
            uid=uid,
            fault_tollerant=False,
            crash_exceptions_types=[custom_exception_class],
            resume=True,
        )
    )
    assert "Could not find a valid checkpoint in" in msg

    _new_uid, metrics, state = remote_fn(
        model=copy.deepcopy(error_wrapper),
        run_config=config,
        uid=uid,
        fault_tollerant=False,
        crash_exceptions_types=[custom_exception_class],
        resume=True,
        clean_reset=True,
    )
    assert state == TrialState.FAIL_RECOVERABLE
    assert not Path(experiment_dir).exists()

    _wrapper = copy.deepcopy(wrapper)
    _new_uid, metrics, state = remote_fn(
        model=_wrapper,
        run_config=config,
        uid=uid,
        fault_tollerant=False,
        crash_exceptions_types=[custom_exception_class],
    )
    assert state == TrialState.COMPLETE

    shutil.rmtree(experiment_dir)
    _new_uid, metrics, state = remote_fn(
        model=copy.deepcopy(divergent_wrapper),
        run_config=config,
        fault_tollerant=False,
        crash_exceptions_types=[custom_exception_class],
    )
    assert state == TrialState.PRUNED_POOR_PERFORMANCE


def test_error_handling(
    tmp_path: Path,
    wrapper,
    make_config,
):
    logger_path = tmp_path.joinpath("log.log")
    mp_logger = FileLogger(logger_path, verbose=True)
    tmp_path = tmp_path / "mock"
    config = make_config(tmp_path)
    wrapper.init_state(
        config,
        resume=False,
        remote_progress_bar=None,
        data_lock=None,
    )
    resource_manager = None
    gpu = None
    uid = 0
    crash_exceptions_types = [RuntimeError]
    fault_tollerant = True
    try:
        raise ValueError
    except Exception as e:
        _uid, _metrics, _state = _handle_exception(
            e,
            wrapper,
            config,
            mp_logger,
            resource_manager,
            gpu,
            uid,
            fault_tollerant,
            crash_exceptions_types,
        )
        assert _uid == 0 and _metrics is None and _state == TrialState.FAIL
    # mock.patch unlock_gpu to raise an Error
    with pytest.raises(RuntimeError, match="in crash_exceptions_types"):
        try:
            raise ValueError
        except Exception as e:
            _handle_exception(
                e,
                wrapper,
                config,
                mp_logger,
                resource_manager,
                gpu,
                uid,
                fault_tollerant,
                [ValueError],
            )
    fault_tollerant = False
    with pytest.raises(ValueError):
        try:
            raise ValueError
        except Exception as e:
            _handle_exception(
                e,
                wrapper,
                config,
                mp_logger,
                resource_manager,
                gpu,
                uid,
                fault_tollerant,
                [],
            )
    gpu = 10
    with pytest.raises(Exception, match="ValueError.* AttributeError"):
        try:
            raise ValueError
        except Exception as e:
            _handle_exception(
                e,
                wrapper,
                config,
                mp_logger,
                resource_manager,
                gpu,
                uid,
                fault_tollerant,
                crash_exceptions_types,
            )

    with pytest.raises(Exception, match="Unknown error"):
        _raise_or_ignore(
            [],
            False,
            mp_logger,
            crash_exceptions_types,
        )


def test_relative_path(tmp_path: Path, wrapper, make_config):
    config = make_config(tmp_path)
    config.experiment_dir = "../dir/../dir2/."
    ablator = ParallelTrainer(wrapper=wrapper, run_config=config)
    assert Path().cwd().parent.joinpath("dir2").absolute() == ablator.experiment_dir
    config.experiment_dir = "~/dir2/."
    ablator = ParallelTrainer(wrapper=wrapper, run_config=config)
    assert (
        expand_path(config.experiment_dir)
        == Path.home().joinpath("dir2")
        == ablator.experiment_dir
    )
    config.experiment_dir = "~/../dir2/."

    ablator.stop()
    ablator = ParallelTrainer(wrapper=wrapper, run_config=config)
    assert (
        expand_path(config.experiment_dir)
        == Path.home().parent.joinpath("dir2")
        == ablator.experiment_dir
    )
    ablator.stop()


if __name__ == "__main__":
    from tests.conftest import run_tests_local
    from tests.test_plugins.model import (
        WORKING_DIR,
        MyCustomException,
        MyCustomModel,
        MyDivCustomModel,
        MyErrorCustomModel,
        TestWrapper,
        _make_config,
    )

    error_wrapper = TestWrapper(MyErrorCustomModel)
    wrapper = TestWrapper(MyCustomModel)
    div_wrapper = TestWrapper(MyDivCustomModel)

    _locals = locals()
    fn_names = [fn for fn in _locals if fn.startswith("test_")]
    test_fns = [_locals[fn] for fn in fn_names]

    kwargs = {
        "wrapper": copy.deepcopy(wrapper),
        "divergent_wrapper": copy.deepcopy(div_wrapper),
        "error_wrapper": error_wrapper,
        "working_dir": WORKING_DIR,
        "make_config": _make_config,
        "custom_exception_class": MyCustomException,
    }
    run_tests_local(test_fns, kwargs)

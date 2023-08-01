import functools
import shutil
import uuid
from pathlib import Path

import mock
import numpy as np
import pytest
import torch

from ablator.main.mp import ParallelTrainer
from ablator.main.state.store import TrialState
from ablator.modules.loggers.file import FileLogger
from ablator.mp.node_manager import NodeManager, Resource
from ablator.mp.train_remote import train_main_remote

N_MOCK_NODES = 10

GPU_UTIL = 100  # mb


def available_resources(
    self: NodeManager,
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
            gpu_free_mem={"v100": free_mem},
            mem=mem,
            cpu_usage=cpu,
            running_tasks=[str(i) for i in range(n_remotes)]
            if incremental_running_tasks
            else [],
        )
        for i in range(mock_nodes)
    }


def _test_bottleneck(trainer: ParallelTrainer, fn, bottleneck_step, soft_limit=1):
    trainer.node_manager.cntr = 1
    with mock.patch(
        "ablator.mp.node_manager.NodeManager.available_resources",
        fn,
    ), mock.patch(
        "ablator.ParallelTrainer._gpu",
        new_callable=mock.PropertyMock,
        return_value=0.001,
    ):
        for i in range(bottleneck_step):
            trainer._heartbeat()
            futures = trainer._make_futures(soft_limit=soft_limit)
            assert len(futures) == soft_limit

        trainer._heartbeat()
        futures = trainer._make_futures(soft_limit=soft_limit)
        assert len(futures) == 0


def test_mp_sampling_limits(tmp_path: Path, error_wrapper, make_config, working_dir):
    search_space_limit = 20
    # -1 because it is zero indexed
    config = make_config(
        tmp_path, search_space_limit=search_space_limit - 1, gpu_util=GPU_UTIL
    )
    n_nodes = 3
    config.optim_metrics = None
    # Starts a head-node for the cluster.
    config.total_trials = 20
    _sample_upper_limit = 10
    config.concurrent_trials = _sample_upper_limit
    # Default limit to `_make_futures`

    # Make sure the concurrent limits are respected for each node.
    # _futures_scheduled_node
    with mock.patch(
        "ablator.main.mp.ParallelTrainer._make_remote",
        lambda self, trial_id, trial, node_ip: node_ip,
    ), mock.patch(
        "ablator.mp.node_manager.NodeManager.available_resources",
        lambda self: available_resources(
            self,
            mem_bottleneck_step=1000,
            cpu_bottleneck_step=1000,
            gpu_bottleneck_step=1000,
            mock_nodes=n_nodes,
            gpu_util=100000,
            incremental_running_tasks=False,
        ),
    ):
        trainer = ParallelTrainer(wrapper=error_wrapper, run_config=config)
        trainer._init_state(working_dir)

        futures = trainer._make_futures()
        # should have sampled 0,1,2,0,1,2 ...
        assert futures == (["0", "1", "2"] * 4)[:_sample_upper_limit]
        assert len(futures) == trainer.run_config.concurrent_trials
        trainer.run_config.concurrent_trials = 1
        futures = trainer._make_futures()
        _sample_upper_limit = n_nodes * trainer.run_config.concurrent_trials
        assert futures == (["0", "1", "2"] * 4)[:_sample_upper_limit]
        assert len(futures) == _sample_upper_limit
        trainer.total_trials = 3
        futures = trainer._make_futures()
        assert len(futures) == 0

        # pass
        trainer.total_trials = 40
        futures = trainer._make_futures(soft_limit=40)
        assert len(futures) == _sample_upper_limit
        _sample_upper_limit = 100
        trainer.run_config.concurrent_trials = _sample_upper_limit
        prev_trials = len(trainer.experiment_state.valid_trials())
        futures = trainer._make_futures(soft_limit=40)
        assert len(futures) + prev_trials == trainer.total_trials

        trainer.total_trials = None
        futures = trainer._make_futures(soft_limit=40)
        assert len(futures) == 40
        assert len(trainer.experiment_state.valid_trials()) == 80


@pytest.mark.order(0)
def test_mp_run(assert_error_msg, working_dir, ablator, ray_cluster):
    complete_configs = ablator.experiment_state.get_trial_configs_by_state(
        TrialState.COMPLETE
    )
    failed_configs = ablator.experiment_state.get_trial_configs_by_state(
        TrialState.FAIL
    )
    lrs = np.array(
        [c.train_config.optimizer_config.arguments.lr for c in complete_configs]
    )
    bad_lrs = np.array(
        [c.train_config.optimizer_config.arguments.lr for c in failed_configs]
    )
    config = ablator.run_config
    n_trials = config.total_trials
    LR_ERROR_LIMIT = config.model_config.lr_error_limit
    n_complete = np.sum(
        np.linspace(0, 19, int(n_trials**0.5)) < LR_ERROR_LIMIT
    ) * int(n_trials**0.5)
    n_failed = np.sum(np.linspace(0, 19, int(n_trials**0.5)) > LR_ERROR_LIMIT) * int(
        n_trials**0.5
    )
    assert len(complete_configs) == n_complete
    assert len(failed_configs) == n_failed
    assert (lrs < LR_ERROR_LIMIT).all()
    assert (bad_lrs > LR_ERROR_LIMIT).all()
    assert len(failed_configs) == n_trials - n_complete
    msg = assert_error_msg(
        lambda: ablator._init_state(working_dir),
    )
    assert (
        f"Experiment Directory " in msg
        and config.experiment_dir in msg
        and "exists" in msg
    )

    prev_trials = len(ablator.experiment_state.valid_trials())
    ablator.launch(working_dir, resume=True)
    assert len(ablator.experiment_state.valid_trials()) == prev_trials
    ablator.run_config.total_trials += 4
    ablator.launch(working_dir, resume=True)
    assert (len(ablator.experiment_state.valid_trials()) != prev_trials) and (
        len(ablator.experiment_state.valid_trials()) == ablator.total_trials
    )


def test_ray_init(
    tmp_path: Path, capture_output, error_wrapper, make_config, working_dir
):
    config = make_config(tmp_path, search_space_limit=None)
    trainer = ParallelTrainer(wrapper=error_wrapper, run_config=config)
    trainer._init_state(working_dir)

    out, err = capture_output(lambda: trainer._init_state(working_dir, resume=True))
    assert "Ray is already initialized." in out


@mock.patch(
    "ablator.ParallelTrainer._make_remote", lambda *args, **kwargs: (args, kwargs)
)
def test_resource_util(
    tmp_path: Path, capture_output, error_wrapper, make_config, working_dir
):
    config = make_config(tmp_path)
    # We remove sampling limits to test the limits in sampling
    # imposed by the resource allocation.
    config.concurrent_trials = None
    config.total_trials = None
    trainer = ParallelTrainer(wrapper=error_wrapper, run_config=config)
    trainer._init_state(working_dir)
    del trainer._cpu
    out, err = capture_output(lambda: trainer._cpu)
    assert "Consider adjusting `concurrent_trials`." in out
    out, err = capture_output(lambda: trainer._cpu)
    assert len(out) == 0

    futures = trainer._make_futures()
    assert len(futures) == 10

    _test_bottleneck(trainer, available_resources, 4)

    # Test internal estimation of GPU usage.

    _test_bottleneck(
        trainer,
        lambda x: available_resources(
            x, mem_bottleneck_step=10, cpu_bottleneck_step=10, gpu_bottleneck_step=5
        ),
        5,
    )
    _test_bottleneck(
        trainer,
        lambda x: available_resources(
            x, mem_bottleneck_step=6, cpu_bottleneck_step=10, gpu_bottleneck_step=10
        ),
        6,
    )
    _test_bottleneck(
        trainer,
        lambda x: available_resources(
            x, mem_bottleneck_step=10, cpu_bottleneck_step=7, gpu_bottleneck_step=10
        ),
        7,
    )

    trainer.node_manager.cntr = 1
    with mock.patch(
        "ablator.mp.node_manager.NodeManager.available_resources",
        lambda x: available_resources(x, gpu_util=GPU_UTIL),
    ), mock.patch(
        "ablator.ParallelTrainer._gpu",
        new_callable=mock.PropertyMock,
        return_value=0.001,
    ):
        trainer._heartbeat()
        futures = trainer._make_futures(soft_limit=N_MOCK_NODES + 1)
        assert len(futures) == N_MOCK_NODES
    with mock.patch(
        "ablator.mp.node_manager.NodeManager.available_resources",
        lambda x: available_resources(x, gpu_util=GPU_UTIL * 2),
    ), mock.patch(
        "ablator.ParallelTrainer._gpu",
        new_callable=mock.PropertyMock,
        return_value=0.001,
    ):
        trainer._heartbeat()
        futures = trainer._make_futures(soft_limit=N_MOCK_NODES + 1)
        assert len(futures) == N_MOCK_NODES + 1
        trainer._heartbeat()
        futures = trainer._make_futures(soft_limit=N_MOCK_NODES * 2 + 1)

        assert len(futures) == N_MOCK_NODES * 2

    assert True


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The test is meant for left-over cuda memory. Not possible to evaluate without cuda support.",
)
def test_zombie_remotes(tmp_path: Path, wrapper, make_config, working_dir):
    config = make_config(tmp_path)
    config.device = "cuda"
    ablator = ParallelTrainer(wrapper=wrapper, run_config=config)
    prev_memory_allocated = torch.cuda.memory_allocated()
    ablator._init_state(working_dir)
    assert prev_memory_allocated == torch.cuda.memory_allocated()


def test_train_main_remote(
    tmp_path: Path,
    assert_error_msg,
    capture_output,
    error_wrapper,
    wrapper,
    make_config,
    divergent_wrapper,
):
    config = make_config(tmp_path)
    config.experiment_dir = tmp_path.joinpath("mock_dir")

    uid = 1029
    fault_tollerant = True
    crash_exceptions_types = None
    resume = False
    clean_reset = False
    progress_bar = None
    logger_path = tmp_path.joinpath("log.log")
    mp_logger = FileLogger(logger_path, verbose=True)
    gpu_manager = None
    gpu_id = None
    remote_fn = functools.partial(
        train_main_remote,
        model=wrapper,
        run_config=config,
        mp_logger=mp_logger,
        gpu_manager=gpu_manager,
        gpu_id=gpu_id,
        uid=uid,
        fault_tollerant=fault_tollerant,
        crash_exceptions_types=crash_exceptions_types,
        resume=resume,
        clean_reset=clean_reset,
        progress_bar=progress_bar,
    )
    _new_uid, metrics, state = remote_fn(
        model=wrapper,
        run_config=config,
    )
    assert _new_uid == uid and state == TrialState.COMPLETE
    assert (
        "val_loss" in metrics and metrics["current_epoch"] == config.train_config.epochs
    )
    out, err = capture_output(remote_fn)
    assert "Resume is set to False" in out
    config.train_config.epochs = 10
    resume = True
    uid = "xxxx"
    _new_uid, metrics, state = remote_fn(
        model=wrapper, run_config=config, uid=uid, resume=resume
    )
    assert _new_uid == uid and state == TrialState.COMPLETE
    assert (
        "val_loss" in metrics and metrics["current_epoch"] == config.train_config.epochs
    )

    shutil.rmtree(config.experiment_dir)
    # this lr causes an error
    config.train_config.optimizer_config.arguments.lr = 11.0
    uid = str(uuid.uuid4())
    _new_uid, metrics, state = remote_fn(
        model=error_wrapper, run_config=config, uid=uid
    )
    assert state == TrialState.FAIL and metrics is None and _new_uid == uid

    shutil.rmtree(config.experiment_dir)
    msg = assert_error_msg(
        lambda: remote_fn(
            model=error_wrapper, run_config=config, uid=uid, fault_tollerant=False
        )
    )
    assert msg == "large lr."

    shutil.rmtree(config.experiment_dir)
    msg = assert_error_msg(
        lambda: remote_fn(
            model=error_wrapper,
            run_config=config,
            uid=uid,
            crash_exceptions_types=[error_wrapper.model.exception_class],
        )
    )
    assert "large lr." in msg and "MyCustomException" in msg

    msg = assert_error_msg(
        lambda: remote_fn(
            model=error_wrapper,
            run_config=config,
            uid=uid,
            fault_tollerant=False,
            crash_exceptions_types=[error_wrapper.model.exception_class],
            resume=True,
        )
    )
    assert "Could not find a valid checkpoint in" in msg

    _new_uid, metrics, state = remote_fn(
        model=error_wrapper,
        run_config=config,
        uid=uid,
        fault_tollerant=False,
        crash_exceptions_types=[error_wrapper.model.exception_class],
        resume=True,
        clean_reset=True,
    )
    assert state == TrialState.FAIL_RECOVERABLE
    assert not Path(config.experiment_dir).exists()
    _new_uid, metrics, state = remote_fn(
        model=wrapper,
        run_config=config,
        uid=uid,
        fault_tollerant=False,
        crash_exceptions_types=[error_wrapper.model.exception_class],
    )
    assert state == TrialState.COMPLETE

    shutil.rmtree(config.experiment_dir)
    _new_uid, metrics, state = remote_fn(
        model=divergent_wrapper,
        run_config=config,
        fault_tollerant=False,
        crash_exceptions_types=[error_wrapper.model.exception_class],
    )
    assert state == TrialState.PRUNED_POOR_PERFORMANCE


def test_relative_path(tmp_path: Path, wrapper, make_config):
    config = make_config(tmp_path)
    config.experiment_dir = "../dir/../dir2/."
    ablator = ParallelTrainer(wrapper=wrapper, run_config=config)
    assert (
        Path().cwd().parent.joinpath("dir2").absolute() == ablator.experiment_dir.parent
    )


if __name__ == "__main__":
    from tests.conftest import DockerRayCluster, _assert_error_msg, _capture_output
    from tests.ray_models.model import (
        WORKING_DIR,
        MyCustomModel,
        MyDivCustomModel,
        MyErrorCustomModel,
        TestWrapper,
        _make_config,
    )

    tmp_path = Path("/tmp/experiment_dir")
    shutil.rmtree(tmp_path, ignore_errors=True)

    ray_cluster = DockerRayCluster(working_dir=WORKING_DIR)
    ray_cluster.setUp()
    error_wrapper = TestWrapper(MyErrorCustomModel)
    wrapper = TestWrapper(MyCustomModel)
    div_wrapper = TestWrapper(MyDivCustomModel)

    # shutil.rmtree(tmp_path, ignore_errors=True)
    # test_train_main_remote(
    #     tmp_path,
    #     _assert_error_msg,
    #     _capture_output,
    #     error_wrapper,
    #     make_config=_make_config,
    #     wrapper=wrapper,
    #     divergent_wrapper=div_wrapper,
    # )

    # shutil.rmtree(tmp_path, ignore_errors=True)
    # test_resource_util(
    #     tmp_path,
    #     _capture_output,
    #     error_wrapper=error_wrapper,
    #     make_config=_make_config,
    #     working_dir=WORKING_DIR,
    # )

    shutil.rmtree(tmp_path, ignore_errors=True)
    error_wrapper = TestWrapper(MyErrorCustomModel)
    test_mp_run(
        tmp_path,
        _assert_error_msg,
        ray_cluster,
        error_wrapper=error_wrapper,
        make_config=_make_config,
        working_dir=WORKING_DIR,
    )

    # shutil.rmtree(tmp_path, ignore_errors=True)

    # test_ray_init(
    #     tmp_path,
    #     _capture_output,
    #     error_wrapper=error_wrapper,
    #     make_config=_make_config,
    #     working_dir=WORKING_DIR,
    # )

    # shutil.rmtree(tmp_path, ignore_errors=True)
    # test_mp_sampling_limits(
    #     tmp_path,
    #     error_wrapper=error_wrapper,
    #     make_config=_make_config,
    #     working_dir=WORKING_DIR,
    # )
    # shutil.rmtree(tmp_path, ignore_errors=True)
    # test_zombie_remotes(
    #     tmp_path, wrapper=wrapper, make_config=_make_config, working_dir=WORKING_DIR
    # )
    # shutil.rmtree(tmp_path, ignore_errors=True)
    # test_relative_path(tmp_path, wrapper=wrapper, make_config=_make_config)

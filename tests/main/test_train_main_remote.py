import builtins
import os.path
import shutil
from contextlib import redirect_stderr, redirect_stdout
# from
import io
from pathlib import Path
import tempfile
import traceback

import pytest
import ray
import torch
from torch import nn

from ablator import (
    ModelConfig,
    ModelWrapper,
    OptimizerConfig,
    RunConfig,
    TrainConfig,
)

from ablator.main.state import ExperimentState, TrialState
from ablator.analysis.results import Results
from ablator.config.main import configclass
from ablator.main.configs import ParallelConfig, SearchSpace
from ablator.main.mp import ParallelTrainer
from ablator import Derived
from ablator.modules.loggers.file import FileLogger
from ablator.modules.loggers.main import SummaryLogger
from ablator.main.model.main import CheckpointNotFoundError, TrainPlateauError
from ablator.modules.loggers.main import DuplicateRunError
from ablator.modules.metrics.main import LossDivergedError
from ablator import Literal


class CustomModelConfig(ModelConfig):
    lr: Derived[int]


@configclass
class MyParallelConfig(ParallelConfig):
    model_config: CustomModelConfig


optimizer_config = OptimizerConfig(name="sgd", arguments={"lr": 0.1})
train_config = TrainConfig(
    dataset="test",
    batch_size=128,
    epochs=2,
    optimizer_config=optimizer_config,
    scheduler_config=None,
)
search_space = {
    "train_config.optimizer_config.arguments.lr": SearchSpace(
        value_range=[0, 10], value_type="int"
    ),
}

config = MyParallelConfig(
    train_config=train_config,
    model_config=CustomModelConfig(),
    verbose="silent",
    device="cpu",
    amp=False,
    search_space=search_space,
    optim_metrics={"val_loss": "min"},
    total_trials=10,
    concurrent_trials=10,
    gpu_mb_per_experiment=100,
    cpus_per_experiment=0.5,
)

class TestWrapper(ModelWrapper):
    def make_dataloader_train(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl

    def make_dataloader_val(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(100)]
        return dl

    def config_parser(self, run_config: MyParallelConfig):
        run_config.model_config.lr = (
            run_config.train_config.optimizer_config.arguments.lr
        )
        return super().config_parser(run_config)

class MyCustomModel(nn.Module):
    def __init__(self, config: CustomModelConfig) -> None:
        super().__init__()
        self.lr = config.lr
        self.param = nn.Parameter(torch.ones(100))
        self.itr = 0

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        self.itr += 1
        if self.itr > 10 and self.lr > 5:
            raise Exception("large lr.")
        return {"preds": x}, x.sum().abs()

def test_complete():
    wrapper = TestWrapper(MyCustomModel)
    logger = FileLogger(os.path.join(Path(__file__).parent, "mp.log"))
    result=train_main_remote(wrapper,config,logger,root_dir=Path(__file__).parent,fault_tollerant=False,crash_exceptions_types=None,resume=False,clean_reset=False)
    assert TrialState.COMPLETE in result, "Test fails while there should not be any problem!"

def test_duplicate_run_error():
    wrapper = TestWrapper(MyCustomModel)
    wrapper.model_dir=os.path.join(Path(__file__).parent)
    logger = FileLogger(os.path.join(Path(__file__).parent, "mp.log"))
    result = train_main_remote(wrapper, config, logger, root_dir=Path(__file__).parent, fault_tollerant=False,
                               crash_exceptions_types=None, resume=False, clean_reset=False)
    assert TrialState.RECOVERABLE_ERROR in result, "DuplicateError is not triggered!"

def test_runtime_error():
    wrapper = TestWrapper(MyCustomModel)
    wrapper._init_state(smoke_test=False,run_config=config)
    logger = FileLogger(Path(__file__).parent / "mp.log")
    with pytest.raises(RuntimeError):
        train_main_remote(wrapper, config, logger, root_dir=os.path.join(Path(__file__).parent , "exist"), fault_tollerant=False,
                               crash_exceptions_types=None, resume=True, clean_reset=False)

def capture_output(fn):
    out = io.StringIO()

    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        fn()

    return out.getvalue(), err.getvalue()

def train_main_remote(
    model: ModelWrapper,
    run_config: ParallelConfig,
    mp_logger: FileLogger,
    root_dir: Path,
    fault_tollerant: bool = True,
    crash_exceptions_types: list[type] | None = None,
    resume: bool = False,
    clean_reset: bool = False,
) -> tuple[ParallelConfig, dict[str, float] | None, TrialState]:
    """
    The trial job that will be executed remotely at a ray node. This is where model training happens.
    In addition, experiment directory will be synchronized to the Google Cloud storage and remote nodes.
    Synchronization is done via GcpConfig and RemoteConfig ``rsync_up()`` methods. Refer to documentation of
    these 2 classes for more details.

    Parameters
    ----------
    model : ModelWrapper
        The ModelWrapper that is used to train a model.
    run_config : ParallelConfig
        Runtime configuration for this trial.
    mp_logger : FileLogger
        The file logger that's used to log training progress.
    root_dir : Path
        The root directory that stores experiment states (experiment directory).
    fault_tollerant : bool, optional, default=True
        Whether to tollerate crashes, aka to cease execution when the ray job crashes.
    crash_exceptions_types : list[type], None, optional, default=None
        Types of exceptions that are considered as crashes.
    resume : bool, default=False
        Whether to resume training the model from existing checkpoints and existing experiment state.
    clean_reset : bool, default=False
        Whether to remove model directory when ``CheckpointNotFoundError`` is raised.

    Returns
    -------
    ParallelConfig
        Running configuration of the trial.
    dict[str, float], None
        If exception raised (Except for LossDivergedError and TrainPlateauError),
        this will be ``None`` object. Otherwise, this will be a dictionary of metrics.
    TrialState
        A TrialState object indicating the state of the trial job

        - If ``LossDivergedError`` or ``TrainPlateauError`` is raised while training,
          returned state will be ``TrialState.PRUNED_POOR_PERFORMANCE``

        - If ``DuplicateRunError``, ``RuntimeError`` (with message ``'CUDA out of memory'``),
          or ``CheckpointNotFoundError`` (with ``clean_reset=True``) is raised while training,
          returned state will be ``TrialState.RECOVERABLE_ERROR``

        - If other types of error or ``CheckpointNotFoundError`` (with ``clean_reset=False``) is raised,
          returned state will be ``TrialState.FAIL``

    """
    if crash_exceptions_types is None:
        crash_exceptions_types = []
        print("None crash_exception_types has been converted to empty list.")

    def handle_exception(e):
        exception_str = traceback.format_exc()
        if hasattr(model, "logger"):
            model.logger.error(exception_str)
        mp_logger.error(f"Error Occured {run_config.uid}")
        traceback.print_exc()
        if not fault_tollerant or isinstance(e, tuple(crash_exceptions_types)):
            error_msg = (
                f"Error {type(e).__name__} in"
                f"{' '.join([c.__name__ for c in crash_exceptions_types])}. Exiting."
            )
            mp_logger.error(error_msg)
            raise type(e)(error_msg)

        return run_config, None, TrialState.FAIL

    try:
        res = model.train(run_config, resume=resume)
        mp_logger.info(f"Finished training - {run_config.uid}")
        return run_config, res.to_dict(), TrialState.COMPLETE
    except (LossDivergedError, TrainPlateauError):
        return (
            run_config,
            model.metrics.to_dict(),
            TrialState.PRUNED_POOR_PERFORMANCE,
        )
    except DuplicateRunError:
        return (
            run_config,
            None,
            TrialState.RECOVERABLE_ERROR,
        )
    except CheckpointNotFoundError:
        if clean_reset:
            if model.model_dir is not None:
                shutil.rmtree(model.model_dir.as_posix())
            return (
                run_config,
                None,
                TrialState.RECOVERABLE_ERROR,
            )

        return (
            run_config,
            None,
            TrialState.FAIL,
        )
    except RuntimeError as e:
        if str(e).startswith("CUDA out of memory."):
            mp_logger.warn(f"Cuda out of memory for {run_config.uid}. Restarting...")
            return (
                run_config,
                None,
                TrialState.RECOVERABLE_ERROR,
            )

        return handle_exception(e)

    except builtins.Exception as e:
        return handle_exception(e)
    finally:
        if model.model_dir is not None:
            kwargs = parse_rsync_paths(model.model_dir, root_dir)
            if run_config.gcp_config is not None:
                run_config.gcp_config.rsync_up(
                    Path(kwargs["local_path"]),
                    str(kwargs["remote_path"]),
                    logger=mp_logger,
                )
                print("Resync to GCP")
            elif run_config.remote_config is not None:
                run_config.remote_config.rsync_up(
                    Path(kwargs["local_path"]), str(kwargs["remote_path"])
                )
                print("Remote Resync successfully")

def parse_rsync_paths(
    rsynced_folder: Path | str,
    root_folder: Path | str | None = None,
) -> dict[str, Path | str]:
    """
    Parse the experiment directory that's being in sync with remote servers (Google cloud storage, other
    remote nodes) and the root folder.

    Parameters
    ----------
    rsynced_folder : Path, str
        The experiment directory that's being in sync with remote servers.
    root_folder : Path, str, None, default=None
        The root folder that contains all experiment directories.

    Returns
    -------
    dict[str, Path]
        A dictionary with 2 keys: ``local_path`` and ``remote_path``, which specifies the local directory
        and the remote path that will be in sync.
    """
    rsync_path = Path(rsynced_folder)
    root_path = rsync_path if root_folder is None else Path(root_folder)

    return {
        "local_path": rsync_path,
        "remote_path": rsync_path.relative_to(root_path.parent).as_posix(),
    }

if __name__ == "__main__":
    import shutil

    # tmp_path =Path(__file__).parent
    # shutil.rmtree(local_path, ignore_errors=True)
    # local_path.mkdir()
    test_complete()
    test_duplicate_run_error()
    test_runtime_error()

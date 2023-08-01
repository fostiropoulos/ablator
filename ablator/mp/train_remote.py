import builtins
import os
import shutil
import traceback
import typing as ty
from functools import wraps
from functools import partial

from ablator.mp.gpu_manager import GPUManager, unlock_gpu
import ablator.utils.base as butils
from ablator.config.mp import ParallelConfig
from ablator.main.model.main import CheckpointNotFoundError, TrainPlateauError
from ablator.main.model.wrapper import ModelWrapper

from ablator.main.state import TrialState
from ablator.modules.loggers.file import RemoteFileLogger
from ablator.modules.metrics.main import LossDivergedError

from ablator.utils.progress_bar import RemoteProgressBar


# pylint: disable=protected-access
def _apply_unlock_hook(
    model: ModelWrapper, gpu_manager: ty.Union[GPUManager, None], gpu: int | None = None
) -> ModelWrapper:
    if gpu_manager is None:
        return model
    model._is_locked = True  # type: ignore

    def lock_on_unlocked():
        if model._is_locked:
            unlock_gpu(gpu_manager, gpu)
            model._is_locked = False

    def hook_function(function, hook_fn):
        @wraps(function)
        def run(*args, **kwargs):
            hook_fn()
            return function(*args, **kwargs)

        return run

    model.train_step = hook_function(model.train_step, lock_on_unlocked)  # type: ignore
    return model


def _handle_exception(
    e: Exception,
    model: ModelWrapper,
    run_config: ParallelConfig,
    mp_logger: RemoteFileLogger,
    gpu_manager: ty.Union[GPUManager, None],
    gpu_id: int,
    uid: int,
    fault_tollerant: bool,
    crash_exceptions_types: list[type] | None,
) -> tuple[int, dict[str, float] | None, TrialState]:
    if crash_exceptions_types is None:
        crash_exceptions_types = []

    if gpu_manager is not None:
        # gpu_manager is a ray-actor
        unlock_gpu(gpu_manager, gpu_id)
    exception_str = traceback.format_exc()
    if hasattr(model, "logger"):
        model.logger.error(exception_str)
    else:
        mp_logger.error(f"{run_config.uid}:\n{exception_str}")
    mp_logger.error(f"Error Occured {run_config.uid}: {str(e)}")
    if not fault_tollerant:
        raise e
    if isinstance(e, tuple(crash_exceptions_types)):
        error_msg = (
            f"Exception `{str(e)}` of type: `{type(e).__name__}` in crash_exceptions_types="
            f"{{c.__name__ for c in crash_exceptions_types}}. Exiting."
        )
        mp_logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    return uid, None, TrialState.FAIL


# TODO refactor into a seperate file and reduce complexity
# pylint: disable=broad-exception-caught,too-many-arguments
def train_main_remote(
    model: ModelWrapper,
    run_config: ParallelConfig,
    mp_logger: RemoteFileLogger,
    gpu_manager: ty.Union[GPUManager, None],
    gpu_id: int | None,
    uid: int,
    fault_tollerant: bool = True,
    crash_exceptions_types: list[type] | None = None,
    resume: bool = False,
    clean_reset: bool = False,
    progress_bar: ty.Optional[RemoteProgressBar] = None,
) -> tuple[int, dict[str, float] | None, TrialState]:
    """
    The trial job that will be executed remotely at a ray node. This is where model training happens.
    In addition, experiment directory will be synchronized to the Cloud storage and remote nodes.

    Parameters
    ----------
    model : ModelWrapper
        The ModelWrapper that is used to train a model.
    run_config : ParallelConfig
        Runtime configuration for this trial.
    mp_logger : FileLogger
        The file logger that's used to log training progress.
    gpu_manager : GPUManager | None
        The gpu manager that is used to inform when the training progress starts
    gpu_id : int | None
        The gpu id to which to assign resources on the current remote.
    uid : int
        the trial unique identifier.
    fault_tollerant : bool, optional, default=True
        Whether to tollerate crashes, aka to cease execution when the ray job crashes.
    crash_exceptions_types : list[type], None, optional, default=None
        Types of exceptions that are considered as crashes.
    resume : bool, default=False
        Whether to resume training the model from existing checkpoints and existing experiment state.
    clean_reset : bool, default=False
        Whether to remove model directory when ``CheckpointNotFoundError`` is raised.
    progress_bar : RemoteProgressBar, optional
        Optionally, we can use a remote progress bar to update the results of the trial.

    Returns
    -------
    int
        The trial uid corresponding to the results.
    dict[str, float], None
        If exception raised (Except for LossDivergedError and TrainPlateauError),
        this will be ``None`` object. Otherwise, this will be a dictionary of metrics.
    TrialState
        A TrialState object indicating the state of the trial job

        - If ``LossDivergedError`` or ``TrainPlateauError`` is raised while training,
          returned state will be ``TrialState.PRUNED_POOR_PERFORMANCE``

        - If ``RuntimeError`` (with message ``'CUDA out of memory'``),
          or ``CheckpointNotFoundError`` (with ``clean_reset=True``) is raised while training,
          returned state will be ``TrialState.FAIL_RECOVERABLE``

        - If other types of error or ``CheckpointNotFoundError`` (with ``clean_reset=False``) is raised,
          returned state will be ``TrialState.FAIL``

    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # We add a hook after the first train_step to update the gpu manager
        _apply_unlock_hook(model, gpu_manager, gpu=gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    handle_exception = partial(
        _handle_exception,
        model=model,
        run_config=run_config,
        mp_logger=mp_logger,
        gpu_manager=gpu_manager,
        gpu_id=gpu_id,
        uid=uid,
        fault_tollerant=fault_tollerant,
        crash_exceptions_types=crash_exceptions_types,
    )

    try:
        # NOTE in order for os.environ to wotk CUDA must be unitialized
        # up to this point.
        res = model.train(run_config, resume=resume, remote_progress_bar=progress_bar)
        mp_logger.info(f"Finished training - {run_config.uid}")
        return uid, res, TrialState.COMPLETE
    except (LossDivergedError, TrainPlateauError) as e:
        mp_logger.warn(
            f"Trial {run_config.uid} was pruned for poor performance. {str(e)}"
        )
        return (
            uid,
            model.metrics,
            TrialState.PRUNED_POOR_PERFORMANCE,
        )
    except CheckpointNotFoundError as e:
        # This is the case for corrupt artifacts.
        if clean_reset:
            if model.experiment_dir is not None:
                shutil.rmtree(model.experiment_dir.as_posix())
            return (
                uid,
                None,
                TrialState.FAIL_RECOVERABLE,
            )

        return handle_exception(e)
    except RuntimeError as e:
        if butils.is_oom_exception(e):
            mp_logger.warn(f"Cuda out of memory for {run_config.uid}. Restarting...")
            return (
                uid,
                None,
                TrialState.FAIL_RECOVERABLE,
            )
        return handle_exception(e)
    except builtins.Exception as e:
        return handle_exception(e)

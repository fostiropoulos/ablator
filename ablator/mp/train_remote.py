import builtins
import os
import shutil
import traceback
import typing as ty
from functools import wraps
from functools import partial


from ablator.mp.gpu import GPU, ResourceManager, unlock_gpu
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
    model: ModelWrapper,
    resource_manager: ResourceManager,
    gpu: GPU,
):
    """
    Sets a hook on the model-wrapper to unlock the GPU resources once
    the `train_step` function is called. This is to be able to ensure
    that the correct GPU utilization has been recorded for the
    train remote.

    Parameters
    ----------
    model : ModelWrapper
        The model to apply the hook to
    resource_manager : ResourceManager
        The source manager to signal to release the resources.
    gpu : GPU
        The GPU resource to release.

    """
    model._is_locked = True

    def lock_on_unlocked():
        if model._is_locked:
            unlock_gpu(resource_manager, gpu)
            model._is_locked = False

    def hook_function(function, hook_fn):
        @wraps(function)
        def run(*args, **kwargs):
            hook_fn()
            return function(*args, **kwargs)

        return run

    # pylint: disable=unnecessary-dunder-call
    _hook_fn = hook_function(
        model.__getattribute__("train_step", True), lock_on_unlocked
    )
    setattr(model, "train_step", _hook_fn)


# pylint: disable=broad-exception-raised
def _raise_or_ignore(
    exceptions: list[Exception],
    fault_tollerant: bool,
    logger: RemoteFileLogger,
    raise_exceptions: list[type],
):
    if not fault_tollerant and len(exceptions) > 1:
        raise Exception(exceptions)
    if not fault_tollerant and len(exceptions) == 1:
        raise exceptions[0]
    if not fault_tollerant:
        raise Exception("Unknown error")
    for e in exceptions:
        if isinstance(e, tuple(raise_exceptions)):
            error_msg = (
                f"Exception `{str(e)}` of type: `{type(e).__name__}` in"
                f" crash_exceptions_types={[c.__name__ for c in raise_exceptions]}."
                " Exiting."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)


# pylint: disable=broad-exception-caught
def _handle_exception(
    e: Exception,
    model: ModelWrapper,
    run_config: ParallelConfig,
    mp_logger: RemoteFileLogger,
    resource_manager: ResourceManager,
    gpu: GPU | None,
    uid: int,
    fault_tollerant: bool,
    crash_exceptions_types: list[type] | None,
) -> tuple[int, dict[str, float] | None, TrialState]:
    if crash_exceptions_types is None:
        crash_exceptions_types = []
    exceptions = [e]
    try:
        if gpu is not None:
            # gpu_manager is a ray-actor
            unlock_gpu(resource_manager, gpu)
        exception_str = traceback.format_exc()
        if hasattr(model, "logger"):
            model.logger.error(exception_str)
        else:
            mp_logger.error(f"{run_config.uid}:\n{exception_str}")
        mp_logger.error(f"Error Occured {run_config.uid}: {str(e)}")
    except Exception as _e:
        exceptions.append(_e)
    finally:
        _raise_or_ignore(
            exceptions,
            fault_tollerant=fault_tollerant,
            logger=mp_logger,
            raise_exceptions=crash_exceptions_types,
        )

    return uid, None, TrialState.FAIL


# TODO refactor as an actor and reduce complexity
# pylint: disable=broad-exception-caught,too-many-arguments,too-complex
def train_main_remote(
    model: ModelWrapper,
    run_config: ParallelConfig,
    mp_logger: RemoteFileLogger,
    resource_manager: ResourceManager | None,
    gpu: GPU | None,
    uid: int,
    fault_tollerant: bool = True,
    crash_exceptions_types: list[type] | None = None,
    resume: bool = False,
    clean_reset: bool = False,
    progress_bar: ty.Optional[RemoteProgressBar] = None,
    data_lock: ty.Optional[butils.Lock] = None,
) -> tuple[int, dict[str, float] | None, TrialState]:
    """
    The trial job will be executed remotely at a ray node. This is where model training happens.
    In addition, the experiment directory will be synchronized to the Cloud storage and remote nodes.

    Parameters
    ----------
    model : ModelWrapper
        The ModelWrapper that is used to train a model.
    run_config : ParallelConfig
        Runtime configuration for this trial.
    mp_logger : RemoteFileLogger
        The file logger that's used to log training progress.
    resource_manager : ResourceManager | None
        The resource manager that is used to release resources after the training process starts.
        When unspecified it also expects `gpu` to be `None`.
    gpu : GPU | None
        The gpu to which to allocate the execution of the current remote.
    uid : int
        the trial unique identifier.
    fault_tollerant : bool
        Whether to tollerate crashes, aka to cease execution when the ray job crashes, by default ``True``.
    crash_exceptions_types : list[type] | None
        Types of exceptions that are considered as crashes, by default ``None``.
    resume : bool
        Whether to resume training the model from existing checkpoints
        and existing experiment state, by default ``False``.
    clean_reset : bool
        Whether to remove model directory when ``CheckpointNotFoundError`` is raised, by default ``False``.
    progress_bar : ty.Optional[RemoteProgressBar]
        Optionally, we can use a remote progress bar to update the results of the trial.
    data_lock : ty.Optional[butils.Lock], optional
        Use a Lock for when building the dataloader to ensure that it does not concurrently
        download data for several processes

    Returns
    -------
    tuple[int, dict[str, float] | None, TrialState]
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

    Raises
    ------
    ValueError
        When only one of `resource_manager` or `gpu` is specified.
    """
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu.device_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if (resource_manager is not None) ^ (gpu is not None):
        raise ValueError(
            "Must specify or leave unspecified `resource_manager` and `gpu`."
        )
    handle_exception = partial(
        _handle_exception,
        model=model,
        run_config=run_config,
        mp_logger=mp_logger,
        resource_manager=resource_manager,
        gpu=gpu,
        uid=uid,
        fault_tollerant=fault_tollerant,
        crash_exceptions_types=crash_exceptions_types,
    )

    try:
        # We add a hook after the first train_step to release the resource from the resource
        # manager
        if resource_manager is not None and gpu is not None:
            _apply_unlock_hook(model, resource_manager, gpu=gpu)

        # NOTE in order for os.environ to wotk CUDA must be unitialized
        # up to this point.
        model.init_state(
            run_config,
            resume=resume,
            remote_progress_bar=progress_bar,
            data_lock=data_lock,
        )
        res = model.train()
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

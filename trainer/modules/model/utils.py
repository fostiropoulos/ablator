import logging
import typing as ty


import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from trainer.config.run import ModelConfigBase, TrainMode
from trainer.modules.optimizers import OptimizerConfig
from trainer.modules.schedulers import Scheduler, SchedulerConfig


def load_model(
    model_class: ty.Type[nn.Module],
    model_config: ModelConfigBase,
    scheduler_config,
    optimizer_config,
    save_dict: ty.Dict,
    device: str = "cpu",
    use_amp: bool = False,
    strict: bool = True,
    init_mode: TrainMode = TrainMode.vanilla,
):

    model_state = save_dict["state_dict"]
    scheduler_state = save_dict["scheduler"] if "scheduler" in save_dict else None
    optimizer_state = save_dict["optimizer"] if "optimizer" in save_dict else None
    scaler_state = save_dict["scaler"] if "scaler" in save_dict else None

    (model, optimizer, scaler, scheduler) = create_model(
        model_class=model_class,
        model_config=model_config,
        model_state=model_state,
        optimizer_config=optimizer_config,
        optimizer_state=optimizer_state,
        scheduler_config=scheduler_config,
        scheduler_state=scheduler_state,
        scaler_state=scaler_state,
        use_amp=use_amp,
        device=device,
        strict_load=strict,
        init_mode=init_mode,
    )

    return model, optimizer, scaler, scheduler, save_dict


def _init_model(
    model_class,
    model_config,
    model_state: ty.Optional[ty.Dict] = None,
    strict_load: bool = True,
    weights_init_fn: ty.Optional[ty.Callable] = None,
):
    model: torch.nn.Module = model_class(model_config)
    if model_state:
        model.load_state_dict(model_state, strict=strict_load)
    elif weights_init_fn is not None:
        model.apply(weights_init_fn)
    return model


def wrap_model_dataparallel(model: nn.Module, device_ids):
    assert isinstance(
        device_ids, list
    ), "device argument must be set to a list for dataparallel."
    model.to(device_ids[0])
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model


def wrap_model_distributed(model, device):
    assert dist.is_initialized()
    model.to(device)
    model = DDP(model, device_ids=[device])
    for _, param in model.state_dict().items():
        dist.broadcast(param, 0)
    dist.barrier()
    torch.cuda.synchronize()
    return model


def _create_optimizer(model, optimizer_config: OptimizerConfig, optimizer_state):

    optimizer: torch.optim.Optimizer = None

    if optimizer_config is not None:
        optimizer = optimizer_config.make_optimizer(model)

    if optimizer_state is not None:
        if optimizer is None:
            logging.warning(
                "Supplied `optimizer_state` without `optimizer_config`. Ignoring optimizer."
            )
            return None
        # NOTE: because https://github.com/pytorch/pytorch/issues/80809
        # TODO any good fix  for this yet?
        for k in optimizer_state["state"].keys():
            if "step" in optimizer_state["state"][k] and isinstance(
                optimizer_state["state"][k]["step"], torch.Tensor
            ):
                optimizer_state["state"][k]["step"] = optimizer_state["state"][k][
                    "step"
                ].cpu()

        optimizer.load_state_dict(optimizer_state)
    return optimizer


def _create_scheduler(
    optimizer,
    scheduler_config: ty.Optional[SchedulerConfig],
    scheduler_state: ty.Optional[ty.Dict],
) -> ty.Optional[Scheduler]:
    scheduler: ty.Optional[Scheduler] = None
    if scheduler_config is not None:
        scheduler = scheduler_config.make_scheduler(optimizer)

    if scheduler_state is not None:
        if scheduler is None:
            logging.warning(
                "Supplied `scheduler_state` without `scheduler_config`. Ignoring scheduler."
            )
            return None
        scheduler.load_state_dict(scheduler_state)
    return scheduler


def _create_scaler(use_amp: bool = False, scaler_state: ty.Optional[ty.Dict] = None):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if scaler_state:
        scaler.load_state_dict(scaler_state)
    return scaler


def create_optimizer_scheduler(
    model: nn.Module,
    optimizer_config: ty.Optional[OptimizerConfig] = None,
    optimizer_state: ty.Optional[ty.Dict] = None,
    scheduler_config: ty.Optional[SchedulerConfig] = None,
    scheduler_state: ty.Optional[ty.Dict] = None,
):
    optimizer = _create_optimizer(
        model=model, optimizer_config=optimizer_config, optimizer_state=optimizer_state
    )
    scheduler = _create_scheduler(
        optimizer=optimizer,
        scheduler_config=scheduler_config,
        scheduler_state=scheduler_state,
    )
    return optimizer, scheduler


def create_model(
    model_class: ty.Type,
    model_config: ModelConfigBase = None,
    model_state: ty.Optional[ty.Dict] = None,
    optimizer_config: ty.Optional[OptimizerConfig] = None,
    optimizer_state: ty.Optional[ty.Dict] = None,
    scheduler_config: ty.Optional[SchedulerConfig] = None,
    scheduler_state: ty.Optional[ty.Dict] = None,
    scaler_state: ty.Optional[ty.Dict] = None,
    use_amp: bool = False,
    device: ty.Union[str, int, ty.List[int]] = "cpu",
    strict_load: bool = True,
    weights_init_fn: ty.Optional[ty.Callable] = None,
    init_mode: TrainMode = TrainMode.vanilla,
):

    model = _init_model(
        model_class=model_class,
        model_config=model_config,
        model_state=model_state,
        strict_load=strict_load,
        weights_init_fn=weights_init_fn,
    )
    if init_mode == TrainMode.data_parallel:
        model = wrap_model_dataparallel(model, device)
    elif init_mode == TrainMode.dist_data_parallel:
        model = wrap_model_distributed(model, device)
    elif init_mode == TrainMode.vanilla:
        model = model.to(device)
    else:
        raise NotImplementedError
    optimizer, scheduler = create_optimizer_scheduler(
        model=model,
        optimizer_config=optimizer_config,
        optimizer_state=optimizer_state,
        scheduler_config=scheduler_config,
        scheduler_state=scheduler_state,
    )
    scaler = _create_scaler(use_amp=use_amp, scaler_state=scaler_state)
    return model, optimizer, scaler, scheduler


def save_checkpoint(state, filename="checkpoint.pt"):
    torch.save(state, filename)

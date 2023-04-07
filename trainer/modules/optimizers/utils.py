import torch
import typing as ty


def get_parameter_names(
    model: torch.nn.Module, forbidden_layer_types: ty.List[ty.Type]
):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def get_optim_parameters(
    model: torch.nn.Module,
    weight_decay: ty.Optional[float] = None,
    only_requires_grad: bool = True,
):
    """
    Setup the optimizer.
    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
    """
    # default_val = lambda k, v: kwargs[k] if k in kwargs else v

    params_to_update = {}
    if only_requires_grad:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update[name] = param
    else:
        params_to_update = model.named_parameters()
    if weight_decay is not None:
        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
        decay_parameters = [
            name
            for name in decay_parameters
            if "bias" not in name and name in params_to_update
        ]
        optimization_params = [
            {
                "params": [
                    p for n, p in params_to_update.items() if n in decay_parameters
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in params_to_update.items() if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimization_params
    return list(params_to_update.values())

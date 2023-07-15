import copy
import typing as ty
from collections import OrderedDict

import numpy as np

from ablator.config.mp import Optim
from ablator.utils.file import nested_set


def _verify_metrics(metrics: dict[str, float] | None):
    if metrics is None:
        return
    for k, v in metrics.items():
        if not isinstance(k, str):
            raise ValueError(
                f"Invalid metrics dictionary key ({type(k)}) {k}, expected `str`"
            )
        if not isinstance(v, (int, float)):
            raise ValueError(
                f"Invalid metrics value ({type(v)}) {v} for {k}, expected `int | float`"
            )


def augment_trial_kwargs(
    trial_kwargs: dict[str, ty.Any], augmentation: dict[str, ty.Any]
) -> dict[str, ty.Any]:
    """
    Augment the ``trial_kwargs`` with additional key-value pairs specified in the augmentation dictionary.

    Parameters
    ----------
    trial_kwargs : dict
        The dictionary containing the key-value pairs to be augmented.
    augmentation : dict
        The dictionary containing the additional key-value pairs.

    Returns
    -------
    dict
        The augmented dictionary.

    Examples
    --------
    >>> trial_kwargs = {'a': 1, 'b': 2}
    >>> augmentation = {'c': 3, 'd.e': 4}
    >>> augment_trial_kwargs(trial_kwargs, augmentation)
    {'a': 1, 'b': 2, 'c': 3, 'd': {'e': 4}}
    """
    trial_kwargs = copy.deepcopy(trial_kwargs)
    config_dot_path: str
    dot_paths = list(augmentation.keys())

    assert len(set(dot_paths)) == len(
        dot_paths
    ), f"Duplicate tune paths: {set(dot_paths).difference(dot_paths)}"
    for config_dot_path, val in augmentation.items():
        path: list[str] = config_dot_path.split(".")
        trial_kwargs = nested_set(trial_kwargs, path, val)
    return trial_kwargs


def _parse_metrics(
    metric_directions: dict[str, Optim], metrics: dict[str, float] | None
) -> dict[str, float] | None:
    """
    Convert metrics to ordered dictionary of float values and use their direction (minimize or maximize)
    if they are missing or are invalid to set to inf and -inf respectively. Returns the subet of metrics
    present in metric_directions

    Parameters
    ----------
    metric_directions : dict
        The ordered dictionary containing the directions of the metrics (minimize or maximize).
    metrics : dict
        The dictionary containing the metric values.

    Returns
    -------
    OrderedDict
        The ordered dictionary of metric values converted to float using their direction.

    Examples
    --------
    >>> metric_directions = OrderedDict([('a', 'max'), ('b', 'min')])
    >>> metrics = {'a': 1, 'b': None}
    >>> _parse_metrics(metric_directions, metrics)
    OrderedDict([('a', 1.0), ('b', inf)])
    """
    if metrics is None:
        return None
    vals = OrderedDict()

    for k in sorted(metric_directions):
        if k not in metrics:
            raise KeyError(
                f"Expected to find {k} in returned model metrics. Instead found: {set(metrics.keys())}"
            )
        v = metric_directions[k]
        val = metrics[k]
        if val is None or not np.isfinite(val):
            val = float("-inf") if Optim(v) == Optim.max else float("inf")
        vals[k] = val
    return vals

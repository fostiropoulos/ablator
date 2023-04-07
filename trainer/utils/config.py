import collections.abc
import copy
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf


def nested_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = nested_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def read_config(default_config: Path, *aug_configs: Path):
    """
    TODO Possibly refactor to config
    """
    def open_config(p: Path):
        config = OmegaConf.load(p.as_posix())
        return config

    augmenting_config = open_config(default_config)
    for aug_config in aug_configs:
        augmenting_config.merge_with(open_config(aug_config))
    return OmegaConf.to_object(augmenting_config)


def flatten_nested_dict(_dict, expand_list=True, seperator=".") -> Dict[str, Any]:
    """
    TODO Possibly refactor to config
    """
    flatten_dict = copy.deepcopy(_dict)
    for k, v in _dict.items():
        _gen = None
        if isinstance(v, dict):
            _gen = v.items()

        if isinstance(v, (list, tuple)) and expand_list:
            _gen = enumerate(v)

        if _gen is not None:
            del flatten_dict[k]
            for _k, _v in _gen:
                flatten_dict[f"{k}{seperator}{_k}"] = _v

    if len(flatten_dict) != len(_dict):
        return flatten_nested_dict(flatten_dict)
    return flatten_dict

def nested_get(_dict: Dict, keys: List[str]) -> Any:
    x = copy.deepcopy(_dict)
    for key in keys:
        x = x[key]
    return x


def nested_set(_dict, keys: List[str], value: Any):
    original_dict = copy.deepcopy(_dict)
    x = original_dict
    for key in keys[:-1]:
        if key not in x:
            x[key] = {}
        x = x[key]  # .setdefault(key, {})
    x[keys[-1]] = value
    return original_dict


import copy
import json
import typing as ty
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def make_sub_dirs(parent: str | Path, *dir_names) -> list[Path]:
    dirs = []
    for dir_name in dir_names:
        dir_path = Path(parent).joinpath(dir_name)
        dir_path.mkdir(parents=True, exist_ok=True)
        dirs.append(dir_path)
    return dirs


def save_checkpoint(state, filename="checkpoint.pt"):
    torch.save(state, filename)


def clean_checkpoints(checkpoint_folder: Path, n_checkpoints: int):
    chkpts = sorted(list(checkpoint_folder.glob("*.pt")))[::-1]

    # Keep only last n checkpoints (or first n because we sort in reverse)
    if len(chkpts) > n_checkpoints:
        chkpts_to_del = chkpts[n_checkpoints:]
        for _chkpt in chkpts_to_del:
            Path(_chkpt).unlink(missing_ok=True)


def default_val_parser(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, torch.Tensor):
        return default_val_parser(val.detach().cpu().numpy())
    if isinstance(val, pd.DataFrame):
        return default_val_parser(np.array(val))
    return str(val)


def json_to_dict(_json):
    _dict = json.loads(_json)
    return _dict


def dict_to_json(_dict):
    _json = json.dumps(_dict, indent=0, default=default_val_parser)
    # make sure it can be decoded
    json_to_dict(_json)
    return _json


def nested_set(_dict, keys: list[str], value: ty.Any):
    original_dict = copy.deepcopy(_dict)
    x = original_dict
    for key in keys[:-1]:
        if key not in x:
            x[key] = {}
        x = x[key]
    x[keys[-1]] = value
    return original_dict

import glob
import json
from os.path import join
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch


def make_sub_dirs(parent: Union[str, Path], *dir_names) -> list[Path]:
    dirs = []
    for dir_name in dir_names:
        dir_path = Path(parent).joinpath(dir_name)
        dir_path.mkdir(parents=True, exist_ok=True)
        dirs.append(dir_path)
    return dirs


def save_checkpoint(state, filename="checkpoint.pt"):
    torch.save(state, filename)


def clean_checkpoints(checkpoint_folder, n_checkpoints):
    chkpts = sorted(glob.glob(join(checkpoint_folder, "*.pt")))[::-1]

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

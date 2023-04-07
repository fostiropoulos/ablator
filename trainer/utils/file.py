import glob
import hashlib
import copy
import json
import os
from os.path import join
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union
import numpy as np
import pandas as pd
import subprocess
import signal
import traceback
from omegaconf import OmegaConf
import torch

from trainer.utils.config import flatten_nested_dict


def make_sub_dirs(parent: Union[str, Path], *dir_names) -> List[Path]:
    dirs = []
    for dir_name in dir_names:
        dir_path = Path(parent).joinpath(dir_name)
        dir_path.mkdir(parents=True, exist_ok=True)
        dirs.append(dir_path)
    return dirs


def run_cmd_wait(cmd, timeout=300, raise_errors=False):
    # timeout is in seconds
    with subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, preexec_fn=os.setsid
    ) as process:
        try:
            output = process.communicate(timeout=timeout)[0]
        except subprocess.TimeoutExpired as e:
            os.killpg(process.pid, signal.SIGINT)  # send signal to the process group
            output = process.communicate()[0]
            traceback.print_exc()
            if raise_errors:
                raise e


def default_val_parser(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, torch.Tensor):
        return default_val_parser(val.detach().cpu().numpy())
    if isinstance(val, pd.DataFrame):
        return default_val_parser(np.array(val))
    return str(val)


def dict_to_json(_dict):
    _json = json.dumps(_dict, indent=0, default=default_val_parser)
    # make sure it can be decoded
    json_to_dict(_json)
    return _json



def write_yaml(_dict, p, clean_dict=False):
    # TODO Refactor into config
    if clean_dict:
        _dict = json_to_dict(dict_to_json(_dict))
    with open(p, "w") as fp:
        fp.write(OmegaConf.to_yaml(OmegaConf.create(_dict)))


def json_to_dict(_json):
    _dict = json.loads(_json)
    return _dict


def dict_hash(*dictionaries: List[Dict[str, Any]], hash_len=4):
    """MD5 hash of a dictionary."""
    concat_dictionaries = [
        copy.deepcopy(_) if isinstance(_, dict) else copy.deepcopy(_).__dict__
        for _ in dictionaries
    ]
    dictionary = reduce(lambda a, b: {**a, **b}, concat_dictionaries)
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    dictionary = flatten_nested_dict(dictionary)
    _dict = {}
    for k, v in dictionary.items():
        if not isinstance(v, (bool, str, int, float, type(None))):
            v = getattr(v, "__name__", str(v))
        _dict[k] = v

    encoded = json.dumps(_dict, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()[:hash_len]


def get_latest_chkpts(checkpoint_dir) -> Sequence[str]:
    checkpoint_dir = join(checkpoint_dir, "*.pt")
    chkpts = glob.glob(checkpoint_dir)
    return sorted(chkpts)[::-1]


def clean_checkpoints(checkpoint_folder, n_checkpoints):
    chkpts = sorted(glob.glob(join(checkpoint_folder, "*.pt")))[::-1]

    # Keep only last n checkpoints (or first n because we sort in reverse)
    if len(chkpts) > n_checkpoints:
        chkpts_to_del = chkpts[n_checkpoints:]
        for _chkpt in chkpts_to_del:
            Path(_chkpt).unlink(missing_ok=True)

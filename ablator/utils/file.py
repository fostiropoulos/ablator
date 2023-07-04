import copy
import json
import os
import typing as ty
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def make_sub_dirs(parent: str | Path, *dir_names) -> list[Path]:
    """
    Create subdirectories under the given parent directory.

    Parameters
    ----------
    parent : str | Path
        Parent directory where subdirectories should be created.
    *dir_names : str
        Names of the subdirectories to create.

    Returns
    -------
    list[Path]
        A list of created subdirectory paths.
    """
    dirs = []
    for dir_name in dir_names:
        dir_path = Path(parent).joinpath(dir_name)
        dir_path.mkdir(parents=True, exist_ok=True)
        dirs.append(dir_path)
    return dirs


def save_checkpoint(state, filename="checkpoint.pt"):
    """
    Save a checkpoint of the given state.

    Parameters
    ----------
    state : dict
        Model State dictionary to save.
    filename : str, optional
        The name of the checkpoint file, by default "checkpoint.pt".
    """
    torch.save(state, filename)


def clean_checkpoints(checkpoint_folder: Path, n_checkpoints: int):
    """
    Remove all but the n latest checkpoints from the given directory.

    Parameters
    ----------
    checkpoint_folder : Path
        Directory containing the checkpoint files.
    n_checkpoints : int
        Number of checkpoints to keep.
    """
    chkpts = sorted(list(checkpoint_folder.glob("*.pt")))[::-1]

    # Keep only last n checkpoints (or first n because we sort in reverse)
    if len(chkpts) > n_checkpoints:
        chkpts_to_del = chkpts[n_checkpoints:]
        for _chkpt in chkpts_to_del:
            Path(_chkpt).unlink(missing_ok=True)


def default_val_parser(val):
    """Converts the input value to a JSON compatible format.

    Parameters
    ----------
    val : ty.Any
        The value to be converted.

    Returns
    -------
    ty.Any
        The converted value.
    """

    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, torch.Tensor):
        return default_val_parser(val.detach().cpu().numpy())
    if isinstance(val, pd.DataFrame):
        return default_val_parser(np.array(val))
    return str(val)


def json_to_dict(json_):
    """
    Convert a JSON string into a dictionary.

    Parameters
    ----------
    json_ : str
        JSON string to be converted.

    Returns
    -------
    dict
        A dictionary representation of the JSON string.
    """

    dict_ = json.loads(json_)
    return dict_


def dict_to_json(dict_):
    """
    Convert a dictionary into a JSON string.

    Parameters
    ----------
    dict_ : dict
        The dictionary to be converted.

    Returns
    -------
    str
        The JSON string representation of the dictionary.
    """
    _json = json.dumps(dict_, indent=0, default=default_val_parser)
    # make sure it can be decoded
    json_to_dict(_json)
    return _json


def nested_set(dict_, keys: list[str], value: ty.Any):
    """
    Set a value in a nested dictionary.

    Parameters
    ----------
    dict_ : dict
        The dictionary to update.
    keys : list[str]
        List of keys representing the nested path.
    value : ty.Any
        The value need to set at the specified path.

    Examples
    --------
    >>> dict_ = {'a': {'b': {'c': 1}}}
    >>> nested_set(dict_, ['a', 'b', 'c'], 2)
    >>> dict_
    {'a': {'b': {'c': 2}}}

    Returns
    -------
    dict
        The updated dictionary with the new value set.
    """
    original_dict = copy.deepcopy(dict_)
    x = original_dict
    for key in keys[:-1]:
        if key not in x:
            x[key] = {}
        x = x[key]
    x[keys[-1]] = value
    return original_dict


def truncate_utf8_chars(filename: Path, last_char: str):
    assert (
        len(last_char) == 1
    ), f"Can not truncate up to a single character. `last_char`: {last_char}"
    last_char_ord = ord(last_char)
    with open(filename, "rb+") as f:
        size = os.fstat(f.fileno()).st_size
        offset = 1
        while offset <= size:
            f.seek(-offset, os.SEEK_END)
            if ord(f.read(1)) == last_char_ord:
                f.seek(-1, os.SEEK_CUR)
                f.truncate()
                return
            offset += 1
        raise RuntimeError(
            f"Could not truncate {filename} since `last_char`: {last_char} was not found in the file."
        )

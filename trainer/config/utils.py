import copy
import hashlib
import json
from functools import reduce
from typing import Any, Dict, List

"""Flatten nested dictionary,list,tuple

Parameters
----------
_dict : Dict[str, Any]
    dict to be flattened
expand_list : bool, optional
    whether to expand list or not, by default True
seperator : str, optional
    seperator to use for flattening, by default "."

Returns
-------
Dict[str, Any]
    flattened dictionary

"""
def flatten_nested_dict(_dict, expand_list=True, seperator=".") -> Dict[str, Any]:
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

"""MD5 hash of a dictionary.

"""

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

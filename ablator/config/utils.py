from collections import abc
import copy
import hashlib
import json
from functools import reduce
import typing as ty


def flatten_nested_dict(
    dict_: dict, expand_list=True, seperator="."
) -> dict[str, ty.Any]:
    """
    Flattens a nested dictionary, expanding lists and tuples if specified.

    Parameters
    ----------
    dict_ : dict
        The input dictionary to be flattened.
    expand_list : bool, optional
        Whether to expand lists and tuples in the dictionary, by default ``True``.
    seperator : str, optional
        The separator used for joining the keys, by default ``"."``.

    Returns
    -------
    dict[str, ty.Any]
        The flattened dictionary.

    Examples
    --------
    >>> nested_dict = {"a": {"b": 1, "c": {"d": 2}}, "e": [3, 4]}
    >>> flatten_nested_dict(nested_dict)
    {'a.b': 1, 'a.c.d': 2, 'e.0': 3, 'e.1': 4}
    """
    flatten_dict = copy.deepcopy(dict_)
    for k, v in dict_.items():
        _gen: ty.Optional[abc.Iterable] = None
        if isinstance(v, dict):
            _gen = v.items()

        if isinstance(v, (list, tuple)) and expand_list:
            _gen = enumerate(v)

        if _gen is not None:
            del flatten_dict[k]
            for _k, _v in _gen:
                flatten_dict[f"{k}{seperator}{_k}"] = _v

    if len(flatten_dict) != len(dict_):
        return flatten_nested_dict(flatten_dict)
    return flatten_dict


def dict_hash(*dictionaries: list[dict[str, ty.Any]], hash_len=4):
    """
    Calculates the MD5 hash of one or more dictionaries.

    Parameters
    ----------
    *dictionaries : list[dict[str, ty.Any]]
        One or more dictionaries to calculate the hash for.
    hash_len : int, optional
        The length of the hash to return, by default 4.

    Returns
    -------
    str
        The MD5 hash of the dictionaries.

    Examples
    --------
    >>> dict1 = {"a": 1, "b": 2}
    >>> dict2 = {"c": 3, "d": 4}
    >>> dict_hash(dict1, dict2)
    '6d75e6'
    """
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

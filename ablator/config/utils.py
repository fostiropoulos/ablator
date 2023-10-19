import ast
import copy
import hashlib
import json
import typing as ty
from collections import abc
from functools import reduce


def flatten_nested_dict(
    dict_: dict, expand_list: bool = True, seperator: str = "."
) -> dict[str, ty.Any]:
    """
    Flattens a nested dictionary, expanding lists and tuples if specified.

    Parameters
    ----------
    dict_ : dict
        The input dictionary to be flattened.
    expand_list : bool
        Whether to expand lists and tuples in the dictionary, by default ``True``.
    seperator : str
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
        return flatten_nested_dict(flatten_dict, expand_list, seperator)
    return flatten_dict


def dict_hash(
    *dictionaries: list[dict[str, ty.Any]] | dict[str, ty.Any], hash_len: int = 4
) -> str:
    """
    Calculates the MD5 hash of one or more dictionaries.

    Parameters
    ----------
    *dictionaries : list[dict[str, ty.Any]] | dict[str, ty.Any]
        One or more dictionaries to calculate the hash for.
    hash_len : int
        The length of the hash to return, by default ``4``.

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


# pylint: disable=bare-except
# flake8: noqa: E722
def _parse_fn_repr(val, fn_name):
    try:
        kwargs = getattr(val, fn_name)
        if isinstance(kwargs, abc.Callable):
            type(val)(**kwargs())
            assert getattr(type(val)(**kwargs()), fn_name)() == kwargs()
            return kwargs()
        assert getattr(type(val)(**kwargs), fn_name) == kwargs
        return kwargs
    except:
        return None


def _parse_ast_repr(str_repr):
    parsed = ast.parse(str_repr, mode="eval")

    # Extract the function call node from the AST
    func_call_node = parsed.body

    # Ensure that the node is actually a function call
    if not isinstance(func_call_node, ast.Call):
        raise ValueError("Input is not a valid function call")

    # Extract the arguments from the function call node
    args = tuple(ast.literal_eval(arg) for arg in func_call_node.args)
    kwargs = {
        str(arg.arg): ast.literal_eval(arg.value) for arg in func_call_node.keywords
    }
    return args, kwargs


# pylint: disable=bare-except,unnecessary-dunder-call
# flake8: noqa: E722
def parse_repr_to_kwargs(
    obj: ty.Any,
) -> tuple[tuple, dict[str, int | float | str | bool | None]]:
    """
    parse a string or dictionary representation to obtain the initialization arguments
    of the same object. It first attempts to do that via user-implemented `to_dict`,
    `as_dict` and `__dict__` methods and when it fails it results to evaluating the
    string representation e.g. `eval(str(obj))`. If all fails... it raises an error.

    NOTE the object `obj` must have the equality operator implemented `__eq__`, ideally
    a user implemented `to_dict`.

    Parameters
    ----------
    obj : ty.Any
        The object to deconstruct.

    Returns
    -------
    tuple[tuple, dict[str, int | float | str | bool | None]]
        a tuple of (args, kwargs) to reconstruct `obj` from above.

    Raises
    ------
    RuntimeError
        is raised when it is unable to obtain a representation that can
        reconstruct the original object. The reconstruction is evaluated by
        the equality operator.
    """
    for fn_name in ("to_dict", "as_dict", "__dict__"):
        if (kwargs := _parse_fn_repr(obj, fn_name)) is not None:
            return (), kwargs

    try:
        str_repr = obj.__repr__()
    except:
        str_repr = str(obj)
    try:
        args, kwargs = _parse_ast_repr(str_repr)
        _kwargs = copy.deepcopy(kwargs)
        _args = copy.deepcopy(args)
        _args, _kwargs = _parse_ast_repr(type(obj)(*_args, **_kwargs).__repr__())
        assert args == _args and kwargs == _kwargs
        return args, kwargs
    except:
        pass
    raise RuntimeError(
        f"Could not parse {type(obj)} from its representation `{str_repr}`. Please make"
        " sure that one of `to_dict`, `as_dict`, `__dict__`, `__repr__`  is correctly"
        " implemented (evaluated in the same order) and the object can be"
        " reconstructed e.g. `eval(value.__repr__())==value` or"
        " `type(value)(**value.to_dict())==value`"
    )

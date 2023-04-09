import typing as ty
from collections import namedtuple
from enum import Enum as _Enum

T = ty.TypeVar("T")


class Dict(ty.Dict[str, T]):
    pass


class List(ty.List[T]):
    pass


class Tuple(ty.Tuple[T]):
    pass


class Optional(ty.Generic[T]):
    pass


Type = type
Literal = ty.Literal


class Enum(_Enum):
    def __eq__(self, __o: object) -> bool:
        val = __o
        if not isinstance(val, type(self)):
            val = type(self)(val)
        return super().__eq__(val)

    def __hash__(self):
        return _Enum.__hash__(self)


# ALLOWED_COLLECTIONS is meant to only support collections that can be
# expressed in a yaml file in an unambigious manner and using primitives
# int, float and str (ALLOWED_TYPES)
# Extending the type system should be intuitive.
# Each annotation is of the format
# "STATE", "OPTIONAL", "COLLECTION", "TYPE"

ALLOWED_TYPES = (int, float, str,bool, None)

ALLOWED_COLLECTIONS = (
    None,
    List,
    Dict,
    Tuple,
    Type,
    Enum,
    Literal,
)
Annotation = namedtuple(
    "Annotation", ["state", "optional", "collection", "variable_type"]
)


doc_type_hint_structure = f"""
[Derived,Stateless, None][Optional,None][{ALLOWED_COLLECTIONS}][{ALLOWED_TYPES}*]

Only Tuple allows non-homogenous types (must be of fixed length)
For more flexibility you can define another "Type" which is a class
and supply a dictionary in the yaml file, i.e.

class MyClass:
    def __init__(self, arg1:int, arg2:str):
        pass

TODO
Parsing the dictionary however can be error prone if you have complex arguments
and is not advised.
"""


def _strip_hint_state(type_hint):
    origin = ty.get_origin(type_hint)
    if origin is None:
        return Stateful, type_hint
    if origin in [Derived, Stateless]:
        assert len(type_hint.__args__) == 1
        return origin, type_hint.__args__[0]

    return Stateful, type_hint


def _strip_hint_optional(type_hint):
    origin = ty.get_origin(type_hint)
    if origin == Optional:  # ty.Union and type_hint._name == "Optional":
        args = ty.get_args(type_hint)
        assert len(args) == 1
        return True, args[0]
    return False, type_hint


def _strip_hint_collection(type_hint):
    origin = ty.get_origin(type_hint)
    assert (
        origin in ALLOWED_COLLECTIONS
    ), f"Invalid collection {origin}. type_hints must be structured as:"
    if origin is None and type_hint in ALLOWED_TYPES:
        return None, type_hint
    if origin in [Dict, List]:
        args = ty.get_args(type_hint)
        assert len(args) == 1
        # Dict and list annotations only support a single type
        assert args[0] in ALLOWED_TYPES, f"Invalid type_hint: {type_hint}."
        collection = Dict if origin == Dict else List
        return collection, args[0]
    if origin == Tuple:
        args = ty.get_args(type_hint)
        # if the user requires support for multiple types they should use tuple
        return Tuple, args

    if origin is Literal:
        return Literal, type_hint.__args__
    if issubclass(type_hint, Enum):
        valid_values = [_v.value for _v in list(type_hint)]
        return type_hint, valid_values
    if isinstance(type(type_hint), Type) and hasattr(type_hint, "__dict__"):
        assert origin is None
        return Type, type_hint
    raise NotImplementedError(f"{type_hint} is not a valid hint. Custom classes must implement __dict__.")


def parse_type_hint(type_hint):
    state, _type_hint = _strip_hint_state(type_hint)

    optional, _type_hint = _strip_hint_optional(_type_hint)

    collection, variable_type = _strip_hint_collection(_type_hint)

    return Annotation(
        state=state,
        optional=optional,
        collection=collection,
        variable_type=variable_type,
    )


def _parse_class(cls, kwargs):
    if isinstance(kwargs, cls):
        # This is when initializing directly from config
        pass
    elif isinstance(kwargs, dict):
        # This is when initializing from a dictionary
        # TODO or not, is to assert that kwargs is composed of primitives?
        kwargs = cls(**kwargs)
    else:
        # not sure what to do.....
        raise RuntimeError(f"Incompatible kwargs {type(kwargs)}: {kwargs}\nand {cls}.")
    return kwargs


def parse_value(val, annot: Annotation, name=None):
    # annot = parse_type_hint(type_hint)
    if val is None:
        if not (annot.state in [Derived, Stateless] or annot.optional):
            raise RuntimeError(f"Missing required value for {name}.")
        return None
    if annot.collection is Literal:
        assert (
            val in annot.variable_type
        ), f"{val} is not a valid Literal {annot.variable_type}"
        return val
    if annot.collection == Dict:
        return {str(_k): annot.variable_type(_v) for _k, _v in val.items()}
    if annot.collection == List:
        return [annot.variable_type(_v) for _v in val]
    if annot.collection == Tuple:
        assert len(val) == len(
            annot.variable_type
        ), f"Incompatible lengths for {name} between {val} and type_hint: {annot.variable_type}"
        return [tp(_v) for tp, _v in zip(annot.variable_type, val)]
    if annot.collection == Type:
        return _parse_class(annot.variable_type, val)
    if annot.collection is None:
        return annot.variable_type(val)
    if issubclass(annot.collection, Enum):
        assert (
            val in annot.variable_type
        ), f"{val} is not supported by {annot.collection}"
        return annot.collection(val)
    raise NotImplementedError


def get_annotation_state(annotation):
    origin = ty.get_origin(annotation)
    if origin is None:
        return Stateful
    if origin in [Derived, Stateless]:
        return annotation

    return Stateful


class Stateful(ty.Generic[T]):
    """
    This is for attributes that are fixed between experiments. By default
    all type_hints are stateful. Do not need to use.
    """


class Derived(ty.Generic[T]):
    """
    This type is for attributes are derived during the experiment.
    """


class Stateless(ty.Generic[T]):
    """
    This type is for attributes that can take different value assignments
    between experiments
    """

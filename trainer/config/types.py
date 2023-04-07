from collections import namedtuple
import operator
import typing as ty
from enum import Enum as _Enum

T = ty.TypeVar("T")
List = ty.List
Tuple = ty.Tuple
Type = ty.Type
Literal = ty.Literal
Optional = ty.Optional


class Dict(ty.Dict[str, T]):
    pass


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

ALLOWED_TYPES = (int, float, str, None)

ALLOWED_COLLECTIONS = (
    None,
    List,
    list,
    Dict,
    Tuple,
    Type,
    Enum,
    Literal,
)
Annotation = namedtuple(
    "Annotation", ["state", "optional", "collection", "variable_type"]
)


def get_value_type(annotation):
    origin = ty.get_origin(annotation)
    assert origin is None
    # primitive type
    assert annotation in ALLOWED_TYPES
    return {"type": annotation}


def strip_hint_state(type_hint):
    origin = ty.get_origin(type_hint)
    if origin is None:
        return Stateful, type_hint
    if origin in [Derived, Stateless]:
        assert len(type_hint.__args__) == 1
        return type_hint, type_hint.__args__[0]

    else:
        return Stateful, type_hint


def strip_hint_optional(type_hint):

    origin = ty.get_origin(type_hint)
    if origin == ty.Union and type_hint._name == "Optional":
        args = ty.get_args(type_hint)
        assert len(args) == 2
        return True, args[0]
    return False, type_hint


def strip_hint_collection(type_hint):
    origin = ty.get_origin(type_hint)
    assert origin in ALLOWED_COLLECTIONS, f"Invalid collection {origin}"
    if origin is None:
        return type_hint
    elif origin in [Dict, list]:
        args = ty.get_args(type_hint)
        assert len(args) == 1
        # Dict and list annotations only support a single type
        assert args[0] in ALLOWED_TYPES, f"Invalid type_hint: {type_hint}."
        collection = Dict if origin == Dict else List
        return collection, args[0]
    elif isinstance(origin, Tuple):
        args = ty.get_args(type_hint)
        # if the user requires support for multiple types they should use tuple
        return Tuple, args
    elif issubclass(origin, Type):
        return Type, None
    elif origin == Literal:
        return Literal, type_hint.__args__
    elif issubclass(origin, Enum):
        valid_values = [_v.value for _v in list(origin)]
        return origin, valid_values
    else:
        raise NotImplementedError


def parse_type_hint(type_hint):

    state, type_hint = strip_hint_state(type_hint)

    optional, type_hint = strip_hint_optional(type_hint)

    collection, variable_type = strip_hint_collection(type_hint)

    return Annotation(
        state=state,
        optional=optional,
        collection=collection,
        variable_type=variable_type,
    )

def parse_value(val, type_hint):
    stripped_type_hint = parse_type_hint(type_hint)


def get_annotation_state(annotation):
    origin = ty.get_origin(annotation)
    if origin is None:
        # assert (
        #     annotation in ALLOWED_TYPE_ARGS
        # ), f"{annotation} is not in the allowed types."
        return Stateful
    if origin in [Derived, Stateless]:
        return annotation

    else:
        return Stateful

    # args = ty.get_args(annotation)
    # assert len(args) == 2
    # val = cls._parse_value_types(args[0])
    # val.required = False
    # return val


# class _Annotation(ty._GenericAlias):
#     """Runtime representation of an annotated type.

#     The class is derived from `ty._AnnotatedAlias` (python 3.10), with some differences.

#     _Annotation is an alias for the type 't'. The alias behaves like a normal typing alias,
#     instantiating is the same as instantiating the underlying type, binding
#     it to types is also the same.
#     """

#     def __init__(self, origin, metadata):
#         super().__init__(origin)
#         self.__metadata__ = metadata

#     def copy_with(self, params):
#         assert len(params) == 1
#         new_type = params[0]
#         return _Annotation(new_type, self.__metadata__)

#     def __repr__(self):
#         return "{}[{}]".format(self.__metadata__, ty._type_repr(self.__origin__))

#     def __reduce__(self):
#         return operator.getitem, (_Annotation, (self.__origin__,) + self.__metadata__)

#     def __eq__(self, other):
#         if not isinstance(other, type(self)):
#             return NotImplemented
#         return (
#             self.__origin__ == other.__origin__
#             and self.__metadata__ == other.__metadata__
#         )

#     def __hash__(self):
#         return hash((self.__origin__, self.__metadata__))

#     def __getattr__(self, attr):
#         if attr in {"__name__", "__qualname__"}:
#             return type(self).__name__
#         return super().__getattr__(attr)


class Missing:
    """
    This type is for computing the diff ONLY for attributes that are missing
    """

    pass


class Stateful:
    """
    TODO is not derived
    """

    pass


class Derived(ty.Generic[T]):
    """
    This type is for attributes are derived during the experiment.
    """

    # __slots__ = ()

    # def __new__(cls, *args, **kwargs):
    #     raise TypeError("Type Derived cannot be instantiated.")

    # @ty._tp_cache
    # def __class_getitem__(cls, params):
    #     msg = "Derived[t]: t must be a type."
    #     origin = ty._type_check(params, msg, allow_special_forms=True)
    #     return _Annotation(origin, cls)

    # def __init_subclass__(cls, *args, **kwargs):
    #     raise TypeError(
    #         "Cannot subclass {}.Derived".format(cls.__module__)
    #     )


class Stateless:
    """
    This type is for attributes that can change between different runs of the same experiment.
    For example changing verbosity.
    """

    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        raise TypeError("Type Stateless cannot be instantiated.")

    @ty._tp_cache
    def __class_getitem__(cls, params):
        msg = "Stateless[t]: t must be a type."
        origin = ty._type_check(params, msg, allow_special_forms=True)

        return _Annotation(origin, cls)

    def __init_subclass__(cls, *args, **kwargs):
        raise TypeError("Cannot subclass {}.Stateless".format(cls.__module__))

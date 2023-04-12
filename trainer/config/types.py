import typing as ty
from collections import namedtuple
from enum import Enum as _Enum

T = ty.TypeVar("T")


class Dict(ty.Dict[str, T]):
    """
    A custom Dict class that extends Python's typing.Dict, providing a more concise
    and convenient way to represent dictionaries with string keys and values of a
    generic type T.

    Example usage: Dict[int] represents a dictionary with string keys and integer values.
    """
    pass


class List(ty.List[T]):
    """
    A custom List class that extends Python's typing.List, providing a more concise
    and convenient way to represent lists with values of a generic type T.

    Example usage: List[int] represents a list of integers.
    """
    pass


class Tuple(ty.Tuple[T]):
    """
    A custom Tuple class that extends Python's typing.Tuple, providing a more concise
    and convenient way to represent tuples with values of a generic type T.

    Example usage: Tuple[int, str] represents a tuple with an integer as the first element
    and a string as the second element.
    """
    pass


class Optional(ty.Generic[T]):
    """
    A custom Optional class that extends Python's typing.Generic, providing a more concise
    and convenient way to represent optional values of a generic type T.

    Example usage: Optional[int] represents a value that can be an integer or None.
    """
    pass


Type = type
Literal = ty.Literal


class Enum(_Enum):
    """
    A custom Enum class that extends Python's Enum class with additional functionality
    for equality comparisons and hashing.

    This Enum class makes it easier to compare instances of the Enum with other values
    and ensures that instances of the Enum can be used as keys in dictionaries or sets.

    Methods
    -------
    __eq__(self, __o: object) -> bool:
        Compares the current Enum instance with another object for equality.

    __hash__(self) -> int:
        Returns the hash value of the current Enum instance.
    """
    def __eq__(self, __o: object) -> bool:
        """
        Compares the current Enum instance with another object for equality.

        If the other object is not an instance of the same Enum class, this method
        attempts to convert the other object to the same Enum class before performing
        the equality comparison.

        Parameters
        ----------
        __o : object
            The other object to compare for equality with the current Enum instance.

        Returns
        -------
        bool
            True if the Enum instances are equal, False otherwise.
        """
        val = __o
        if not isinstance(val, type(self)):
            val = type(self)(val)
        return super().__eq__(val)

    def __hash__(self):
        """
        Returns the hash value of the current Enum instance.

        This method allows Enum instances to be used as keys in dictionaries or sets.

        Returns
        -------
        int
            The hash value of the current Enum instance.
        """
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
    """
    Extracts the state (Stateful, Derived, or Stateless) from the given type hint.

    This function checks if the provided type hint has a state (Stateful, Derived, or Stateless)
    and returns a tuple containing the state and the underlying type hint. If the type hint has
    no state specified, it defaults to Stateful.

    Parameters
    ----------
    type_hint : Type
        The type hint to be analyzed for its state.

    Returns
    -------
    tuple
        A tuple containing:
        - state (Type): The state of the type hint (Stateful, Derived, or Stateless).
        - type_hint (Type): The underlying type hint without the state information.
    """
    origin = ty.get_origin(type_hint)
    if origin is None:
        return Stateful, type_hint
    if origin in [Derived, Stateless]:
        assert len(type_hint.__args__) == 1
        return origin, type_hint.__args__[0]

    return Stateful, type_hint


def _strip_hint_optional(type_hint):
    """
    Checks if the given type hint is an Optional type, and if so, extracts its underlying type.

    This function checks if the provided type hint is an Optional type (e.g., `Optional[int]`).
    If the type hint is Optional, it returns a tuple with a flag set to True and the underlying
    type of the Optional. Otherwise, it returns a tuple with the flag set to False and the original
    type hint.

    Parameters
    ----------
    type_hint : Type
        The type hint to be analyzed for Optional.

    Returns
    -------
    tuple
        A tuple containing:
        - optional (bool): A flag indicating whether the type hint is an Optional type.
        - type_hint (Type): The underlying type of the Optional or the original type hint if not Optional.
    """
    origin = ty.get_origin(type_hint)
    if origin == Optional:  # ty.Union and type_hint._name == "Optional":
        args = ty.get_args(type_hint)
        assert len(args) == 1
        return True, args[0]
    return False, type_hint


def _strip_hint_collection(type_hint):
    """
    Analyzes the type hint to identify the collection type and variable type(s) associated with it.

    This function processes various collection-related type hints, such as Dict, List,
    Tuple, Literal, Enum, and Type. It extracts the collection type (if present) and the
    variable type(s) within the collection. If the type hint does not correspond to a collection,
    it returns the type hint as is.

    Parameters
    ----------
    type_hint : Type
        The type hint to be analyzed.

    Returns
    -------
    tuple
        A tuple containing:
        - collection (Type): The collection type of the type hint, if any (e.g. List, Dict, Tuple).
        - variable_type (Type or tuple of Types): The variable type(s) within the collection or the standalone variable type.
        In case of Literal and Enum, a tuple of valid values is returned.
    
    Raises
    ------
    NotImplementedError
        If the type_hint is not valid or custom classes don't implement __dict__.
    """
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
    if isinstance(type(type_hint), Type):
        assert origin is None
        return Type, type_hint
    raise NotImplementedError


def parse_type_hint(type_hint):
    """
    Parse a type hint to extract its state, optional status, collection type, and
    variable type. This function breaks down the type hint into its constituent
    parts, allowing for easier processing of the type hint during runtime.

    Parameters
    ----------
    type_hint : Type
        The type hint to be parsed.

    Returns
    -------
    Annotation
        A namedtuple containing the following fields:
        - state (Stateful, Derived, Stateless): The state of the type hint.
        - optional (bool): Indicates if the type hint is optional.
        - collection (Type): The collection type of the type hint (e.g. List, Dict, Tuple).
        - variable_type (Type): The variable type within the collection or standalone variable.
    """
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
    """
    Parse and initialize an instance of a custom class based on the provided
    keyword arguments. This function handles the initialization of a custom
    class either directly from a config object or from a dictionary.
    
    Parameters
    ----------
    cls : Type
        The custom class or type that needs to be initialized.
    kwargs : Union[dict, object]
        The keyword arguments or object to be parsed with the given type.

    Returns
    -------
    object
        An instance of the custom class initialized with the provided keyword
    arguments.

    Raises
    ------
    RuntimeError
        If the keyword arguments are of an incompatible type with the custom class,
    a RuntimeError is raised with a message indicating the incompatibility.
        
    """
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
    """
    Parse a value based on the provided annotation. This function helps
    in converting the input value according to the expected type and structure
    defined by the annotation. It handles different collection types, optional
    values, literals, enums, and custom classes.

    Parameters
    ----------
    val : Any
        The value to be parsed.
    annot : Annotation
        The annotation object that describes the expected type and structure
        of the input value.
    name : Optional[str]
        The name of the value, used for error messages (default: None).

    Returns
    -------
    Any
        The parsed value, converted according to the expected type and structure
        defined by the annotation. This can include dictionaries, lists, tuples,
        instances of custom classes, enum values, or literals, depending on the
        annotation.

    Raises
    ------
    RuntimeError
        If a required value is missing or an incompatible length is provided
        for a tuple.
    AssertionError
        If the value is not a valid literal or not supported by the enum.
    NotImplementedError
        If the collection type is not supported.
    """

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
    """
    Get the state of a given annotation. The state can be Stateful, Derived,
    or Stateless, depending on the annotation's origin.

    Parameters
    ----------
    annotation : Type
        The annotation for which the state is required.

    Returns
    -------
    Type
        The state of the given annotation, either Stateful, Derived, or Stateless.
    """
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

"""
Custom types for runtime checking
"""

import typing as ty
from collections import namedtuple
from enum import Enum as _Enum

T = ty.TypeVar("T")


# pylint: disable=deprecated-typing-alias
class Dict(ty.Dict[str, T]):
    """
    A class for dictionary data type, with keys as strings. Used when you need to specify a config
    attribute as a dictionary (in fact, ablator defines ``search_space`` as a dictionary of ``SearchSpace``
    in config class ``ParallelConfig``).
    
    Examples
    --------
    You can declare an attribute of type ``Dict`` as follows:

    >>> @configclass
    >>> class MyConfig(ConfigBase):
    >>>     my_str_dict: Dict[str]
    >>>     my_int_dict: Dict[int]
    >>>     my_space_dict: Dict[SearchSpace]

    When initializing a config object, you can pass a dictionary with keys as strings.
    For values, ablator will automatically cast them to the correct type if possible. For example:

    >>> str_dict = {"str1": "val1", "str2": 2}
    >>> int_dict = {"int1": 1, "int2": 2.5}
    >>> space_dict = {"space1": SearchSpace(value_range = [0, 10], value_type = 'int')}
    >>> MyConfig(my_str_dict=str_dict, my_int_dict=int_dict, my_space_dict=space_dict)
    my_str_dict:
    str1: val1
    str2: '2'
    my_int_dict:
    int1: 1
    int2: 2
    my_space_dict:
        space1:
            value_range:
            - '0'
            - '10'
            categorical_values: null
            subspaces: null
            sub_configuration: null
            value_type: int
            n_bins: null
            log: false

    Notice that the value at key ``str2`` is cast to a string, and the value at key ``int2`` is
    cast to an integer.

    """
    pass


# pylint: disable=deprecated-typing-alias
class List(ty.List[T]):
    """
    A class for list data type, used when you need to specify a config attribute to be a list.
    Remember to wrap the type of the list elements in ``List[]``, e.g ``List[str]``, ``List[int]``.
    
    Examples
    --------
    You can declare an attribute of type ``List`` as follows:

    >>> @configclass
    >>> class MyConfig(ConfigBase):
    >>>     my_str_list: List[str]  # list of strings
    >>>     my_int_list: List[int]  # list of integers

    When initializing a config object, you can pass a list of proper values. In addition,
    ablator will automatically cast them to the correct type if possible. For example:

    >>> MyConfig(my_str_list=["a", "b", 1.5, 2],
    ...          my_int_list=[1, 2, -3.5, 4])
    my_str_dict:
    - a
    - b
    - '1.5'
    - '2'
    my_int_dict:
    - 1
    - 2
    - -3
    - 4

    Notice that the value of ``my_str_list[2]`` and ``my_int_list[3]`` are cast to string,
    and the value of ``my_int_list[2]`` is cast to an integer.

    """
    pass


# pylint: disable=deprecated-typing-alias
class Tuple(ty.Tuple[T]):
    """
    A class for tuple data type, used when you need to specify a config attribute to be a tuple.
    Remember to wrap the type of the tuple elements in ``Tuple[]``. You also have the flexibility
    to specify the number of elements in the tuple and the data type for each of them.

    Examples
    --------
    You can declare an attribute of type ``Tuple`` as follows:

    >>> @configclass
    >>> class MyConfig(ConfigBase):
    >>>     my_str_int_tuple: Tuple[str, int]
    >>>     my_2str_int_tuple: Tuple[str, int, str]

    When initializing a config object, you can pass a tuple of proper values. In addition,
    ablator will automatically cast them to the correct type if possible. For example:

    >>> MyConfig(my_str_int_tuple=("a", 1.5), my_2str_int_tuple=("a", 1, 2))
    my_str_int_tuple:
    - a
    - 1
    my_2str_int_tuple:
    - a
    - 1
    - '2'
    
    Notice how data are cast in ``my_str_int_tuple[1]`` and ``my_2str_int_tuple[2]``.

    .. note::
        The number of elements in the tuple must match the number of types specified in ``Tuple[]``.
        So for the example above, ``my_str_int_tuple`` must have exactly 2 elements, and
        ``my_2str_int_tuple`` must have exactly 3 elements.
    """
    pass


class Optional(ty.Generic[T]):
    """
    A class for optional data types. This is helpful when a config attribute is optional,
    meaning that we can leave an optional config attribute empty. (in fact, ablator defines ``scheduler_config``
    as optional in config class ``TrainConfig``).
    
    Examples
    --------
    You can declare an attribute of type ``Optional`` as follows:

    >>> @configclass
    >>> class MyConfig(ConfigBase):
    >>>     my_optional_list: Optional[List[str]]

    When initializing a config object, you can pass a ``List[str]`` value to ``a4``, or not passing
    values at all:

    >>> MyConfig(my_optional_list=["a"])
    my_optional_list:
    - a
    >>> MyConfig()
    my_optional_list: null
    """
    pass


# Support for nested objects
class Self:
    pass


Type = type
Literal = ty.Literal


class Enum(_Enum):
    """
    A custom Enum class that provides additional equality and hashing methods. This is useful when creating
    custom data types that take as value elements from a fixed set. In ablator, we use this class to define
    ``Optim``, which specifies the optimization direction: ``Optim.min`` or ``Optim.max``. ``Optim`` is used
    in config class ``ParallelConfig`` (``optim_metrics`` attribute).

    Methods
    -------
    __eq__(self, __o: object) -> bool:
        Checks for equality between the Enum instance and another object.

    __hash__(self) -> int:
        Calculates the hash of the Enum instance.

    Examples
    --------
    Create a custom Enum class by inheriting from ``Enum``:

    >>> from ablator import Enum
    >>> class Color(Enum):
    >>>     RED = 1
    >>>     GREEN = 2
    >>>     BLUE = 3
    
    ``RED``, ``GREEN``, and ``BLUE`` are fixed value set for Color type. Internally, these values are
    mapped to integers 1, 2, and 3. The custom data type ``Color`` can now be used in config classes:

    >>> @configclass
    >>> class MyConfig(ConfigBase):
    >>>     my_color: Color
    >>> MyConfig(my_color=Color.RED)
    my_color: 1

    """

    def __eq__(self, __o: object) -> bool:
        """
        Checks for equality between the Enum instance and another object.

        Parameters
        ----------
        __o : object
            The object to compare with the Enum instance.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.

        Examples
        --------
        >>> Color.RED == Color.RED
        True
        >>> Color.RED == 1
        True
        """
        val = __o
        if not isinstance(val, type(self)):
            val = type(self)(val)
        return super().__eq__(val)

    def __hash__(self):
        """
        Calculates the hash of the Enum instance.

        Returns
        -------
        int
            The hash value of the Enum instance.

        Examples
        --------
        >>> hash(Color.RED) == hash(Color.RED)
        True
        """
        return _Enum.__hash__(self)


# ALLOWED_COLLECTIONS is meant to only support collections that can be
# expressed in a yaml file in an unambigious manner and using primitives
# int, float and str (ALLOWED_TYPES)
# Extending the type system should be intuitive.
# Each annotation is of the format
# "STATE", "OPTIONAL", "COLLECTION", "TYPE"

ALLOWED_TYPES = (int, float, str, bool, None)

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

Parsing the dictionary however can be error prone if you have complex arguments
and is not advised.
"""


def _val2bool(val: str | bool) -> bool:
    """
    _val2bool parses the string representation as a boolean expression. It
    returns `True` if `val` is in ["true", "t", "1"] (case-insentivive)
    and `False` if `val` is in ["false", "f", "0"] (case-insentivive).

    Parameters
    ----------
    val : str | bool
        the value to parse

    Returns
    -------
    bool
        the boolean representation of `val`

    Raises
    ------
    ValueError
        It raises an error if `val` is not in ["true", "t", "1", "false", "f", "0"] (case-insentivive).
    """
    if isinstance(val, bool):
        return val
    if val.lower() in {"true", "t", "1"}:
        return True
    if val.lower() in {"false", "f", "0"}:
        return False
    raise ValueError(f"Cannot parse {val} as bool.")


def _strip_hint_state(type_hint):
    """
    Strips the hint state from a type hint.

    Parameters
    ----------
    type_hint : Type
        The input type hint to strip the state from.

    Returns
    -------
    tuple
        A tuple containing the state and the remaining type hint.

    Examples
    --------
    >>> _strip_hint_state(Stateful[int])
    (Stateful, int)
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
    Strips the optional part of a type hint.

    Parameters
    ----------
    type_hint : Type
        The input type hint to strip the optional part from.

    Returns
    -------
    tuple
        A tuple containing a boolean indicating if the type hint is optional and the remaining type hint.

    Examples
    --------
    >>> _strip_hint_optional(Optional[int])
    (True, int)
    """
    if ty.get_origin(type_hint) == Optional:
        args = ty.get_args(type_hint)
        assert len(args) == 1
        return True, args[0]
    return False, type_hint


def _strip_hint_collection(type_hint):
    """
    Strips the collection from a type hint.

    Parameters
    ----------
    type_hint : Type
        The input type hint to strip the collection from.

    Returns
    -------
    tuple
        A tuple containing the collection and the variable type.

    Raises
    ------
    NotImplementedError
        If the type hint is not valid or custom classes don't implement __dict__.

    Examples
    --------
    >>> _strip_hint_collection(List[int])
    (List, int)
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
    raise NotImplementedError(
        f"{type_hint} is not a valid hint. Custom classes must implement __dict__."
    )


def parse_type_hint(cls, type_hint):
    """
    Parses a type hint and returns a parsed annotation.

    Parameters
    ----------
    cls : Any
        The class being annotated.
    type_hint : Type
        The input type hint to parse.

    Returns
    -------
    Annotation
        A namedtuple containing ``state``, ``optional``, ``collection``, and ``variable_type`` information.

    Examples
    --------
    >>> parse_type_hint(Optional[List[int]])
    Annotation(state=Stateful, optional=True, collection=List, variable_type=int)
    """
    state, _type_hint = _strip_hint_state(type_hint)

    optional, _type_hint = _strip_hint_optional(_type_hint)

    collection, variable_type = _strip_hint_collection(_type_hint)
    if variable_type == Self:
        variable_type = cls
    return Annotation(
        state=state,
        optional=optional,
        collection=collection,
        variable_type=variable_type,
    )


def _parse_class(cls, kwargs):
    """
    Parse values whose types are not  a collection or in ALLOWED_TYPES
    eg. bool, added dict(tune configs)

    Parameters
    ----------
    cls : Type
        The input Type
    kwargs : dict or object
        The keyword arguments or object to parse with the given type

    Returns
    -------
    object
        Parsed object

    Raises
    ------
    RuntimeError
        If the input kwargs is incompatible
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


# pylint: disable=too-complex
def parse_value(val, annot: Annotation, name=None):
    """
    Parses a value based on the given annotation.

    Parameters
    ----------
    val : Any
        The input value to parse.
    annot : Annotation
        The annotation namedtuple to guide the parsing.
    name : str, optional
        The name of the value, by default ``None``.

    Returns
    -------
    Any
        The parsed value.

    Raises
    ------
    RuntimeError
        If the required value is missing and it is not optional or derived or stateless.
    ValueError
        If the value type in dict is not valid
        If the value of a list is no valid

    Examples
    --------
    >>> annotation = parse_type_hint(Optional[List[int]])
    >>> parse_value([1, 2, 3], annotation)
    [1, 2, 3]
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
    if annot.collection == Dict and (
        annot.variable_type in ALLOWED_TYPES or issubclass(annot.variable_type, Enum)
    ):
        return {str(_k): annot.variable_type(_v) for _k, _v in val.items()}
    if annot.collection == Dict and issubclass(type(annot.variable_type), Type):
        return_dictionary = {}
        for _k, _v in val.items():
            if isinstance(_v, dict):
                return_dictionary[_k] = annot.variable_type(**_v)
            elif isinstance(_v, annot.variable_type):
                return_dictionary[_k] = _v
            else:
                raise ValueError(f"Invalid type {type(_v)} for {_k} and field {name}")
        return return_dictionary
    if annot.collection == List:
        if not isinstance(val, list):
            raise ValueError(f"Invalid type {type(val)} for type List")
        if annot.variable_type in ALLOWED_TYPES or issubclass(
            annot.variable_type, Enum
        ):
            return_list = []
            for _v in val:
                return_list.append(annot.variable_type(_v))
            return return_list
        if issubclass(type(annot.variable_type), Type):
            _kwargs = annot._asdict()
            _kwargs["collection"] = Type
            return [parse_value(_v, Annotation(**_kwargs)) for _v in val]
        raise ValueError(f"Invalid type {type(annot.variable_type)} and field {name}")
    if annot.collection == Tuple:
        assert len(val) == len(
            annot.variable_type
        ), f"Incompatible lengths for {name} between {val} and type_hint: {annot.variable_type}"
        return [tp(_v) for tp, _v in zip(annot.variable_type, val)]
    if annot.collection == Type:
        return _parse_class(annot.variable_type, val)
    if annot.collection is None and annot.variable_type == bool:
        return _val2bool(val)
    if annot.collection is None:
        return annot.variable_type(val)
    if issubclass(annot.collection, Enum):
        assert (
            val in annot.variable_type
        ), f"{val} is not supported by {annot.collection}"
        return annot.collection(val)
    raise NotImplementedError


class Stateful(ty.Generic[T]):
    """
    This is for attributes that are fixed between experiments. By default, we assume that unannotated attributes
    are stateful. Unlike ``Derived`` and ``Stateless``, in which you have to annotate attributes with these classes,
    e.g. ``attr: Statess[int]`` or ``attr: Statess[List[str]]``, for stateful, just define them without
    ``Stateful``, e.g ``attr: int`` or ``attr: List[str]``.

    Examples
    --------
    The below example defines a model config that has stateful embedding dimensions, which means among every experiment,
    the embedding dimension must be the same (and will be 100).

    >>> @configclass
    >>> class MyModelConfig(ModelConfig):
    >>>     embed_dim: int
    >>> model_config = MyModelConfig(embed_dim=100) # Must provide values for ``embed_dim`` before launching experiment

    .. note::
        - In contrary to ``Derived``, when initializing config objects (aka before launching the experiment), you have to
          assign values to their stateful attributes.
        - Stateful is only applied in the context of experiments. So a stateful attribute must be the same between different
          run of the same experiment configurations. However, within each experiment, a search space on stateful attributes
          can be defined to run HPO on them.

    """


class Derived(ty.Generic[T]):
    """
    This type is for attributes that are derived during the experiment (after launching the experiment).
    To make an attribute derived, wrap ``Derived`` around its type defenition, e.g ``Derived[List[int]]``,
    ``Derived[str]``.

    Examples
    --------
    For example, you want to test how different pretrained word embeddings (e.g word2vec 100d, word2vec 300d) affect the
    performance of a classification model, and you will use ablator to run ablation study on the effect of word embeddings.
    Plus, the classification model architecture depends on the size of the embedding length of each pretrained set of word
    embeddings. In this case, the model architecture is derived from the pretrained word embeddings. So you can define a model
    config class as follows:

    >>> @configclass
    >>> class MyModelConfig(ModelConfig):
    >>>     embed_dim: Derived[int]

    Then you can define a model class that takes in the model config as input and set input length using ``embed_dim``:
    
    >>> class MyModel(nn.Module):
    >>>     def __init__(self, config: MyModelConfig):
    >>>         super().__init__()
    >>>         self.embed_dim = config.embed_dim

    Finally, ``config_parser`` is used to set the value of Derived attribute ``embed_dim`` based on the pretrained word embeddings:

    >>> class MyLMWrapper(ModelWrapper):
    >>>     def config_parser(self, run_config: RunConfig):
    >>>         run_config.model_config.embed_dim = len(self.train_dataloader.word2vec.wv.vocab)
    >>>         return run_config

    .. note::
        When initializing config objects, you do not have to assign values to attributes that are of ``Derived`` type.

    """


class Stateless(ty.Generic[T]):
    """
    This type is for attributes that can take different value assignments between experiments. To make an
    attribute stateless, wrap ``Stateless`` around its type defenition, e.g ``Stateless[List[int]]``,
    ``Stateless[str]``.

    Examples
    --------

    >>> @configclass
    >>> class MyModelConfig(ConfigBase):
    >>>     attr: Stateless[List[int]]
    >>> config = MyModelConfig(attr=[5,"6",7.25])  # Must provide values for ``attr`` before launching experiment

    .. note::
        Unlike ``Derived``, when initializing config objects (aka before launching the experiment) that have stateless
        attributes, you have to assign values to these attributes.
    """

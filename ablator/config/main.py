from collections import ChainMap, abc
import copy
import inspect
import logging
import operator
import typing as ty
from typing import Any, Union
from functools import partial
from pathlib import Path
from typing_extensions import Self

from omegaconf import OmegaConf

from ablator.config.types import (
    Annotation,
    Derived,
    Dict,
    Enum,
    List,
    Literal,
    Stateless,
    Tuple,
    Type,
    parse_type_hint,
    parse_value,
)
from ablator.config.utils import dict_hash, flatten_nested_dict, parse_repr_to_kwargs


def configclass(cls: type["ConfigBase"]) -> type["ConfigBase"]:
    """
    Decorator for ``ConfigBase`` subclasses, adds the ``config_class`` attribute to the class.

    Parameters
    ----------
    cls : type["ConfigBase"]
        The class to be decorated.

    Returns
    -------
    type[ConfigBase]
        The decorated class with the ``config_class`` attribute.
    """

    assert issubclass(cls, ConfigBase), f"{cls.__name__} must inherit from ConfigBase"
    setattr(cls, "config_class", cls)
    return cls


def _freeze_helper(obj):
    def __setattr__(self, k, v):
        if getattr(self, "_freeze", False):
            raise RuntimeError(
                f"Can not set attribute {k} on a class of a frozen configuration"
                f" ``{type(self).__name__}``."
            )
        super(type(self), self).__setattr__(k, v)

    try:
        obj._freeze = True  # pylint: disable=protected-access
        type(obj).__setattr__ = __setattr__
    except Exception:  # pylint: disable=broad-exception-caught
        # this is the case where the object does not have
        # attribute setter function
        pass


def _unfreeze_helper(obj):
    if hasattr(obj, "_freeze"):
        super(type(obj), obj).__setattr__("_freeze", False)


def _parse_reconstructor(val, ignore_stateless: bool, flatten: bool):
    if isinstance(val, (int, float, bool, str, type(None))):
        return val
    if issubclass(type(val), ConfigBase):
        return val.make_dict(
            val.annotations, ignore_stateless=ignore_stateless, flatten=flatten
        )
    if issubclass(type(val), Enum):
        return val.value
    args, kwargs = parse_repr_to_kwargs(val)
    if len(args) == 0:
        return kwargs
    if len(kwargs) == 0:
        return args
    return args, kwargs


class Missing:
    """
    This type is defined only for raising an error
    """


# @dataclass(repr=False)
class ConfigBase:
    # NOTE: this allows for non-defined arguments to be created. It is very bug-prone and will be disabled.
    """

    This class is the building block for all configuration objects within ablator. It serves as the base class for
    configurations such as ``ModelConfig``, ``TrainConfig``, ``OptimizerConfig``, and more. Together with
    ``@configclass``, it allows for the creation of config classes of customized attributes without the need to
    define a constructor. ``ConfigBase`` and ``@configclass`` take care of the initialization and parsing of the
    attributes. The example section below shows this in more detail.

    In summary, to customize configurations for specific needs, you can create your own configuration class by
    inheriting it from ``ConfigBase``. It's essential to annotate it with ``@configclass``. In the tutorial
    `Search space for different types of optimizers and scheduler
    <./notebooks/Searchspace-for-diff-optimizers.ipynb>`_, a custom optimizer config class is created to enable
    ablation study on various optimizers and schedulers. You can refer to this tutorial for a realistic example of
    how to create your custom configuration class.

    .. note::
        One key takeaway is that when initializing a config object, you can look into the list of attributes defined
        in the config class to see what arguments you can pass.

    Parameters
    ----------
    *args : Any
        This argument is just for disabling passing by positional arguments.
    debug : bool, optional
        Whether to load the configuration in debug mode and ignore discrepancies/errors, by default ``False``.
    **kwargs : Any
        Keyword arguments. Possible arguments are from the annotations of the configuration class. You can look into the
        Examples section for more details.

    Attributes
    ----------
    config_class : Type
        The class of the configuration object.

    Raises
    ------
    ValueError
        If positional arguments are provided or there are missing required values.
    KeyError
        If unexpected arguments are provided.
    RuntimeError
        If the class is not decorated with ``@configclass``.

    .. note::
       All config classes must be decorated with ``@configclass``.

    Examples
    --------

    >>> @configclass
    >>> class MyCustomConfig(ConfigBase):
    ...     attr1: int = 1
    ...     attr2: Tuple[str, int, str]
    >>> my_config = MyCustomConfig(attr1=4, attr2=("hello", 1, "world"))  # Pass by named arguments
    >>> kwargs = {"attr1": 4, "attr2": ("hello", 1, "world")}   # Pass by keyword arguments
    >>> my_config = MyCustomConfig(**kwargs)

    Note that since we defined ``MyCustomConfig`` as a config class with two annotated attributes ``attr1``
    and ``attr2`` (without a constructor, which is automatically handled by ``ConfigBase`` and
    ``@configclass``), when creating the config object, you can directly pass ``attr1`` and ``attr2``. You
    can also pass these arguments as keyword arguments.

    """
    config_class = type(None)

    def __init__(self, *args: Any, debug: bool = False, **kwargs: Any):
        self._debug: bool
        self._freeze: bool
        self._class_name: str
        self.__setattr__internal("_debug", debug)
        self.__setattr__internal("_freeze", False)
        self.__setattr__internal("_class_name", type(self).__name__)

        missing_vals = self._validate_inputs(*args, debug=debug, **kwargs)

        assert len(missing_vals) == 0 or debug
        for k in self.annotations:
            if k in kwargs:
                v = kwargs[k]
                del kwargs[k]
            else:
                v = getattr(self, k, None)
            if k in missing_vals:
                logging.warning(
                    (
                        "Loading %s in `debug` mode. Setting missing required value %s"
                        " to `None`."
                    ),
                    self._class_name,
                    k,
                )
                self.__setattr__internal(k, None)

            else:
                try:
                    setattr(self, k, v)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    if not debug:
                        raise e
                    logging.warning(
                        (
                            "Loading %s in `debug` mode. Unable to parse `%s` value %s."
                            " Setting to `None`."
                        ),
                        self._class_name,
                        k,
                        v,
                    )
                    self.__setattr__internal(k, None)

        if len(kwargs) > 0 and not debug:
            unspected_args = ", ".join(kwargs.keys())
            raise KeyError(f"Unexpected arguments: `{unspected_args}`")
        if len(kwargs) > 0:
            unspected_args = ", ".join(kwargs.keys())
            logging.warning(
                "Loading %s in `debug` mode. Ignoring unexpected arguments: `%s`",
                self._class_name,
                unspected_args,
            )

    def _validate_inputs(self, *args, debug: bool, **kwargs) -> list[str]:
        added_variables = {
            item[0]
            for item in inspect.getmembers(type(self))
            if not inspect.isfunction(item[1]) and not item[0].startswith("_")
        }

        base_variables = {
            item[0]
            for item in inspect.getmembers(ConfigBase)
            if not inspect.isfunction(item[1])
        }
        non_annotated_variables = (
            added_variables - base_variables - set(self.annotations.keys())
        )
        assert (
            len(non_annotated_variables) == 0
        ), f"All variables must be annotated. {non_annotated_variables}"
        if len(args) > 0:
            raise ValueError(
                f"{self._class_name} does not support positional arguments."
            )
        if not isinstance(self, self.config_class):  # type: ignore[arg-type]
            raise RuntimeError(
                f"You must decorate your Config class '{self._class_name}' with"
                " ablator.configclass."
            )
        missing_vals = self._validate_missing(**kwargs)
        if len(missing_vals) != 0 and not debug:
            raise ValueError(f"Missing required values {missing_vals}.")
        return missing_vals

    def _validate_missing(self, **kwargs) -> list[str]:
        missing_vals = []
        for k, annotation in self.annotations.items():
            if not annotation.optional and annotation.state not in [Derived]:
                # make sure non-optional and derived values are not empty or
                # without a default assignment
                if not (
                    (k in kwargs and kwargs[k] is not None)
                    or getattr(self, k, None) is not None
                ):
                    missing_vals.append(k)
        return missing_vals

    def __setattr__internal(self, k, v):
        super().__setattr__(k, v)

    def __setattr__(self, k, v):
        if self._freeze:
            raise RuntimeError(
                f"Can not set attribute {k} on frozen configuration"
                f" ``{type(self).__name__}``."
            )
        annotation = self.annotations[k]
        v = parse_value(v, annotation, k, self._debug)
        self.__setattr__internal(k, v)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return len(self.diff(other)) == 0
        return False

    def __repr__(self) -> str:
        """
        Return the string representation of the configuration object.

        Returns
        -------
        str
            The string representation of the configuration object.

        """

        return (
            self._class_name
            + "("
            + ", ".join(
                [
                    f"{k}='{v}'" if isinstance(v, str) else f"{k}={v.__repr__()}"
                    for k, v in self.to_dict().items()
                ]
            )
            + ")"
        )

    def keys(self) -> abc.KeysView[str]:
        """
        Get the keys of the configuration dictionary.

        Returns
        -------
        abc.KeysView[str]
            The keys of the configuration dictionary.
        """
        return self.to_dict().keys()

    @classmethod
    def load(cls, path: Union[Path, str], debug: bool = False) -> Self:
        """
        Load a configuration object from a file.

        Parameters
        ----------
        path : Union[Path, str]
            The path to the configuration file.
        debug : bool, optional
            Whether to load the configuration in debug mode, and ignore discrepancies/errors,
            by default ``False``.

        Returns
        -------
        Self
            The loaded configuration object.
        """
        # TODO{iordanis} remove OmegaConf dependency
        kwargs: dict = OmegaConf.to_object(  # type: ignore[assignment]
            OmegaConf.create(Path(path).read_text(encoding="utf-8"))
        )
        return cls(**kwargs, debug=debug)

    @property
    def annotations(self) -> dict[str, Annotation]:
        """
        Get the parsed annotations of the configuration object.

        Returns
        -------
        dict[str, Annotation]
            A dictionary of parsed annotations.
        """
        annotations = {}
        if hasattr(self, "__annotations__"):
            annotation_types = ChainMap(
                *(
                    c.__annotations__
                    for c in type(self).__mro__
                    if "__annotations__" in c.__dict__
                )
            )
            annotations = {
                field_name: parse_type_hint(type(self), annotation)
                for field_name, annotation in annotation_types.items()
            }
        return annotations

    def get_val_with_dot_path(self, dot_path: str) -> Any:
        """
        Get the value of a configuration object attribute using dot notation.

        Parameters
        ----------
        dot_path : str
            The dot notation path to the attribute.

        Returns
        -------
        Any
            The value of the attribute.
        """
        return operator.attrgetter(dot_path)(self)

    def get_type_with_dot_path(self, dot_path: str) -> Type:
        """
        Get the type of a configuration object attribute using dot notation.

        Parameters
        ----------
        dot_path : str
            The dot notation path to the attribute.

        Returns
        -------
        Type
            The type of the attribute.
        """
        val = self.get_val_with_dot_path(dot_path)
        return type(val)

    def get_annot_type_with_dot_path(self, dot_path: str) -> Type:
        """
        Get the type of a configuration object annotation using dot notation.

        Parameters
        ----------
        dot_path : str
            The dot notation path to the annotation.

        Returns
        -------
        Type
            The type of the annotation.
        """
        *base_path, element = dot_path.split(".")
        annot_dot_path = ".".join(base_path + ["annotations"])
        annot: dict[str, Annotation] = self.get_val_with_dot_path(annot_dot_path)
        return annot[element].variable_type

    # pylint: disable=too-complex
    def make_dict(
        self,
        annotations: dict[str, Annotation],
        ignore_stateless: bool = False,
        flatten: bool = False,
    ) -> dict:
        """
        Create a dictionary representation of the configuration object.

        Parameters
        ----------
        annotations : dict[str, Annotation]
            A dictionary of annotations.
        ignore_stateless : bool
            Whether to ignore stateless values, by default ``False``.
        flatten : bool
            Whether to flatten nested dictionaries, by default ``False``.

        Returns
        -------
        dict
            The dictionary representation of the configuration object.

        Raises
        ------
        NotImplementedError
            If the type of annot.collection is not supported.
        """
        return_dict = {}
        parse_reconstructor = partial(
            _parse_reconstructor, ignore_stateless=ignore_stateless, flatten=flatten
        )
        for field_name, annot in annotations.items():
            if ignore_stateless and (annot.state in {Stateless, Derived}):
                continue

            _val = getattr(self, field_name)
            if annot.collection in [None, Literal] or _val is None:
                val = _val
            elif annot.collection == List:
                val = [parse_reconstructor(_lval) for _lval in _val]
            elif annot.collection == Tuple:
                val = tuple(parse_reconstructor(_lval) for _lval in _val)
            elif annot.collection in [Dict]:
                val = {k: parse_reconstructor(_dval) for k, _dval in _val.items()}
            elif issubclass(type(_val), ConfigBase):
                val = _val.make_dict(
                    _val.annotations, ignore_stateless=ignore_stateless, flatten=flatten
                )

            elif annot.collection == Type:
                if annot.optional and _val is None:
                    val = None
                else:
                    val = parse_reconstructor(_val)
            elif issubclass(type(_val), Enum):
                val = _val.value
            else:
                raise NotImplementedError
            return_dict[field_name] = val
        if flatten:
            return_dict = flatten_nested_dict(return_dict)
        return return_dict

    def write(self, path: Union[Path, str]):
        """
        Write the configuration object to a file.

        Parameters
        ----------
        path : Union[Path, str]
            The path to the file.

        """
        Path(path).write_text(self.to_yaml(), encoding="utf-8")

    def diff_str(
        self, config: "ConfigBase", ignore_stateless: bool = False
    ) -> list[str]:
        """
        Get the differences between the current configuration object and another configuration object as strings.

        Parameters
        ----------
        config : ConfigBase
            The configuration object to compare.
        ignore_stateless : bool
            Whether to ignore stateless values, by default ``False``.

        Returns
        -------
        list[str]
            The list of differences as strings.

        """
        diffs = self.diff(config, ignore_stateless=ignore_stateless)
        str_diffs = []
        for p, (l_t, l_v), (r_t, r_v) in diffs:
            _diff = f"{p}:({l_t.__name__}){l_v}->({r_t.__name__}){r_v}"
            str_diffs.append(_diff)
        return str_diffs

    def diff(
        self, config: "ConfigBase", ignore_stateless: bool = False
    ) -> list[tuple[str, tuple[type, Any], tuple[type, Any]]]:
        """
        Get the differences between the current configuration object and another configuration object.

        Parameters
        ----------
        config : ConfigBase
            The configuration object to compare.
        ignore_stateless : bool
            Whether to ignore stateless values, by default ``False``

        Returns
        -------
        list[tuple[str, tuple[type, Any], tuple[type, Any]]]
            The list of differences as tuples.

        Examples
        --------
        Let's say we have two configuration objects ``config1`` and ``config2`` with the following attributes:

        >>> config1:
            learning_rate: 0.01
            optimizer: 'Adam'
            num_layers: 3

        >>> config2:
            learning_rate: 0.02
            optimizer: 'SGD'
            num_layers: 3

        The diff between these two configurations would look like:

        >>> config1.diff(config2)
        [('learning_rate', (float, 0.01), (float, 0.02)), ('optimizer', (str, 'Adam'), (str, 'SGD'))]

        In this example, the learning_rate and optimizer values are different between the two configuration objects.

        """
        left_config = copy.deepcopy(self)
        right_config = copy.deepcopy(config)
        left_dict = left_config.make_dict(
            left_config.annotations, ignore_stateless=ignore_stateless, flatten=True
        )

        right_dict = right_config.make_dict(
            right_config.annotations, ignore_stateless=ignore_stateless, flatten=True
        )
        left_keys = set(left_dict.keys())
        right_keys = set(right_dict.keys())
        diffs: list[tuple[str, tuple[type, ty.Any], tuple[type, ty.Any]]] = []
        for k in left_keys.union(right_keys):
            if k not in left_dict:
                right_v = right_dict[k]
                right_type = type(right_v)
                diffs.append((k, (Missing, None), (right_type, right_v)))

            elif k not in right_dict:
                left_v = left_dict[k]
                left_type = type(left_v)
                diffs.append((k, (left_type, left_v), (Missing, None)))

            elif left_dict[k] != right_dict[k] or not isinstance(
                left_dict[k], type(right_dict[k])
            ):
                right_v = right_dict[k]
                left_v = left_dict[k]
                left_type = type(left_v)
                right_type = type(right_v)
                diffs.append((k, (left_type, left_v), (right_type, right_v)))
        return diffs

    def to_dict(self, ignore_stateless: bool = False) -> dict:
        """
        Convert the configuration object to a dictionary.

        Parameters
        ----------
        ignore_stateless : bool
            Whether to ignore stateless values, by default ``False``.

        Returns
        -------
        dict
            The dictionary representation of the configuration object.

        """
        return self.make_dict(self.annotations, ignore_stateless=ignore_stateless)

    def to_yaml(self) -> str:
        """
        Convert the configuration object to YAML format.

        Returns
        -------
        str
            The YAML representation of the configuration object.

        """
        # TODO: investigate https://github.com/crdoconnor/strictyaml as an alternative to OmegaConf
        conf = OmegaConf.create(self.to_dict())
        return OmegaConf.to_yaml(conf)

    def to_dot_path(self, ignore_stateless: bool = False) -> str:
        """
        Convert the configuration object to a dictionary with dot notation paths as keys.

        Parameters
        ----------
        ignore_stateless : bool
            Whether to ignore stateless values, by default ``False``.

        Returns
        -------
        str
            The YAML representation of the configuration object in dot notation paths.

        """
        _flat_dict = self.make_dict(
            self.annotations, ignore_stateless=ignore_stateless, flatten=True
        )
        return OmegaConf.to_yaml(OmegaConf.create(_flat_dict))

    @property
    def uid(self) -> str:
        """
        Get the unique identifier for the configuration object.

        Returns
        -------
        str
            The unique identifier for the configuration object.

        """
        return dict_hash(self.make_dict(self.annotations, ignore_stateless=True))[:5]

    def assert_unambigious(self):
        """
        Assert that the configuration object is unambiguous and has all the required values.

        Raises
        ------
        RuntimeError
            If the configuration object is ambiguous or missing required values.

        """
        for k, annot in self.annotations.items():
            if not annot.optional and getattr(self, k) is None:
                raise RuntimeError(
                    f"Ambiguous configuration `{self._class_name}`. Must provide value"
                    f" for {k}"
                )
        self._apply_lambda_recursively("assert_unambigious")

    def freeze(self):
        self.__setattr__internal("_freeze", True)
        self._apply_lambda_recursively("freeze")

        for k, annot in self.annotations.items():
            if (
                isinstance(annot.variable_type, type)
                and not issubclass(annot.variable_type, ConfigBase)
                and getattr(self, k) is not None
                and hasattr(getattr(self, k), "__setattr__")
            ):
                if annot.collection in [List, Tuple]:
                    for _lval in getattr(self, k):
                        _freeze_helper(_lval)
                elif annot.collection in [Dict]:
                    for _lval in getattr(self, k).values():
                        _freeze_helper(_lval)
                else:
                    _freeze_helper(getattr(self, k))

    def _unfreeze(self):
        self.__setattr__internal("_freeze", False)
        self._apply_lambda_recursively("_unfreeze")

        for k, annot in self.annotations.items():
            if (
                isinstance(annot.variable_type, type)
                and not issubclass(annot.variable_type, ConfigBase)
                and getattr(self, k) is not None
            ):
                if annot.collection in [List, Tuple]:
                    for _lval in getattr(self, k):
                        _unfreeze_helper(_lval)
                elif annot.collection in [Dict]:
                    for _lval in getattr(self, k).values():
                        _unfreeze_helper(_lval)
                else:
                    _unfreeze_helper(getattr(self, k))

    def _apply_lambda_recursively(self, lam: str, *args):
        for k, annot in self.annotations.items():
            if (
                isinstance(annot.variable_type, type)
                and issubclass(annot.variable_type, ConfigBase)
                and getattr(self, k) is not None
            ):
                if annot.collection in [List, Tuple]:
                    for _lval in getattr(self, k):
                        getattr(_lval, lam)(*args)
                elif annot.collection in [Dict]:
                    for _lval in getattr(self, k).values():
                        getattr(_lval, lam)(*args)
                else:
                    getattr(getattr(self, k), lam)(*args)

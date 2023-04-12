import copy
import inspect
import operator
import typing as ty
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf
from trainer.config.types import (
    Annotation,
    Derived,
    Dict,
    Enum,
    List,
    Stateless,
    Tuple,
    Type,
    parse_type_hint,
    parse_value,
)
from trainer.config.utils import dict_hash, flatten_nested_dict


def configclass(cls):
    """
    A decorator that ensures the provided class inherits from ConfigBase and adds a
    'config_class' attribute to the class. Additionally, it utilizes the 'dataclass'
    decorator to provide extra functionality to the class, such as skipping the
    automatic generation of the init method, not using the default repr method,
    and requiring keyword arguments for instantiation.

    Parameters
    ----------
    cls : Type[ConfigBase]
        The class to be processed, which must inherit from ConfigBase.

    Returns
    -------
    Type[ConfigBase]
        The processed config class, with the 'config_class' attribute and the
        enhanced functionality provided by the 'dataclass' decorator.

    Raises
    ------
    AssertionError
        If the provided class does not inherit from ConfigBase, an AssertionError is raised.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> class ConfigBase:
    ...     pass
    ...
    >>> @configclass
    ... class MyConfig(ConfigBase):
    ...     a: int
    ...     b: str
    ...
    >>> config = MyConfig(a=1, b="test")
    >>> config.config_class
    <class '__main__.MyConfig'>

    Notes
    -----
    More information about `dataclass`:
        https://docs.python.org/3/library/dataclasses.html
    """
    assert issubclass(cls, ConfigBase), f"{cls.__name__} must inherit from ConfigBase"
    setattr(cls, "config_class", cls)
    return dataclass(cls, init=False, repr=False, kw_only=True)


class Missing:
    """
    This type is defined only for raising an error
    """


@dataclass(repr=False)
class ConfigBase:
    """
    ConfigBase class provides the base functionality for configuration classes.

    This class should be inherited by all configuration classes, and the resulting
    subclasses should be decorated with the `configclass` decorator. It provides
    methods for parsing, loading, merging, and comparing configurations.

    Attributes
    ----------
    config_class : Type
        A type attribute that stores the reference to the actual configuration class.

    """
    # NOTE: this allows for non-defined arguments to be created. It is very bug-prone and will be disabled.
    config_class = type(None)

    def __init__(self, *args, add_attributes=False, **kwargs):
        """
        Initialize the ConfigBase object by processing the input arguments and setting the
        appropriate attributes for the object. It also checks for any missing required values
        and raises errors if necessary.

        Parameters
        ----------
        *args : tuple
            Positional arguments are not supported, and will raise a ValueError if provided.
        add_attributes : bool, optional
            If True, any additional keyword arguments not specified in the annotations
            will be added as attributes to the object. Default is False.
        **kwargs : dict
            Keyword arguments corresponding to the attributes of the ConfigBase object.

        Raises
        ------
        AssertionError
            If there are non-annotated variables in the class.
        ValueError
            If positional arguments are provided.
        RuntimeError
            If the Config class is not decorated with trainer.configclass.
        KeyError
            If unexpected arguments are provided and add_attributes is False.
        """
        
        class_name = type(self).__name__
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
            raise ValueError(f"{class_name} does not support positional arguments.")
        if not isinstance(self, self.config_class):
            raise RuntimeError(
                f"You must decorate your Config class '{class_name}' with trainer.configclass."
            )
        missing_vals = []
        for k, annotation in self.annotations.items():
            if not annotation.optional and annotation.state not in [Derived, Stateless]:
                # make sure non-optional and derived values are not empty or
                # without a default assignment
                if not (
                    (k in kwargs and kwargs[k] is not None)
                    or getattr(self, k, None) is not None
                ):
                    missing_vals.append(k)
        assert len(missing_vals) == 0, f"Missing required value {missing_vals}"

        for k, annotation in self.annotations.items():
            if k in kwargs:
                v = kwargs[k]
                del kwargs[k]
            else:
                v = getattr(self, k, None)

            v = parse_value(v, annotation, k)
            setattr(self, k, v)

        if add_attributes and len(kwargs) > 0:
            setattr(self, k, v)
        elif len(kwargs) > 0:
            unspected_args = ", ".join(kwargs.keys())
            raise KeyError(f"Unexpected arguments: `{unspected_args}`")

    def keys(self):
        """
        Return the keys of the configuration dictionary.

        Returns
        -------
        dict_keys
            The keys of the configuration dictionary.
        """
        return self.to_dict().keys()

    # def __getitem__(self, item):
    #     return self.to_dict()[item]

    @classmethod
    def load(cls, path: ty.Union[Path, str]):
        """
        Load a configuration object from a YAML file located at the given path.

        Parameters
        ----------
        path : ty.Union[Path, str]
            The path to the YAML file containing the configuration data.

        Returns
        -------
        ConfigBase
            An instance of the ConfigBase subclass with the loaded configuration data.

        Examples
        --------
        >>> config = MyConfig.load("config.yaml")
        """
        kwargs: dict = OmegaConf.to_object(OmegaConf.create(Path(path).read_text(encoding="utf-8")))  # type: ignore
        return cls(**kwargs)

    @property
    def annotations(self) -> dict[str, Annotation]:
        """
        Get the annotations for the ConfigBase object, including dataclass fields and
        custom annotations.

        Returns
        -------
        dict[str, Annotation]
            A dictionary with field names as keys and Annotation objects as values.

        Examples
        --------
        >>> config = MyConfig(a=1, b="test")
        >>> config.annotations
        {'a': Annotation(variable_type=<class 'int'>, ...), 'b': Annotation(variable_type=<class 'str'>, ...)}
        """
        annotations = {}
        if hasattr(self, "__annotations__"):
            annotation_types = dict(self.__annotations__)
            # pylint: disable=no-member
            dataclass_types = {k: v.type for k, v in self.__dataclass_fields__.items()}
            annotation_types.update(dataclass_types)

            annotations = {
                field_name: parse_type_hint(annotation)
                for field_name, annotation in annotation_types.items()
            }
        return annotations

    def get_val_with_dot_path(self, dot_path: str):
        """
        Retrieve the value of an attribute using a dot-separated path string.

        Parameters
        ----------
        dot_path : str
            A dot-separated path string representing the attribute.

        Returns
        -------
        Any
            The value of the attribute specified by the dot-separated path string.

        Examples
        --------
        >>> config = MyConfig(a=1, b={"c": 2})
        >>> config.get_val_with_dot_path("b.c")
        2
        """
        return operator.attrgetter(dot_path)(self)

    def get_type_with_dot_path(self, dot_path: str):
        """
        Get the type of a value specified by a dot-separated path string.

        Parameters
        ----------
        dot_path : str
            A dot-separated path string representing the attribute.

        Returns
        -------
        type
            The type of the value specified by the dot-separated path string.

        """
        val = self.get_val_with_dot_path(dot_path)
        return type(val)

    def get_annot_type_with_dot_path(self, dot_path: str):
        """
        Get the type of an annotated variable specified by a dot-separated path string.

        Parameters
        ----------
        dot_path : str
            A dot-separated path string representing the annotated variable.

        Returns
        -------
        type
            The type of the annotated variable specified by the dot-separated path string.

        """
        *base_path, element = dot_path.split(".")
        annot_dot_path = ".".join(base_path + ["annotations"])
        annot: dict[str, Annotation] = self.get_val_with_dot_path(annot_dot_path)
        return annot[element].variable_type

    def make_dict(
        self,
        annotations: dict[str, Annotation],
        ignore_stateless=False,
        flatten=False,
    ):
        """
        Convert the configuration object to a dictionary representation.

        Parameters
        ----------
        annotations : dict[str, Annotation]
            A dictionary containing the annotations for each field of the configuration.
        ignore_stateless : bool, optional (default=False)
            If True, ignore stateless fields in the output dictionary.
        flatten : bool, optional (default=False)
            If True, flatten the nested dictionary structure into a single level dictionary.

        Returns
        -------
        dict
            A dictionary representation of the configuration object.
        """
        return_dict = {}
        for field_name, annot in annotations.items():
            if ignore_stateless and annot.state == Stateless:
                continue

            _val = getattr(self, field_name)
            if annot.collection is None or annot.collection in [Dict, List, Tuple]:
                val = _val
            elif annot.collection == Type:
                val = _val.__dict__
            elif issubclass(type(_val), ConfigBase):
                val = _val.make_dict(
                    _val.annotations, ignore_stateless=ignore_stateless, flatten=flatten
                )
            elif issubclass(type(_val), Enum):
                # _val: Enum
                val = _val.value

            else:
                raise NotImplementedError
            return_dict[field_name] = val
        if flatten:
            return_dict = flatten_nested_dict(return_dict)
        return return_dict

    def write(self, path: ty.Union[Path, str]):
        """
        Write the configuration object to a file.

        Parameters
        ----------
        path : ty.Union[Path, str]
            The path to the file where the configuration should be saved.
        """
        Path(path).write_text(str(self), encoding="utf-8")

    def to_str(self):
        """
        Convert the configuration object to a YAML string representation.

        Returns
        -------
        str
            A YAML string representation of the configuration object.
        """
        # TODO: investigate https://github.com/crdoconnor/strictyaml as an alternative to OmegaConf
        conf = OmegaConf.create(self.to_dict())
        return OmegaConf.to_yaml(conf)

    def assert_state(self, config: "ConfigBase") -> bool:
        """
        Compare the current configuration object with another, ensuring they are in the same state.

        Parameters
        ----------
        config : ConfigBase
            The configuration object to compare with.

        Returns
        -------
        bool
            True if both configurations are in the same state, otherwise raises an AssertionError.
        """
        diffs = sorted(self.diff_str(config, ignore_stateless=True))
        diff = "\n\t".join(diffs)
        assert len(diffs) == 0, f"Differences between configurations:\n\t{diff}"
        return True

    def merge(self, config: "ConfigBase") -> "ty.Self":  # type: ignore
        """
        Merge the current configuration object with another, replacing stateless and derived properties.

        Parameters
        ----------
        config : ConfigBase
            The configuration object to merge with.

        Returns
        -------
        ty.Self
            A new configuration object that is the result of merging the current object with the given one.
        """
        # TODO ty.Self is currently supported by mypy? fixme above
        # replaces stateless and derived properties
        self_config = copy.deepcopy(self)

        left_config = self_config
        right_config = copy.deepcopy(config)
        right_annotations = right_config.annotations
        left_annotations = right_config.annotations
        left_config.assert_state(right_config)
        right_config.assert_state(left_config)
        assert isinstance(left_config, type(right_config))

        for k in right_annotations:
            assert left_annotations[k] == right_annotations[k]
            if left_annotations[k].state in [Stateless, Derived]:
                right_val = getattr(right_config, k)
                setattr(left_config, k, right_val)

        return left_config

    def diff_str(self, config: "ConfigBase", ignore_stateless=False):
        """
        Get a list of differences between the current configuration object and another as strings.

        Parameters
        ----------
        config : ConfigBase
            The configuration object to compare with.
        ignore_stateless : bool, optional (default=False)
            If True, ignore stateless fields when comparing the configurations.

        Returns
        -------
        list[str]
            A list of differences between the configurations as strings.
        """
        diffs = self.diff(config, ignore_stateless=ignore_stateless)
        str_diffs = []
        for p, (l_t, l_v), (r_t, r_v) in diffs:
            _diff = f"{p}:({l_t.__name__}){l_v}->({r_t.__name__}){r_v}"
            str_diffs.append(_diff)
        return str_diffs

    def diff(
        self, config: "ConfigBase", ignore_stateless=False
    ) -> list[tuple[str, tuple[type, ty.Any], tuple[type, ty.Any]]]:
        """
        Get a list of differences between the current configuration object and another.

        Parameters
        ----------
        config : ConfigBase
            The configuration object to compare with.
        ignore_stateless : bool, optional (default=False)
            If True, ignore stateless fields when comparing the configurations.

        Returns
        -------
        list[tuple[str, tuple[type, ty.Any], tuple[type, ty.Any]]]
            A list of differences between the configurations as tuples.
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

    def to_dict(self, ignore_stateless=False):
        """
        Convert the configuration object to a dictionary representation.

        Parameters
        ----------
        ignore_stateless : bool, optional (default=False)
            If True, ignore stateless fields in the output dictionary.

        Returns
        -------
        dict
            A dictionary representation of the configuration object.
        """
        return self.make_dict(self.annotations, ignore_stateless=ignore_stateless)

    def to_yaml(self):
        """
        Convert the configuration object to a YAML string representation.

        Returns
        -------
        str
            A YAML string representation of the configuration object.
        """
        return str(self)

    def to_dot_path(self, ignore_stateless=False):
        """
        Convert the configuration object to a dictionary representation with dot-separated paths as keys.

        Parameters
        ----------
        ignore_stateless : bool, optional (default=False)
            If True, ignore stateless fields in the output dictionary.

        Returns
        -------
        str
            A YAML string representation of the configuration object with dot-separated paths as keys.
        """
        _flat_dict = self.make_dict(
            self.annotations, ignore_stateless=ignore_stateless, flatten=True
        )
        return OmegaConf.to_yaml(OmegaConf.create(_flat_dict))

    def __repr__(self) -> str:
        """
        Return the string representation of the configuration object.

        Returns
        -------
        str
            A YAML string representation of the configuration object.
        """
        return self.to_str()

    @property
    def uid(self):
        """
        Generate a unique identifier for the configuration object based on its content.

        Returns
        -------
        str
            A unique identifier (first 5 characters of the hash) for the configuration object.
        """
        return dict_hash(self.make_dict(self.annotations, ignore_stateless=True))[:5]

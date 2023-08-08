import copy
import inspect
import operator
import typing as ty
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf
from ablator.config.types import (
    Annotation,
    Derived,
    Dict,
    Enum,
    List,
    Stateless,
    Tuple,
    Type,
    Literal,
    parse_type_hint,
    parse_value,
)
from ablator.config.utils import dict_hash, flatten_nested_dict


def configclass(cls):
    """
    Decorator for ConfigBase subclasses, adds the ``config_class`` attribute to the class.

    Parameters
    ----------
    cls : Type[ConfigBase]
        The class to be decorated.

    Returns
    -------
    Type[ConfigBase]
        The decorated class with the ``config_class`` attribute.
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
    # NOTE: this allows for non-defined arguments to be created. It is very bug-prone and will be disabled.
    """

    This class this the building block for all configuration objects within ablator. It serves as the base class for
    configurations such as ``ModelConfig``, ``TrainConfig``, ``OptimizerConfig``, and more.

    To customize configurations for specific needs, you can create your own configuration class by inheriting from ``ConfigBase``.
    It's essential to annotate it with ``@configclass``. For instance, in the tutorial :ref:`Search space for different types
    of optimizers and scheduler <search_space_optim_schedule>`, a custom optimizer config class is created to enable ablation study on various optimizers
    and schedulers. You can refer to this tutorial for an example of how to create your custom configuration class.

    Examples
    --------

    >>> @configclass
    >>> class MyCustomConfig(ConfigBase):
    ...     attr1: int = 1
    ...     attr2: Tuple[str, int, str]

    Parameters
    ----------
    *args : Any
        Positional arguments.
    **kwargs : Any
        Keyword arguments.

    Attributes
    ----------
    config_class : Type
        The class of the configuration object.

    Raises
    ------
    ValueError
        If positional arguments are provided.
    KeyError
        If unexpected arguments are provided.

    .. note::
       All config class must be decorated with ``@configclass``

    """
    config_class = type(None)

    def __init__(self, *args, **kwargs):
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
                f"You must decorate your Config class '{class_name}' with ablator.configclass."
            )
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
        assert len(missing_vals) == 0, f"Missing required value {missing_vals}"
        for k, annotation in self.annotations.items():
            if k in kwargs:
                v = kwargs[k]
                del kwargs[k]
            else:
                v = getattr(self, k, None)

            v = parse_value(v, annotation, k)
            setattr(self, k, v)

        if len(kwargs) > 0:
            unspected_args = ", ".join(kwargs.keys())
            raise KeyError(f"Unexpected arguments: `{unspected_args}`")

    def keys(self):
        """
        Get the keys of the configuration dictionary.

        Returns
        -------
        KeysView[str]
            The keys of the configuration dictionary.
        """
        return self.to_dict().keys()

    @classmethod
    def load(cls, path: ty.Union[Path, str]):
        """
        Load a configuration object from a file.

        Parameters
        ----------
        path : Union[Path, str]
            The path to the configuration file.

        Returns
        -------
        ConfigBase
            The loaded configuration object.
        """
        kwargs: dict = OmegaConf.to_object(OmegaConf.create(Path(path).read_text(encoding="utf-8")))  # type: ignore
        return cls(**kwargs)

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
            annotation_types = dict(self.__annotations__)
            # pylint: disable=no-member
            # Without the if statement it will over-write new configurations
            # e.x.

            # class ReConfig(RunConfig):
            #     train_config: SomeTrainConfig = SomeTrainConfig()
            #     model_config: SomeModelConfig = SomeModelConfig()
            # TODO test-me

            dataclass_types = {
                k: v.type
                for k, v in self.__dataclass_fields__.items()
                if k not in annotation_types
            }
            annotation_types.update(dataclass_types)

            annotations = {
                field_name: parse_type_hint(type(self), annotation)
                for field_name, annotation in annotation_types.items()
            }
        return annotations

    def get_val_with_dot_path(self, dot_path: str):
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

    def get_type_with_dot_path(self, dot_path: str):
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

    def get_annot_type_with_dot_path(self, dot_path: str):
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
    ):
        """
        Create a dictionary representation of the configuration object.

        Parameters
        ----------
        annotations : dict[str, Annotation]
            A dictionary of annotations.
        ignore_stateless : bool, optional, default=False
            Whether to ignore stateless values.
        flatten : bool, optional, default=False
            Whether to flatten nested dictionaries.

        Returns
        -------
        dict
            The dictionary representation of the configuration object.
        """
        return_dict = {}

        def __parse_nested_value(val):
            if issubclass(type(val), Type):
                return val.__dict__
            if issubclass(type(val), ConfigBase):
                # TODO test-case for
                # val.make_dict(val.annotations) vs below
                return val.make_dict(
                    val.annotations, ignore_stateless=ignore_stateless, flatten=flatten
                )

            return val

        for field_name, annot in annotations.items():
            if ignore_stateless and (annot.state in {Stateless, Derived}):
                continue

            _val = getattr(self, field_name)
            if annot.collection in [None, Literal] or _val is None:
                val = _val
            elif annot.collection == List:
                val = [__parse_nested_value(_lval) for _lval in _val]
            elif annot.collection == Tuple:
                val = tuple(__parse_nested_value(_lval) for _lval in _val)
            elif annot.collection in [Dict]:
                val = {k: __parse_nested_value(_dval) for k, _dval in _val.items()}
            elif issubclass(type(_val), ConfigBase):
                val = _val.make_dict(
                    _val.annotations, ignore_stateless=ignore_stateless, flatten=flatten
                )

            elif annot.collection == Type:
                if annot.optional and _val is None:
                    val = None
                else:
                    val = _val.__dict__
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
        path : Union[Path, str]
            The path to the file.

        """
        Path(path).write_text(str(self), encoding="utf-8")

    def to_str(self):
        """
        Convert the configuration object to a string.

        Returns
        -------
        str
            The string representation of the configuration object.
        """
        # TODO: investigate https://github.com/crdoconnor/strictyaml as an alternative to OmegaConf
        conf = OmegaConf.create(self.to_dict())
        return OmegaConf.to_yaml(conf)

    def assert_state(self, config: "ConfigBase") -> bool:
        """
        Assert that the configuration object has a valid state.

        Parameters
        ----------
        config : ConfigBase
            The configuration object to compare.

        Returns
        -------
        bool
            ``True`` if the configuration object has a valid state, ``False`` otherwise.

        """
        diffs = sorted(self.diff_str(config, ignore_stateless=True))
        diff = "\n\t".join(diffs)
        assert len(diffs) == 0, f"Differences between configurations:\n\t{diff}"
        return True

    def merge(self, config: "ConfigBase") -> "ty.Self":  # type: ignore
        # TODO ty.Self is currently supported by mypy? fixme above
        # replaces stateless and derived properties
        """
        Merge the current configuration object with another configuration object.

        Parameters
        ----------
        config : ConfigBase
            The configuration object to merge.

        Returns
        -------
        ty.Self
            The merged configuration object.

        """
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

    def diff_str(self, config: "ConfigBase", ignore_stateless: bool = False):
        """
        Get the differences between the current configuration object and another configuration object as strings.

        Parameters
        ----------
        config : ConfigBase
            The configuration object to compare.
        ignore_stateless : bool, optional, default=False
            Whether to ignore stateless values.

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
    ) -> list[tuple[str, tuple[type, ty.Any], tuple[type, ty.Any]]]:
        """
        Get the differences between the current configuration object and another configuration object.

        Parameters
        ----------
        config : ConfigBase
            The configuration object to compare.
        ignore_stateless : bool, optional, default=False
            Whether to ignore stateless values.

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

    def to_dict(self, ignore_stateless: bool = False):
        """
        Convert the configuration object to a dictionary.

        Parameters
        ----------
        ignore_stateless : bool, optional, default=False
            Whether to ignore stateless values.

        Returns
        -------
        dict
            The dictionary representation of the configuration object.

        """
        return self.make_dict(self.annotations, ignore_stateless=ignore_stateless)

    def to_yaml(self):
        """
        Convert the configuration object to YAML format.

        Returns
        -------
        str
            The YAML representation of the configuration object.

        """
        return str(self)

    def to_dot_path(self, ignore_stateless: bool = False):
        """
        Convert the configuration object to a dictionary with dot notation paths as keys.

        Parameters
        ----------
        ignore_stateless : bool, optional, default=False
            Whether to ignore stateless values.

        Returns
        -------
        str
            The YAML representation of the configuration object in dot notation paths.

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
            The string representation of the configuration object.

        """
        return self.to_str()

    @property
    def uid(self):
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
        AssertionError
            If the configuration object is ambiguous or missing required values.

        """
        for k, annot in self.annotations.items():
            if not annot.optional:
                assert (
                    getattr(self, k) is not None
                ), f"Ambigious configuration. Must provide value for {k}"
            if (
                isinstance(annot.variable_type, type)
                and issubclass(annot.variable_type, ConfigBase)
                and getattr(self, k) is not None
            ):
                if annot.collection in [List, Tuple]:
                    for _lval in getattr(self, k):
                        _lval.assert_unambigious()
                elif annot.collection in [Dict]:
                    for _lval in getattr(self, k).values():
                        _lval.assert_unambigious()
                else:
                    getattr(self, k).assert_unambigious()

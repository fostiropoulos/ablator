from collections import namedtuple
import copy
import operator
import typing as ty
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import trainer.config.types as cty
from trainer.config.types import (
    ALLOWED_TYPES,
    ALLOWED_COLLECTIONS,
    Derived,
    Enum,
    Missing,
    Stateful,
    Stateless,
    is_optional,
    strip_type_hint,
)
from trainer.utils.config import flatten_nested_dict
from trainer.utils.file import dict_hash



def configclass(cls):
    assert issubclass(cls, ConfigBase), f"{cls.__name__} must inherit from ConfigBase"
    setattr(cls, "config_class", cls)
    return dataclass(cls, init=False, repr=False, kw_only=True)


@dataclass(repr=False)
class ConfigBase:
    # TODO: investigate https://github.com/crdoconnor/strictyaml as an alternative
    # NOTE: this allows for non-defined arguments to be created. It is very bug-prone and will be disabled.
    config_class = False

    def __init__(self, *args, add_attributes=False, **kwargs):
        class_name = type(self).__name__
        # TODO assert all variables are annotated with the types allowed.
        # TODO if Type assert via inspect that arguments are int, str, and float
        if len(args) > 0:
            raise ValueError(f"{class_name} does not support positional arguments.")
        if not (type(self) == self.config_class):
            raise RuntimeError(
                f"You must decorate your Config class '{class_name}' with trainer.configclass."
            )
        for k, annotation in self.annotations.items():
            required = is_optional(self._type_hints[k])
            value_type = self.value_types[k]

            if not required and not annotation is Derived:
                # make sure non-optional and derived values are not empty or
                # without a default assignment
                assert (k in kwargs and kwargs[k] is not None) or getattr(
                    self, k, None
                ) is not None, f"Missing required value {k}"
            if k in kwargs:
                v = kwargs[k]
                del kwargs[k]
            else:
                v = getattr(self, k)

            if required and v is None:
                setattr(self, k, None)
                continue
            type_hint = self._type_hints[k]
            v = self._parse_value(v, value_type, type_hint)

            setattr(self, k, v)

        if add_attributes and len(kwargs) > 0:
            setattr(self, k, v)
        elif len(kwargs) > 0:
            unspected_args = ", ".join(kwargs.keys())
            raise KeyError(f"Unexpected arguments: `{unspected_args}`")

    def keys(self):
        return self.to_dict().keys()

    def __getitem__(self, item):
        return self.to_dict()[item]

    @classmethod
    def _parse_annotation(cls, annotation) -> ty.Union[Stateful, Stateless, Derived]:
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
    [State, [Optional], TYPE, TypeArg=None]
    @classmethod
    def _parse_value_types(cls, annotation) -> ValueType:
        """bool is for required or not"""
        origin = ty.get_origin(annotation)
        if origin is None:
            return get_value_type(annotation)

        if origin in [Derived, Stateless]:
            assert len(annotation.__args__) == 1
            return cls._parse_value_types(annotation.__args__[0])
        # return ty.get_args(value_type)
        if origin == ty.Union and annotation._name == "Optional":
            args = ty.get_args(annotation)
            assert len(args) == 2
            val = cls._parse_value_types(args[0])
            kwargs = val._asdict()
            kwargs["required"] = False
            return ValueType(**kwargs)
        if origin in [cty.Dict, list]:
            args = ty.get_args(annotation)
            assert len(args) == 1
            assert args[0] in ALLOWED_TYPES, f"Invalid annotation: {annotation}"
            return get_value_type(args[0])
        elif isinstance(origin, cty.Tuple):
            args = ty.get_args(annotation)
            return ValueType(types=args, origin=cty.Tuple)
        elif issubclass(origin, cty.Type):
            return ValueType(types=ty.Any, origin=origin)
        raise NotImplementedError
    #     if not origin in ALLOWED_TYPES:
    #         raise RuntimeError(f"invalid annotation type {annotation}.")
    #     raise RuntimeError(f"invalid annotation type {annotation}.")
    #     pass
    #     # if Dict, union or Tuple return multiple annotations
    #     # if isinstance(value_type, (Derived, Stateless, ty.Optional)):
    #     #     assert (
    #     #         len(value_type.__args__) == 1
    #     #     ), f"{type(value_type)} annotations support only a single type assignment. Given: {value_type.__args__}"
    #     #     return cls._parse_value_types(value_type.__args__[0])

    #     # TODO assert not Optional[Derived[x]]
    #     # TODO assert Enum == EnumAttr
    #     # if not isinstance(value_type, Type) and value_type._name == "Optional":
    #     #     annotations = [Optional]
    #     #     # second index is always None on annotations
    #     #     value_types, sub_annotations = cls._parse_value_types(annotation.__args__[0])
    #     #     annotations += sub_annotations
    #     #     return value_types, annotations

    #     def flatten_value_type(value_type):
    #         if not isinstance(value_type, Type) and value_type._name == "List":
    #             return [List] + [flatten_value_type(value_type.__args__[0])]
    #         if not isinstance(value_type, Type) and value_type._name == "Dict":
    #             return [Dict] + [
    #                 (value_type.__args__[0], flatten_value_type(value_type.__args__[1]))
    #             ]
    #         if not isinstance(value_type, Type) and value_type._name == "Tuple":
    #             return [Tuple] + [
    #                 tuple([flatten_value_type(arg) for arg in value_type.__args__])
    #             ]
    #         return value_type

    #     value_types = [flatten_value_type(value_type) for value_type in [annotation]]
    #     return value_types

    # @classmethod
    # def _parse_value(cls, v, value_type: ValueType, type_hint):
    #     # stripped_type_hint = strip_type_hint(type_hint)
    #     if value_type.origin is None and issubclass(value_type.types, ConfigBase):
    #         v = cls._parse_class(value_type.types, v)
    #         return v
    #     elif value_type.origin is None and issubclass(type(value_type.types),  cty.Type):
    #         return cls._parse_class(value_type.types, v)
    #     elif value_type.origin is None:
    #         return value_type.types(v)

    #     if value_type.origin == cty.Literal:
    #         if v not in type_hint.__args__:
    #             raise ValueError(
    #                 f"{v} not found in {type_hint.__args__} for {value_type}"
    #             )
    #         # Skip type-check, it is already validated since it is in the list.
    #         return v

    #     elif issubclass(value_type.origin, cty.Enum):
    #         valid_values = [_v.value for _v in list(value_type.origin)]
    #         if v not in valid_values:
    #             raise ValueError(f"{v} not found in {valid_values} for {value_type}")
    #         return value_type.origin(v)

    #     elif value_type.origin == cty.Dict:
    #         return {str(_k): value_type.types(_v) for _k, _v in v.items()}

    #     elif value_type.origin == cty.List:
    #         return [value_type.types(_v) for _v in v]
    #     elif value_type.origin == cty.Tuple:
    #         assert len(v) == len(
    #             value_type.types
    #         ), f"Incompatible {v} and {value_type.types}"
    #         return [tp(_v) for tp, _v in zip(value_type.types, v)]
    #     elif isinstance(value_type.origin, cty.Type):
    #         return v

    #     elif value_type.origin is None:
    #         return value_type.types(v)
    #     else:
    #         raise NotImplementedError

    @classmethod
    def _parse_class(cls, field_type, v):

        if isinstance(v, field_type):
            # This is when initializing directly from config
            pass
        elif isinstance(v, dict):
            # This is when initializing from a dictionary
            v = field_type(**v)
        else:

            # Could also attempt to convert configbase to dict but
            # let's throw an error instead.
            raise NotImplementedError(
                f"Incompatible config classes {type(v)} and {field_type}."
            )
        return v

    @classmethod
    def load(cls, path: ty.Union[Path, str]):
        kwargs: Dict = OmegaConf.to_object(OmegaConf.create(Path(path).read_text()))  # type: ignore
        return cls(**kwargs)

    @property
    def annotations(self):
        annotations = {}
        if len(self.__dataclass_fields__):

            annotations = {
                field_name: self._parse_annotation(annotation)
                for field_name, annotation in self._type_hints.items()
            }
        return annotations

    # @property
    # def _type_hints(self):

    #     if hasattr(self, "__annotations__"):
    #         annotation_types = {k: v for k, v in self.__annotations__.items()}
    #     else:
    #         annotation_types = {}
    #     dataclass_types = {k: v.type for k, v in self.__dataclass_fields__.items()}
    #     annotation_types.update(dataclass_types)
    #     return annotation_types

    # @property
    # def value_types(self):
    #     annotations = {}
    #     if len(self.__dataclass_fields__):
    #         annotations = {
    #             field_name: self._parse_value_types(annotation)
    #             for field_name, annotation in self._type_hints.items()
    #         }
    #     return annotations

    def get_val_with_dot_path(self, dot_path):
        return operator.attrgetter(dot_path)(self)

    def get_type_with_dot_path(self, dot_path):
        val = self.get_val_with_dot_path(dot_path)
        # TODO Fixme. This will break because infering type for optional values will be troublesome. returns None.
        return type(val)
        # return operator.attrgetter(dot_path)(self)

    def _is_annotation_stateless(self, annotations):
        return any(
            [
                issubclass(annotation, Stateless)
                for annotation in annotations
                if isinstance(annotation, ty.Type)
            ]
        )

    def make_dict(self, annotations, ignore_stateless=False, flatten=False):
        return_dict = {}
        for field_name, (value_types, annotations) in annotations.items():

            if ignore_stateless and self._is_annotation_stateless(annotations):
                continue
            val = getattr(self, field_name)
            if issubclass(type(val), ConfigBase):
                val: ConfigBase
                val = val.make_dict(
                    val.annotations, ignore_stateless=ignore_stateless, flatten=flatten
                )
            elif issubclass(type(val), Enum):
                val: Enum
                val = val.value
            # TODO fix types allowed and forbid Enum!!!
            return_dict[field_name] = val
        if flatten:
            return_dict = flatten_nested_dict(return_dict)
        return return_dict

    def write(self, path: ty.Union[Path, str]):
        _repr = self.__repr__()
        Path(path).write_text(_repr)

    def to_str(self):
        conf = OmegaConf.create(self.to_dict())
        return OmegaConf.to_yaml(conf)

    def assert_state(self, config: "ConfigBase") -> bool:
        diffs = self.diff_str(config, ignore_stateless=True)
        diff = "\n\t".join(diffs)
        assert len(diffs) == 0, f"Differences between configurations:\n\t{diff}"

        return True

    def merge(
        self,
        config: "ConfigBase",
        how: cty.Literal["left", "union"] = "left",
        force=False,
    ) -> "ConfigBase":

        self_config = copy.deepcopy(self)

        left_config = self_config
        right_config = copy.deepcopy(config)
        for field_name, (value_types, annotations) in left_config.annotations.items():
            left_v = getattr(left_config, field_name)
            if not hasattr(right_config, field_name):
                continue

            config_v = getattr(right_config, field_name)

            is_subconfig = issubclass(type(left_v), ConfigBase) or isinstance(
                left_v, ConfigBase
            )

            if self._is_annotation_stateless(annotations) or (
                force and not is_subconfig
            ):
                setattr(left_config, field_name, config_v)
            elif is_subconfig or force:
                merged_subconfig = left_v.merge(config_v, force=force)
                setattr(left_config, field_name, merged_subconfig)
            else:
                assert (
                    config_v == left_v
                ), f"Different configuration values for {field_name}. Left config - {left_v}, Right config - {config_v}"

        if how == "union":
            left_config = right_config.merge(left_config, how="left", inplace=False)  # type: ignore

        return left_config  # type: ignore

    def diff_str(self, config: "ConfigBase", ignore_stateless=False):
        diffs = self.diff(config, ignore_stateless=ignore_stateless)
        str_diffs = []
        for p, (l_t, l_v), (r_t, r_v) in diffs:
            _diff = f"{p}:({l_t.__name__}){l_v}->({r_t.__name__}){r_v}"
            str_diffs.append(_diff)
        return str_diffs

    def diff(
        self, config: "ConfigBase", ignore_stateless=False
    ) -> ty.List[ty.Tuple[str, ty.Tuple[ty.Type, ty.Any], ty.Tuple[ty.Type, ty.Any]]]:
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
        diffs: ty.List[
            ty.Tuple[str, ty.Tuple[ty.Type, ty.Any], ty.Tuple[ty.Type, ty.Any]]
        ] = []
        for k in left_keys.union(right_keys):
            if k not in left_dict:
                right_v = right_dict[k]
                right_type = type(right_v)
                diffs.append((k, (Missing, None), (right_type, right_v)))

            elif k not in right_dict:
                left_v = left_dict[k]
                left_type = type(left_v)
                diffs.append((k, (left_type, left_v), (Missing, None)))

            elif left_dict[k] != right_dict[k] or type(left_dict[k]) != type(
                right_dict[k]
            ):
                right_v = right_dict[k]
                left_v = left_dict[k]
                left_type = type(left_v)
                right_type = type(right_v)
                diffs.append((k, (left_type, left_v), (right_type, right_v)))
        return diffs

    def to_dict(self, ignore_stateless=False):
        return self.make_dict(self.annotations, ignore_stateless=ignore_stateless)

    def to_yaml(self):
        return self.__repr__()

    def __repr__(self) -> str:
        return self.to_str()

    def attributes(self):
        model_class_attr = [
            attr
            for attr in list(self.__dict__.keys())
            if not (attr.startswith("__") and attr.endswith("__"))
        ] + list(self.__dataclass_fields__.keys())
        return list(np.unique(model_class_attr))

    @property
    def uid(self):
        return dict_hash(self.make_dict(self.annotations, ignore_stateless=True))[:5]

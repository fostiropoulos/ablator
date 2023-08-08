import typing as ty

from ablator.config.main import ConfigBase, configclass
from ablator.config.types import Annotation, Enum, List, Optional, Self, Tuple, Type


class SubConfiguration:
    def __init__(self, **kwargs) -> None:
        _search_space_annotations = list(SearchSpace.__annotations__.keys())

        def _parse_value(v):
            if isinstance(v, dict) and any(_k in v for _k in _search_space_annotations):
                return SearchSpace(**v)
            if isinstance(v, dict):
                return {k: _parse_value(v[k]) for k in v}
            return v

        self.arguments: dict[str, ty.Any] = _parse_value(kwargs)

    def __getitem__(self, item):
        return self.arguments[item]

    @property
    def __dict__(self):
        def _parse_nested_value(val):
            if issubclass(type(val), Type):
                return _parse_nested_value(val.__dict__)
            if issubclass(type(val), ConfigBase):
                return _parse_nested_value(
                    val.make_dict(
                        val.annotations, ignore_stateless=False, flatten=False
                    )
                )
            if isinstance(val, dict):
                return {k: _parse_nested_value(v) for k, v in val.items()}

            return val

        return {k: _parse_nested_value(v) for k, v in self.arguments.items()}

    def contains(self, value: dict[str, ty.Any]):
        def _contains_value(arguments, v):
            if isinstance(arguments, SearchSpace):
                return arguments.contains(v)
            if isinstance(v, dict) and not isinstance(arguments, dict):
                return False
            if isinstance(v, dict):
                return all(
                    _contains_value(arguments[k], v[k]) if k in arguments else False
                    for k in v
                )
            return v == arguments

        return _contains_value(arguments=self.arguments, v=value)


class FieldType(Enum):
    """
    Type of search space.
    """

    discrete = "int"
    continuous = "float"


@configclass
class SearchSpace(ConfigBase):
    """
    Search space configuration, required in parallel experiments, is used to define
    the search space for a hyperparameter.

    Examples
    --------

    In ablator, search space is defined for HPO that runs in parallel. For example, we want to
    run hyperparameter optimization on the model's hidden size and activation function:

    - Given the following model configuration:

    >>> @configclass
    >>> class CustomModelConfig(ModelConfig):
    >>>     hidden_size: int
    >>>     activation: str
    >>> my_model_config = CustomModelConfig(hidden_size=100, activation="relu")

    - The search space, which will be passed to ``ParallelConfig`` as a dictionary (notice how the
      key is expressed as ``model_config.<model-hyperparameter>``), should look like this:
    
    >>> search_space = {
    ...     "model_config.hidden_size": SearchSpace(value_range = [32, 64], value_type = 'int'),
    ...     "model_config.activation": SearchSpace(categorical_values = ["relu", "elu", "leakyRelu"])
    ... }

    """

    value_range: Optional[Tuple[str, str]]
    categorical_values: Optional[List[str]]
    subspaces: Optional[List[Self]]
    sub_configuration: Optional[SubConfiguration]
    value_type: Optional[FieldType]
    n_bins: Optional[int]
    log: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        nan_values = sum(
            [
                self.value_range is not None,
                self.categorical_values is not None,
                self.subspaces is not None,
                self.sub_configuration is not None,
            ]
        )
        assert nan_values == 1, (
            "Must specify only one of 'value_range', 'subspaces', "
            "'categorical_values' and / or 'sub_configurations' for SearchSpace."
        )
        if self.value_range is not None:
            assert (
                self.value_type is not None
            ), "`value_type` is required for `value_range` of SearchSpace"
        else:
            assert (
                self.value_type is None
            ), "Can not specify `value_type` without `value_range`."
        if self.n_bins is not None:
            assert (
                self.value_range is not None or self.categorical_values is not None
            ), "Can not specify `n_bins` without `value_range` or `categorical_values`."

    def parsed_value_range(self) -> tuple[int, int] | tuple[float, float]:
        assert self.value_range is not None
        assert self.value_type is not None
        fn = int if self.value_type == FieldType.discrete else float

        low_str, high_str = self.value_range
        low = fn(low_str)
        high = fn(high_str)
        assert min(low, high) == low, "`value_range` must be in the format of (min,max)"
        return low, high

    def make_dict(
        self,
        annotations: dict[str, Annotation],
        ignore_stateless: bool = False,
        flatten: bool = False,
    ):
        return_dict = super().make_dict(
            annotations=annotations, ignore_stateless=ignore_stateless, flatten=flatten
        )

        return return_dict

    def make_paths(self):
        paths = []

        def _traverse_dict(_dict, prefix):
            if not isinstance(_dict, dict):
                paths.append(prefix)

            elif "sub_configuration" not in _dict:
                for k, v in _dict.items():
                    _traverse_dict(v, prefix + [k])

            elif _dict["sub_configuration"] is not None:
                _traverse_dict(_dict["sub_configuration"], prefix)
            elif _dict["subspaces"] is not None:
                for _v in _dict["subspaces"]:
                    _traverse_dict(_v, prefix)
            else:
                paths.append(prefix)

        _traverse_dict(self.to_dict(), [])
        return list({".".join(p) for p in paths})

    def to_str(self):
        # TODO make me pretty (e.g. print in an indented format.)
        if self.value_range is not None:
            str_repr = f"SearchSpace(value_range={self.parsed_value_range()}"
            if self.value_type is not None:
                str_repr += f", value_type='{self.value_type.value}'"
            if self.n_bins is not None:
                str_repr += f", n_bins='{self.n_bins}'"
            str_repr += ")"
            return str_repr
        if self.categorical_values is not None:
            return f"SearchSpace(categorical_values={self.categorical_values})"
        if self.subspaces is not None:
            subspaces = ",".join([v.to_str() for v in self.subspaces])
            str_repr = f"SearchSpace(subspaces=[{subspaces}])"
            return str_repr
        if self.sub_configuration is not None:
            sub_config = self.sub_configuration.arguments
            str_repr = f"SearchSpace(sub_configuration={sub_config})"
            return str_repr

        return None

    def contains(self, value: float | int | str | dict[str, ty.Any]):
        if self.value_range is not None and isinstance(value, (int, float, str)):
            min_val, max_val = self.parsed_value_range()
            return float(value) >= min_val and float(value) <= max_val
        if self.value_range is not None:
            raise ValueError(f"Invalid value {value}.")
        if self.categorical_values is not None:
            return str(value) in self.categorical_values
        if self.subspaces is not None:
            return any(s.contains(value) for s in self.subspaces)
        if self.sub_configuration is not None and isinstance(value, dict):
            return self.sub_configuration.contains(value)

        return False

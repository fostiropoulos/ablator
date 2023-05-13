# from tests.configs.test_types import test_error_configs, test_hierarchical, test_types
import copy
from pathlib import Path
from collections import namedtuple

import pytest

from ablator import (
    Annotation,
    ConfigBase,
    Derived,
    Dict,
    Enum,
    Optional,
    Stateful,
    Stateless,
    Type,
    configclass,
)


class Pass:
    def __init__(self, a=10) -> None:
        self.a = a


@configclass
class SimpleConfig(ConfigBase):
    a1: int = 10


class myEnum(Enum):
    A = "a"


@configclass
class ParentTestConfig(ConfigBase):
    a1: int = 10
    a2: str = 10
    a8: Derived[Optional[str]] = 10

    a9: Derived[Optional[Dict[str]]]

    a5: Pass = Pass()
    c2: SimpleConfig = SimpleConfig()
    a10: Stateless[str]
    a6: myEnum = "a"


annotations = {
    "a1": Annotation(
        state=Stateful, optional=False, collection=None, variable_type=int
    ),
    "a2": Annotation(
        state=Stateful, optional=False, collection=None, variable_type=str
    ),
    "a8": Annotation(state=Derived, optional=True, collection=None, variable_type=str),
    "a9": Annotation(state=Derived, optional=True, collection=Dict, variable_type=str),
    "a10": Annotation(
        state=Stateless, optional=False, collection=None, variable_type=str
    ),
    "a5": Annotation(
        state=Stateful, optional=False, collection=Type, variable_type=Pass
    ),
    "a6": Annotation(
        state=Stateful, optional=False, collection=myEnum, variable_type=["a"]
    ),
    "c2": Annotation(
        state=Stateful, optional=False, collection=Type, variable_type=SimpleConfig
    ),
}


@configclass
class ParentLeftTestConfig(ConfigBase):
    a13: int = 10
    a14: str = 10


@configclass
class ParentRightTestConfig(ConfigBase):
    a13: int = 10
    a14: str = 10


@configclass
class ParentRightTestConfigDiff(ConfigBase):
    a13: int = 15
    a14: str = 16



#
# @configclass
# class MergedTestConfig(ConfigBase):


def test_merge():
    """
        Testing merge function when all the keys and values match in both configs

        *** I have few doubts in the implementation, i have implemented the
        test based on the actual implementation. ***
    """
    left_config = ParentLeftTestConfig()
    right_config = ParentRightTestConfig()
    assert left_config == left_config.merge(right_config), 'Merged configs are not equal'


def test_merge_diff_keys_values():
    """
        Testing merge function with different keys or values but of same configbase class.
    """
    left_config = ParentLeftTestConfig()
    right_config = ParentRightTestConfigDiff()
    with pytest.raises(AssertionError):
        assert left_config == left_config.merge(right_config)


def test_merge_diff_class():
    """
            Testing merge function when the right config of a different config class.
    """
    left_config = ParentLeftTestConfig()
    right_config = namedtuple('TestConfig', 'annotations')
    with pytest.raises(AssertionError):
        assert left_config == left_config.merge(right_config)


@configclass
class EmptyConfig(ConfigBase):
    pass


def test_attrs(tmp_path: Path):
    e = EmptyConfig()
    e.annotations
    assert len(e.annotations) == 0
    p = ParentTestConfig()
    # NOTE: to make new test cases:
    # print({k:f"Annotation(state={v.state.__name__},
    # optional={v.optional}, collection={v.collection},
    # variable_type={v.variable_type.__name__})"
    # for k,v in p.annotations.items()})
    for k, v in p.annotations.items():
        assert annotations[k] == v
    assert list(p.keys()) == list(p.annotations.keys())

    p.write(tmp_path.joinpath("test.yaml"))
    loaded_p = p.load(tmp_path.joinpath("test.yaml"))
    assert len(loaded_p.diff(p)) == 0
    assert loaded_p.uid == p.uid

    loaded_p.a10 = "a"
    assert len(loaded_p.diff(p)) == 1
    assert len(loaded_p.diff(p, ignore_stateless=True)) == 0
    assert loaded_p.uid == p.uid
    loaded_p.a2 = 0
    diffs = sorted(loaded_p.diff(p))
    var_name, (left_type, left_val), (right_type, right_val) = diffs[1]
    assert (
            var_name == "a2"
            and left_val == 0
            and left_type == int
            and right_type == str
            and right_val == "10"
    )
    loaded_p.c2.a1 = "a"
    diff_str = loaded_p.diff_str(p)
    assert len(diff_str) == 3 and [
        "a10:(str)a->(NoneType)None",
        "a2:(int)0->(str)10",
        "c2.a1:(str)a->(int)10",
    ] == sorted(diff_str)

    assert (
            loaded_p.to_dot_path()
            == "a1: 10\na2: 0\na8: '10'\na9: null\na10: a\na6: a\na5.a: 10\nc2.a1: a\n"
    )
    # TODO do we want them immutable?
    assert loaded_p.get_val_with_dot_path("c2.a1") == "a"
    assert loaded_p.get_type_with_dot_path("c2.a1") == str
    assert loaded_p.get_annot_type_with_dot_path("c2.a1") == int
    assert p.get_val_with_dot_path("c2.a1") == 10
    assert p.get_type_with_dot_path("c2.a1") == int

    try:
        loaded_p = loaded_p.merge(p)
        assert False
    except Exception as e:
        assert (
                str(e)
                == "Differences between configurations:\n\ta2:(int)0->(str)10\n\tc2.a1:(str)a->(int)10"
        )
    p_prime = copy.deepcopy(loaded_p)
    p_prime.a10 = 1000
    loaded_p.a10 = "a"
    loaded_p = loaded_p.merge(p_prime)

    assert loaded_p.a10 == 1000


if __name__ == "__main__":
    test_attrs(Path("/tmp/"))

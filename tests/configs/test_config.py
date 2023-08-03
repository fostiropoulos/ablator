# from tests.configs.test_types import test_error_configs, test_hierarchical, test_types
import copy
from pathlib import Path

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


@configclass
class ParentTestConfig2(ParentTestConfig):
    a10: Stateless[int] = 10


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


def test_merge():
    pass


@configclass
class EmptyConfig(ConfigBase):
    pass


def test_attrs(tmp_path: Path, assert_error_msg):
    e = EmptyConfig()
    e.annotations
    assert len(e.annotations) == 0
    assert_error_msg(ParentTestConfig, "Missing required value ['a10']")
    p = ParentTestConfig(a10="")
    # NOTE: to make new test cases:
    # print({k:f"Annotation(state={v.state.__name__},
    # optional={v.optional}, collection={v.collection},
    # variable_type={v.variable_type.__name__})"
    # for k,v in p.annotations.items()})
    for k, v in p.annotations.items():
        assert annotations[k] == v
    assert list(p.keys()) == list(p.annotations.keys())
    p.a10 = "10"
    p.write(tmp_path.joinpath("test.yaml"))
    loaded_p = p.load(tmp_path.joinpath("test.yaml"))
    assert len(loaded_p.diff(p)) == 0
    assert loaded_p.uid == p.uid
    loaded_p = ParentTestConfig2.load(tmp_path.joinpath("test.yaml"))
    assert len(loaded_p.diff(p)) == 1
    assert len(loaded_p.diff(p, ignore_stateless=True)) == 0
    assert loaded_p.uid == p.uid
    loaded_p.a10 = 2
    diffs = sorted(loaded_p.diff(p))
    var_name, (left_type, left_val), (right_type, right_val) = diffs[0]
    assert (
        var_name == "a10"
        and left_val == 2
        and left_type == int
        and right_type == str
        and right_val == "10"
    )
    loaded_p.c2.a1 = 4
    diff_str = loaded_p.diff_str(p)
    assert len(diff_str) == 2 and [
        "a10:(int)2->(str)10",
        "c2.a1:(int)4->(int)10",
    ] == sorted(diff_str)

    assert (
        loaded_p.to_dot_path()
        == "a10: 2\na1: 10\na2: '10'\na8: '10'\na9: null\na6: a\na5.a: 10\nc2.a1: 4\n"
    )
    # TODO do we want them immutable?
    assert loaded_p.get_val_with_dot_path("a10") == 2
    assert loaded_p.get_type_with_dot_path("a10") == int
    assert loaded_p.get_annot_type_with_dot_path("a10") == int
    assert p.get_val_with_dot_path("a10") == "10"
    assert p.get_type_with_dot_path("a10") == str
    msg = assert_error_msg(lambda: loaded_p.merge(p))
    assert msg == "Differences between configurations:\n\tc2.a1:(int)4->(int)10"

    p_prime = copy.deepcopy(loaded_p)
    p_prime.a10 = 1000
    loaded_p.a10 = 2
    loaded_p = loaded_p.merge(p_prime)

    assert loaded_p.a10 == 1000


def test_set_attr(assert_error_msg):
    c = ParentTestConfig(a10="")
    c.a2 = 1.231
    assert isinstance(c.a2, str) and c.a2 == "1.231"

    c.a1 = 1.999
    assert isinstance(c.a1, int) and c.a1 == 1

    def _error():
        c.a5 = 0

    msg = assert_error_msg(_error)
    assert msg == f"Incompatible kwargs <class 'int'>: 0\nand {Pass}."
    c.a5 = {"a": 1}
    assert c.a5.a == 1
    c.a5 = {"a": 5}
    assert c.a5.a == 5


if __name__ == "__main__":
    from tests.conftest import _assert_error_msg

    test_set_attr(_assert_error_msg)
    test_attrs(Path("/tmp/"), _assert_error_msg)

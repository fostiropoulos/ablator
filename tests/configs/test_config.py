import copy
import io
import logging
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


@configclass
class ParentTestConfig3(ParentTestConfig):
    a6: Pass = Pass()
    a2: int = 5
    a1: str = "5"
    a10: str = "10"


@configclass
class ParentTestConfig4(ParentTestConfig3):
    b1: Pass


@configclass
class NestedParentConfig(ConfigBase):
    b1: ParentTestConfig4
    a1: str


@configclass
class NestedParentConfig2(ConfigBase):
    b1: ParentTestConfig3


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
    assert_error_msg(ParentTestConfig, "Missing required values ['a10'].")
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
    p.freeze()
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


def test_freeze_unfreeze(assert_error_msg):
    c = ParentTestConfig(a10="")
    c.freeze()

    def _set():
        c.a10 = "1"

    msg = assert_error_msg(_set)
    assert (
        msg == "Can not set attribute a10 on frozen configuration ``ParentTestConfig``."
    )
    c._unfreeze()
    c.a10 = "1"
    c.freeze()
    msg = assert_error_msg(_set)
    assert (
        msg == "Can not set attribute a10 on frozen configuration ``ParentTestConfig``."
    )
    # this should not be allowed.

    def _set():
        c.a5.a = 0

    msg = assert_error_msg(_set)
    assert (
        msg == "Can not set attribute a on a class of a frozen configuration ``Pass``."
    )

    def _set():
        c.c2.a1 = 0

    # TODO context
    # with assert_error_msg:
    msg = assert_error_msg(_set)
    assert msg == "Can not set attribute a1 on frozen configuration ``SimpleConfig``."
    c._unfreeze()

    c.a5.a = 54
    assert c.a5.a == 54

    c.c2.a1 = 52
    assert c.c2.a1 == 52


def test_debug_load(tmp_path: Path, assert_error_msg):
    out = io.StringIO()
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(out))
    ParentTestConfig(debug=True)
    last_line = out.getvalue().split("\n")[-2]
    assert (
        last_line
        == "Loading ParentTestConfig in `debug` mode. Setting missing required value a10 to `None`."
    )
    p = ParentTestConfig(debug=True, a123=123, a543=543)
    last_line = out.getvalue().split("\n")[-2]
    assert (
        last_line
        == "Loading ParentTestConfig in `debug` mode. Ignoring unexpected arguments: `a123, a543`"
    )
    yaml_p = tmp_path.joinpath("test.yaml")
    p.write(yaml_p)
    msg = assert_error_msg(lambda: p.load(yaml_p))
    assert msg == "Missing required values ['a10']."
    p.load(yaml_p, debug=True)
    last_line = out.getvalue().split("\n")[-2]
    assert (
        last_line
        == "Loading ParentTestConfig in `debug` mode. Setting missing required value a10 to `None`."
    )
    ParentTestConfig4.load(yaml_p, debug=True)
    args = out.getvalue().split("\n")[-4:-1]
    msgs = [
        "Loading ParentTestConfig4 in `debug` mode. Setting missing required value b1 to `None`.",
        "Loading ParentTestConfig4 in `debug` mode. Unable to parse `a10` value None. Setting to `None`.",
        "Loading ParentTestConfig4 in `debug` mode. Unable to parse `a6` value a. Setting to `None`.",
    ]
    assert all(msg in args for msg in msgs)
    msg = assert_error_msg(lambda: ParentTestConfig4.load(yaml_p, debug=False))
    assert msg == "Missing required values ['b1']."
    msg = assert_error_msg(lambda: ParentTestConfig3.load(yaml_p, debug=False))
    assert (
        msg == f"Incompatible kwargs <class 'str'>: a\nand <class '{__name__}.Pass'>."
    )
    pconfig_3 = ParentTestConfig3.load(yaml_p, debug=True)
    pconfig_4 = ParentTestConfig4.load(yaml_p, debug=True)
    # Testing nested configs.
    nested_c = NestedParentConfig(b1=pconfig_4, a1="")
    msg = assert_error_msg(lambda: NestedParentConfig(b1=pconfig_3, a1=""))
    assert (
        msg
        == f"Incompatible kwargs <class '{__name__}.ParentTestConfig3'>: a6: null\na2: 10\na1: '10'\na10: null\na8: '10'\na9: null\na5:\n  a: 10\nc2:\n  a1: 10\n\nand <class '{__name__}.ParentTestConfig4'>."
    )

    nested_c = NestedParentConfig(b1=pconfig_3, a1="", debug=True)
    assert (
        "\n".join(out.getvalue().split("\n")[-12:-1])
        == "Loading NestedParentConfig in `debug` mode. Unable to parse `b1` value a6: null\na2: 10\na1: '10'\na10: null\na8: '10'\na9: null\na5:\n  a: 10\nc2:\n  a1: 10\n. Setting to `None`."
    )
    nested_c = NestedParentConfig(b1=pconfig_4, a1="")
    assert nested_c.b1.a6 is None
    nested_c.write(yaml_p)
    msg = assert_error_msg(lambda: NestedParentConfig2.load(yaml_p))
    assert msg == "Missing required value for a6."
    nested_c2 = NestedParentConfig2.load(yaml_p, debug=True)
    assert nested_c2.b1.a6 is None


if __name__ == "__main__":
    from tests.conftest import _assert_error_msg

    tmp_path = Path("/tmp/xxx")
    tmp_path.mkdir(exist_ok=True)
    test_debug_load(tmp_path, _assert_error_msg)
    test_freeze_unfreeze(_assert_error_msg)
    test_set_attr(_assert_error_msg)
    test_attrs(Path("/tmp/"), _assert_error_msg)

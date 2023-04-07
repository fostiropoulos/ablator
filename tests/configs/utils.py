from trainer.config.run import ConfigBase

from trainer.config.main import configclass
from trainer import Derived, Stateless
import typing as ty

from trainer.config.types import Dict, Enum, List, Literal, Optional


@configclass
class TestConfig(ConfigBase):
    a1: int = 10


@configclass
class ParentTestConfig(ConfigBase):
    a1: int = 10
    c: TestConfig
    c2: TestConfig
    a2: str = 10


@configclass
class ParentTestTestConfig(ConfigBase):
    # TODO fix mypy issue
    a1: Derived[int] = 10
    c: ParentTestConfig


@configclass
class ErrorConfig(ConfigBase):
    # TODO error for duplicate attrs?
    a1: int = 10
    # TODO error for duplicate attrs?
    a2: int = 10

    a1: Derived[ty.Literal["a", "b", "c"]] = 10
    a3: Derived[ty.Optional[str]] = 10
    # Should throw an error for Optional[Derived]
    a4: ty.Optional[Derived[str]] = "a"
    a4: List[Derived[str]] = "a"


class myEnum(Enum):
    A = "a"


class Pass:
    def __init__(self, a) -> None:
        self.a = a


@configclass
class MultiTypeConfig(ConfigBase):
    # TODO error for duplicate attrs?
    a1: int = 10
    # TODO error for duplicate attrs?
    a2: int = 10
    # a5 = 10

    a0: Derived[Literal["a", "b", "2"]] = 10
    a9: Derived[Dict[str]] = 10
    a8: Derived[Optional[str]] = 10
    # Should throw an error for Optional[Derived]
    a4: Optional[Derived[str]] = "a"
    a5: Pass

    # c2: TestConfig = TestConfig()
    # a4: List[str] = "a"
    # a6: myEnum = "a"


@configclass
class EmptyConfig(ConfigBase):
    pass


# TODO test annotations optional, derived, stateless, Dict list, Tuple, Literal, Specific Type, Enum
# TODO merge test
# TODO diff test
# TODO uid test
if __name__ == "__main__":
    # e = EmptyConfig()
    # e.annotations
    e = MultiTypeConfig( a5={"a":1})
    e.annotations
    e = ErrorConfig()
    e.annotations
    c = TestConfig(a1="10")
    assert type(c.a1) == int and c.a1 == int("10")
    # Should fail
    # pc = ParentTestConfig(0,c,c,0)
    # Should not fail
    pc = ParentTestConfig(a1=0, c=c, c2=c)
    assert pc.a2 == str(10), "Could not cast"
    pc.c.a1 = 2
    assert pc.c2.a1 == pc.c.a1, "Lost reference"
    pc = ParentTestConfig(a1=0, c={"a1": 10}, c2={"a1": "2"}, a2=0)
    assert type(pc.c) == TestConfig
    assert pc.c2.a1 == 2
    pc_dict = ParentTestTestConfig(c=pc.to_dict())
    pc_obj = ParentTestTestConfig(c=pc)
    assert pc_dict == pc_obj
    print(pc3)

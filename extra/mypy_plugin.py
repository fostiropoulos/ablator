from mypy.plugin import *
from mypy.types import TupleType,UnionType,NoneType,AnyType,TypeOfAny
from mypy import errorcodes
prefix="ablator.config.types"
class TestPlugin(Plugin):
    def get_type_analyze_hook(self, fullname: str) -> Callable[[AnalyzeTypeContext], Type] | None:
        if(fullname==f"{prefix}.Optional"):
            return Optionalcallback
        elif(fullname==f"{prefix}.Tuple"):
            return Tuplecallback
        elif(fullname==f"{prefix}.Stateless" or fullname==f"{prefix}.Derived" or fullname==f"{prefix}.Stateful"):
            return StatesCallback
        elif(fullname==f"{prefix}.Literal"):
            return LiteralCallback


def Optionalcallback(ctx: AnalyzeTypeContext) -> Type:
    t=ctx.type
    if(len(t.args)==1):
        item=ctx.api.anal_type(t.args[0])
        return UnionType([item,NoneType()],t.line,t.column)
    ctx.api.fail(f"mOptional should only has 1 arg got {len(t.args)}",ctx.context,code=errorcodes.VALID_TYPE)
    return AnyType(TypeOfAny.from_error)
def Tuplecallback(ctx: AnalyzeTypeContext) -> Type:
    t=ctx.type
    any_type=AnyType(TypeOfAny.special_form)
    items=ctx.api.anal_array(t.args)
    fallbackTyp=ctx.api.named_type("builtins.tuple",[any_type])
    return TupleType(items,fallback=fallbackTyp)
def StatesCallback(ctx: AnalyzeTypeContext) -> Type:
    t=ctx.type
    if(len(t.args)==1):
        return ctx.api.anal_type(t.args[0])
    ctx.api.fail(f"States should only has 1 arg got {len(t.args)}",ctx.context,code=errorcodes.VALID_TYPE)
    return AnyType(TypeOfAny.from_error)
def LiteralCallback(ctx:AnalyzeTypeContext)->Type:
    return ctx.api.analyze_literal_type(ctx.type)
def plugin(version: str):
    return TestPlugin
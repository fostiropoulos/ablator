from mypy.plugin import *
from mypy.types import TupleType, UnionType, NoneType, AnyType, TypeOfAny
from mypy import errorcodes
from mypy.typevars import fill_typevars

prefix = "ablator.config.types"


class AblatorPlugin(Plugin):
    def get_type_analyze_hook(
        self, fullname: str
    ) -> Callable[[AnalyzeTypeContext], Type] | None:
        if fullname == f"{prefix}.Optional":
            return Optionalcallback
        elif fullname == f"{prefix}.Tuple":
            return Tuplecallback
        elif (
            fullname == f"{prefix}.Stateless"
            or fullname == f"{prefix}.Derived"
            or fullname == f"{prefix}.Stateful"
        ):
            return StatesCallback
        elif fullname == f"{prefix}.Literal":
            return LiteralCallback
        elif fullname == f"{prefix}.Self":
            return SelfCallback


def Optionalcallback(ctx: AnalyzeTypeContext) -> Type:
    t = ctx.type
    if len(t.args) == 1:
        item = ctx.api.anal_type(t.args[0])
        return UnionType([item, NoneType()], t.line, t.column)
    ctx.api.fail(
        f"Optional should only has 1 arg got {len(t.args)}",
        ctx.context,
        code=errorcodes.VALID_TYPE,
    )
    return AnyType(TypeOfAny.from_error)


def Tuplecallback(ctx: AnalyzeTypeContext) -> Type:
    t = ctx.type
    any_type = AnyType(TypeOfAny.special_form)
    items = ctx.api.anal_array(t.args)
    fallbackTyp = ctx.api.named_type("builtins.tuple", [any_type])
    return TupleType(items, fallback=fallbackTyp)


def StatesCallback(ctx: AnalyzeTypeContext) -> Type:
    t = ctx.type
    if len(t.args) == 1:
        return ctx.api.anal_type(t.args[0])
    ctx.api.fail(
        f"States should only has 1 arg got {len(t.args)}",
        ctx.context,
        code=errorcodes.VALID_TYPE,
    )
    return AnyType(TypeOfAny.from_error)


def LiteralCallback(ctx: AnalyzeTypeContext) -> Type:
    return ctx.api.analyze_literal_type(ctx.type)


def SelfCallback(ctx: AnalyzeTypeContext) -> Type:
    t = ctx.type
    ctx.api.api.setup_self_type()
    if t.args:
        ctx.api.fail("Self type cannot have type arguments", t)
    if ctx.api.prohibit_self_type is not None:
        ctx.api.fail(f"Self type cannot be used in {ctx.api.prohibit_self_type}", t)
        return AnyType(TypeOfAny.from_error)
    if ctx.api.api.type is None:
        ctx.api.fail("Self type is only allowed in annotations within class definition", t)
        return AnyType(TypeOfAny.from_error)
    if ctx.api.api.type.has_base("builtins.type"):
        ctx.api.fail("Self type cannot be used in a metaclass", t)
    if ctx.api.api.type.self_type is not None:
        return fill_typevars(ctx.api.api.type)
    ctx.api.fail("Unexpected Self type", t)
    return AnyType(TypeOfAny.from_error)


def plugin(version: str):
    return AblatorPlugin

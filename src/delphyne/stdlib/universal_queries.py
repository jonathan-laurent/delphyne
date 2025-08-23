"""
Universal queries.
"""

import importlib
import inspect
import pathlib
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, assert_never, overload, override

import varname

import delphyne.core as dp
from delphyne.stdlib.nodes import Branch, Fail, branch, fail
from delphyne.stdlib.policies import IPDict
from delphyne.stdlib.queries import Query, last_code_block
from delphyne.utils.typing import TypeAnnot, pydantic_dump, pydantic_load


@dataclass
class UniversalQuery(Query[object]):
    """
    A universal query defined by the context surrounding its call.

    Attributes:
        strategy: Fully qualified name of the surrounding strategy
            (e.g., `my_package.my_module.my_strategy`).
        tags: Tags associated with the space induced by the query, which
            can be used to locate the exact location where the query is
            issued (the default tag takes the name of the variable that
            the query result is assigned to).
        locals: A dictionary that provides the values of a subset of
            local variables (as JSON values).

    !!! warning "Experimental"
        This feature is experimental and subject to change.
    """

    # TODO: add a `context` field where we store objects whose
    # documentation or source should be added to the prompt.

    strategy: str
    tags: Sequence[str]
    locals: dict[str, object]

    __parser__ = last_code_block.yaml

    @override
    def default_tags(self):
        return self.tags

    @property
    def strategy_source(self) -> str:
        """
        Return the source code of the strategy that contains this query.
        """
        strategy_obj = _load_from_qualified_name(self.strategy)
        assert callable(strategy_obj)
        return _source_code(strategy_obj)


@overload
def guess[T](
    annot: type[T], /, using: Sequence[object]
) -> dp.Strategy[Branch | Fail, IPDict, T]: ...


@overload
def guess(
    annot: TypeAnnot[Any], /, using: Sequence[object]
) -> dp.Strategy[Branch | Fail, IPDict, Any]: ...


def guess(
    annot: TypeAnnot[Any], /, using: Sequence[object]
) -> dp.Strategy[Branch | Fail, IPDict, Any]:
    """
    Attempt to guess a value of a given type, using the surrounding
    context of the call site along with the value of some local
    variables.

    !!! note
        Our use of an overloaded type should not be necessary anymore
        when `TypeExpr` is released with Python 3.14.
    """

    # Building the query
    ret_val_name = varname.varname(frame=1, strict=False)  # type: ignore
    assert isinstance(ret_val_name, str)
    strategy = _surrounding_qualname(skip=1)
    assert strategy is not None
    strategy = _rename_main_module_in_qualified_name(strategy)
    locals_list_name = varname.argname("using", vars_only=False)  # type: ignore
    assert isinstance(locals_list_name, str)
    local_names = _parse_list_of_ids(locals_list_name)
    assert len(using) == len(local_names)
    locals = {k: pydantic_dump(type(v), v) for k, v in zip(local_names, using)}
    query = UniversalQuery(strategy, [ret_val_name], locals)

    ret = yield from branch(query.using(...))
    try:
        parsed = pydantic_load(annot, ret)
    except Exception as e:
        assert_never((yield from fail("parse_error", message=str(e))))
    return parsed


#####
##### Inspection Utilities
#####


def _parse_list_of_ids(s: str) -> list[str]:
    """
    Take as an input a string representing a python list of python
    identifiers (e.g., "[foo, bar]") and return a list all identifiers.
    """
    # Remove brackets and whitespace
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    # Split by comma and strip whitespace
    return [item.strip() for item in s.split(",") if item.strip()]


def _rename_main_module_in_qualified_name(name: str) -> str:
    comps = name.split(".")
    assert len(comps) > 1
    if comps[0] == "__main__":
        # Rename __main__ to the actual module name if possible
        main_mod = sys.modules.get("__main__")
        if main_mod:
            mod_file = getattr(main_mod, "__file__", None)
            assert mod_file is not None
            path = pathlib.Path(mod_file)
            comps[0] = path.stem
    return ".".join(comps)


def _load_from_qualified_name(name: str) -> object:
    components = name.split(".")
    module_name = ".".join(components[:-1])
    attr_name = components[-1]
    mod = importlib.import_module(module_name)
    return getattr(mod, attr_name)


def _source_code(f: Callable[..., Any]) -> str:
    return inspect.getsource(f)


def _surrounding_qualname(skip: int = 1) -> str | None:
    """
    Return the qualified name of the surrounding function by inspecting
    the stack.

    Args:
        skip: how many call frames to skip:
              0 -> this function (not useful), 1 -> caller (default),
              2 -> caller's caller, etc.

    Returns:
        A string like 'pkg.mod.Class.method' or
        'pkg.mod.outer.<locals>.inner'. Returns None if we’re at module
        level (no surrounding function).
    """
    frame = inspect.currentframe()
    # Move up skip+1 frames: current -> caller is 1 hop
    for _ in range(skip + 1):
        if frame is None:
            return None
        frame = frame.f_back

    if frame is None:
        return None

    code = frame.f_code
    # If we're at module level, there is no surrounding function
    if code.co_name == "<module>":
        return None

    # Module name (may be None in rare cases)
    mod = inspect.getmodule(frame)
    modname = mod.__name__ if mod and hasattr(mod, "__name__") else None
    qual = getattr(code, "co_qualname", None)
    return f"{modname}.{qual}" if modname else qual

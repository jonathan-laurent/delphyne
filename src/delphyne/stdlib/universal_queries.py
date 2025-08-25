"""
Universal queries.
"""

import ast
import importlib
import inspect
import linecache
import pathlib
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, assert_never, overload, override

import delphyne.core as dp
from delphyne.stdlib.nodes import Branch, Fail, branch, fail
from delphyne.stdlib.policies import IPDict
from delphyne.stdlib.queries import Query, last_code_block
from delphyne.utils.typing import TypeAnnot, pydantic_dump, pydantic_load


@dataclass
class UniversalQuery(Query[object]):
    """
    A universal query, implicitly defined by the surrounding context of
    its call. See `guess` for more information.

    Attributes:
        strategy: Fully qualified name of the surrounding strategy
            (e.g., `my_package.my_module.my_strategy`).
        expected_type: A string rendition of the expected answer type.
        tags: Tags associated with the space induced by the query, which
            can be used to locate the exact location where the query is
            issued (the default tag takes the name of the variable that
            the query result is assigned to).
        locals: A dictionary that provides the values of a subset of
            local variables or expressions (as JSON values).

    !!! warning "Experimental"
        This feature is experimental and subject to change.
    """

    # TODO: add a `context` field where we store objects whose
    # documentation or source should be added to the prompt.

    strategy: str
    expected_type: str
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
    annot: type[T], /, *, using: Sequence[object]
) -> dp.Strategy[Branch | Fail, IPDict, T]: ...


@overload
def guess(
    annot: TypeAnnot[Any], /, *, using: Sequence[object]
) -> dp.Strategy[Branch | Fail, IPDict, Any]: ...


def guess(
    annot: TypeAnnot[Any], /, *, using: Sequence[object]
) -> dp.Strategy[Branch | Fail, IPDict, Any]:
    """
    Attempt to guess a value of a given type, using the surrounding
    context of the call site along with the value of some local
    variables or expressions.

    This function inspects the call stack to determine the context in
    which it is called and issues a `UniversalQuery`, with a tag
    corresponding to the name of the assigned variable. A failure node is
    issued if the oracle result cannot be parsed into the expected type.
    For example:

    ```python
    res = yield from guess(int, using=[x, y.summary()])
    ```

    issues a `UniversalQuery` query tagged `res`, with attribute
    `locals` a dictionary with string keys `"x"` and `"y.summary()"`.

    Attributes:
        annot: The expected type of the value to be guessed.
        using: A sequence of local variables or expressions whose value
            should be communicated to the oracle (a label for each
            expression is automatically generated using source information).

    !!! note
        Our use of an overloaded type should not be necessary anymore
        when `TypeExpr` is released with Python 3.14.

    !!! warning "Experimental"
        This feature is experimental and subject to change.
    """

    # Extracting the name of the surrounding strategy
    strategy = surrounding_qualname(skip=1)
    assert strategy is not None
    strategy = _rename_main_module_in_qualified_name(strategy)

    # Extracting the name of the variable being assigned
    cur_instr_src = current_instruction_source(skip=1)
    ret_val_name = assigned_var_name(cur_instr_src)
    assert isinstance(ret_val_name, str)

    # Computing the 'locals' dictionary
    guess_args = call_argument_sources(cur_instr_src, guess)
    assert guess_args is not None
    _args, kwargs = guess_args
    using_args = _parse_list_of_ids(kwargs["using"])
    assert len(using) == len(using_args)
    locals = {k: pydantic_dump(type(v), v) for k, v in zip(using_args, using)}

    # Building the query
    query = UniversalQuery(strategy, str(annot), [ret_val_name], locals)

    ret = yield from branch(query.using(...))
    try:
        parsed = pydantic_load(annot, ret)
    except Exception as e:
        assert_never((yield from fail("parse_error", message=str(e))))
    return parsed


#####
##### Utilities
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


#####
##### General Stack Inspection Utilities
#####


def surrounding_qualname(skip: int = 1) -> str | None:
    """
    Return the qualified name of the surrounding function by inspecting
    the stack.

    Arguments:
        skip: how many call frames to skip: 0 for this function (not
            useful), 1 for the caller, 2 for the caller's caller, etc.

    Returns:
        A string like 'pkg.mod.function'.
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


def current_instruction_source(skip: int = 0) -> str:
    """
    Return the source text of the currently executing *statement* for a
    frame `skip` levels up the call stack (0 = caller of this function).
    In generators/coroutines, this correctly shows the suspended
    caller’s line.
    """
    # +1 to skip this helper's own frame
    frame = sys._getframe(1 + skip)  # type: ignore
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    # Warm up linecache so we can read source quickly
    linecache.checkcache(filename)
    # Try to assemble the full logical statement starting at current line
    stmt = _collect_statement_lines(filename, lineno)
    if stmt:
        return stmt
    # Fallback: just the single line
    return linecache.getline(filename, lineno).strip()


def _collect_statement_lines(filename: str, start_lineno: int):
    """
    Return the source text for the full logical statement that begins on
    start_lineno. Handles implicit (brackets) and explicit (backslash)
    continuations.
    """
    lines: list[str] = []
    lineno: int = start_lineno
    # Track bracket balance: (), [], {}
    opens = {"(": ")", "[": "]", "{": "}"}
    closes = {")", "]", "}"}
    balance = 0
    cont = False

    while True:
        line = linecache.getline(filename, lineno)
        if not line:
            break  # EOF or unavailable
        # Always include at least the first line
        if not lines:
            lines.append(line)
        else:
            # Keep adding lines while we're in a continuation
            lines.append(line)

        # Simple bracket-balance heuristic ignoring strings/comments
        # (good enough for most cases)
        for ch in line:
            if ch in opens:
                balance += 1
            elif ch in closes:
                balance -= 1

        # Explicit backslash continuation?
        stripped = line.rstrip("\n")
        cont = stripped.endswith("\\")
        # If the logical statement seems complete, stop
        if balance <= 0 and not cont:
            break

        lineno += 1

    # Return without leading indentation
    return "".join(lines).strip()


def assigned_var_name(stmt_src: str):
    """
    If stmt_src is a simple assignment to a single name, return that
    name. Handles Assign, AnnAssign, AugAssign, and := (NamedExpr) at
    top level. Returns None otherwise.
    """
    try:
        mod = ast.parse(stmt_src, mode="exec")
    except SyntaxError:
        return None

    if not mod.body:
        return None

    stmt = mod.body[0]

    # x = ...
    if isinstance(stmt, ast.Assign):
        # only one target and it must be a Name
        if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            return stmt.targets[0].id
        return None

    # x: T = ...   or   x: T
    if isinstance(stmt, ast.AnnAssign):
        if isinstance(stmt.target, ast.Name):
            return stmt.target.id
        return None

    # x += ...
    if isinstance(stmt, ast.AugAssign):
        if isinstance(stmt.target, ast.Name):
            return stmt.target.id
        return None

    # Top-level walrus: (x := ...)
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.NamedExpr):
        target = stmt.value.target
        return target.id

    return None


def _node_src(src: str, node: ast.AST) -> str:
    """
    Best-effort recovery of original source text for a node.
    Prefers ast.get_source_segment; falls back to ast.unparse.
    """
    seg = ast.get_source_segment(src, node)
    if seg is not None:
        return seg
    return ast.unparse(node)


def _call_matches_func(call_node: ast.Call, f: Callable[..., Any]) -> bool:
    """
    Heuristic: match by function name (f.__name__) against either Name
    or Attribute.attr. This intentionally avoids importing modules; it's
    a pragmatic best-effort.
    """
    fname = getattr(f, "__name__", None)
    if fname is None:
        return False
    func = call_node.func
    if isinstance(func, ast.Name):
        return func.id == fname
    if isinstance(func, ast.Attribute):
        return func.attr == fname
    return False


def call_argument_sources(
    stmt_src: str, f: Callable[..., Any]
) -> tuple[list[str], dict[str, str]] | None:
    """
    Find the call to function `f` within the given statement source and
    return a tuple (args, kwargs) where:
        - args: list of positional argument source strings (including
          starred as '*expr')
        - kwargs: dict mapping keyword names to their source strings
          (including '**expr' as key None)
    If no matching call is found, returns None. If multiple matches
    exist, returns the first one in a pre-order walk.
    """
    try:
        mod = ast.parse(stmt_src, mode="exec")
    except SyntaxError:
        return None

    # Find first matching Call node
    target_call = None
    for node in ast.walk(mod):
        if isinstance(node, ast.Call) and _call_matches_func(node, f):
            target_call = node
            break

    if target_call is None:
        return None

    args: list[str] = []
    kwargs: dict[str, str] = {}

    # Positional (including starred)
    for arg in target_call.args:
        if isinstance(arg, ast.Starred):
            args.append("*" + _node_src(stmt_src, arg.value))
        else:
            args.append(_node_src(stmt_src, arg))

    # Keywords (including **kwargs where kw.arg is None)
    for kw in target_call.keywords:
        if kw.arg is None:
            # **expr
            kwargs["**"] = _node_src(stmt_src, kw.value)
        else:
            kwargs[kw.arg] = _node_src(stmt_src, kw.value)

    return args, kwargs

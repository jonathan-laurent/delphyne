"""
Mkdocs Extensions for Generating the Delphyne Documentation.
"""

# pyright: basic

import ast
import fnmatch
import importlib
import re
from typing import Any

import griffe

#####
##### Automatic cross references in docstrings
#####

logger = griffe.get_logger(__name__)

# Backticked simple identifiers: `Foo`
_ID_IN_BACKTICKS = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")

# If the text immediately before the backtick ends with one of:
#   "[" <spaces>    or
#   "(method|Method|argument|Argument)" <spaces>
# then we skip rewriting.
_SKIP_PRECEDENCE = re.compile(r"(?:\[\s*|(?:[Mm]ethod|[Aa]rgument)\s*)$")


class DelphyneAutoCrossrefs(griffe.Extension):
    """
    Rewrite docstrings to turn `ID` into [`ID`][delphyne.ID].
    """

    def __init__(self, root: str = "delphyne") -> None:
        super().__init__()
        self.root = root

    def on_instance(
        self,
        node: ast.AST | griffe.ObjectNode,
        obj: griffe.Object,
        agent: griffe.Visitor | griffe.Inspector,
        **kwargs,
    ) -> None:
        # Only process objects that actually have a docstring.
        if not obj.docstring:
            return

        text = obj.docstring.value
        if "`" not in text:
            return  # fast path

        # Resolve the root module once.
        root_mod = self._get_root_module(obj)
        if root_mod is None:
            logger.debug(
                "DelphyneCrossrefs: root module %r not found; skipping",
                self.root,
            )
            return

        def replace(m: re.Match[str]) -> str:
            start = m.start()
            # Examine a short slice before the match to check the
            # exclusion prefix.
            before = text[max(0, start - 64) : start]
            if _SKIP_PRECEDENCE.search(before):
                return m.group(0)

            ident = m.group(1)

            # Already in link form like [`X`][delphyne.X]? The preceding
            # "[" rule handles it, so we only need to check membership.
            if not self._exists_in_root(root_mod, ident):
                return m.group(0)

            return f"[`{ident}`][{self.root}.{ident}]"

        new_text = _ID_IN_BACKTICKS.sub(replace, text)
        if new_text != text:
            obj.docstring.value = new_text

    def _get_root_module(self, obj: griffe.Object) -> Any:
        return importlib.import_module(self.root)

    def _exists_in_root(self, root_mod: Any, name: str) -> bool:
        import inspect

        return hasattr(root_mod, name) and not inspect.ismodule(
            getattr(root_mod, name)
        )


#####
##### Workaround to support type aliases
#####


class TypeAliasesAsAttributes(griffe.Extension):
    """
    Turn `type X = ...` (PEP 695) into Attributes for rendering
    purposes. This is useful since mkdocs-material does not support type
    aliases yet.

    Options (all optional):

    - include: fnmatch-style patterns of fully-qualified names to include
    - exclude: fnmatch-style patterns of fully-qualified names to exclude
    """

    def __init__(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.include = include or []
        self.exclude = exclude or []

    def _wanted(self, path: str) -> bool:
        if self.include and not any(
            fnmatch.fnmatch(path, pat) for pat in self.include
        ):
            return False
        if any(fnmatch.fnmatch(path, pat) for pat in self.exclude):
            return False
        return True

    # Hook fired after a TypeAlias is created and added to its parent.
    def on_type_alias_instance(
        self,
        *,
        node: ast.AST | griffe.ObjectNode,
        type_alias: griffe.TypeAlias,
        agent: griffe.Visitor | griffe.Inspector,
        **kwargs: Any,
    ) -> None:
        parent = type_alias.parent
        if parent is None:
            return
        if not self._wanted(type_alias.path):
            return

        # Create an Attribute that carries the alias's name and points
        # its *annotation* to the alias target. Keep docstring, lines,
        # labels, etc., when available.
        attr = griffe.Attribute(type_alias.name, annotation=type_alias.value)
        # Preserve common metadata useful for mkdocstrings’ rendering.
        attr.docstring = type_alias.docstring
        attr.lineno = type_alias.lineno
        attr.endlineno = type_alias.endlineno
        attr.public = type_alias.public
        attr.parent = parent
        attr.labels.update(type_alias.labels)
        attr.extra.update(type_alias.extra)

        # Setting Kind.TYPE_ALIAS would cause the wrong template to be used.
        attr.kind = griffe.Kind.ATTRIBUTE

        # Replace in the parent’s members (the TypeAlias was just
        # inserted there).
        parent.members[type_alias.name] = attr

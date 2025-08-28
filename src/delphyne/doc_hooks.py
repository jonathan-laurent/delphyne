"""
Mkdocs Extensions for Generating the Delphyne Documentation.
"""

# pyright: basic

import ast
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

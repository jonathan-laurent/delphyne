"""
Resolving references within a trace.
"""

from dataclasses import dataclass
from typing import Any

import delphyne.core as dp
from delphyne.core import refs

#####
##### Navigation Trees
#####


@dataclass(frozen=True)
class NavTree:
    """
    A tree wrapper that caches all the nodes it visits.
    """

    tree: dp.Tree[Any, Any, Any]
    _cache: "dict[refs.GlobalNodeRef, NavTree]"

    def __post_init__(self):
        self._cache_current()

    def _cache_current(self):
        self._cache[self.tree.ref] = self

    @staticmethod
    def make(tree: dp.Tree[Any, Any, Any]) -> "NavTree":
        return NavTree(tree, {})

    @property
    def node(self) -> dp.Node | dp.Success[Any]:
        return self.tree.node

    @property
    def ref(self) -> refs.GlobalNodePath:
        return self.tree.ref

    def child(self, action: dp.Value) -> "NavTree":
        aref = refs.value_ref(action)
        cref = refs.child_ref(self.ref, aref)
        if cref in self._cache:
            return self._cache[cref]
        return NavTree(self.tree.child(action), self._cache)

    def nested(
        self, space: refs.SpaceName, args: tuple[dp.Value, ...]
    ) -> "NavTree":
        assert False


#####
##### Resolving References
#####

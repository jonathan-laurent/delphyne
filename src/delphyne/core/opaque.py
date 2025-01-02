"""
Opaque Spaces.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from delphyne.core.environment import PolicyEnv
from delphyne.core.streams import StreamRet
from delphyne.core.trees import AttachedQuery, Space, StrategyComp, Tag


@dataclass
class OpaqueSpace[P, T](Space[T]):
    stream: Callable[[PolicyEnv, P], StreamRet[T]]
    _source: "StrategyComp[Any, P, T] | AttachedQuery[T]"

    def source(self) -> "StrategyComp[Any, P, T] | AttachedQuery[T]":
        return self._source

    def tags(self) -> Sequence[Tag]:
        return self._source.tags()

    @staticmethod
    def from_query():
        assert False

    @staticmethod
    def from_strategy():
        assert False

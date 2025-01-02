"""
Opaque Spaces.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from delphyne.core.policies import PolicyEnv, StreamRet
from delphyne.core.trees import AttachedQuery, Space, StrategyComp, Tag


@dataclass
class OpaqueSpace[T](Space[T]):
    stream: Callable[[PolicyEnv, Any], StreamRet[T]]
    _source: "StrategyComp[Any, Any, T] | AttachedQuery[T]"

    def source(self) -> "StrategyComp[Any, Any, T] | AttachedQuery[T]":
        return self._source

    def tags(self) -> Sequence[Tag]:
        return self._source.tags()

    @staticmethod
    def from_query():
        assert False

    @staticmethod
    def from_strategy():
        assert False

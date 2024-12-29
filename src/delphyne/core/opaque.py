"""
Opaque Spaces.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from delphyne.core.policies import Policy, PolicyEnv, StreamRet
from delphyne.core.trees import AttachedQuery, Space, StrategyComp, Tag


@dataclass
class OpaqueSpace[T](Space[T]):
    stream: Callable[[PolicyEnv, Policy[Any]], StreamRet[T]]
    _source: "StrategyComp[Any, T] | AttachedQuery[T]"

    def source(self) -> "StrategyComp[Any, T] | AttachedQuery[T]":
        return self._source

    def tags(self) -> Sequence[Tag]:
        return self._source.tags()

    @staticmethod
    def from_query():
        assert False

    @staticmethod
    def from_strategy():
        assert False

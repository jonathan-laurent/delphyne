"""
Policies and opaque spaces.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from delphyne.core import trees as tr
from delphyne.core.environments import PolicyEnv
from delphyne.core.queries import AbstractQuery
from delphyne.core.streams import Stream
from delphyne.core.trees import AttachedQuery, NestedTree, Node, Space, Tree

#####
##### Search and Prompting Policies
#####


type Policy[N: Node, P] = tuple[AbstractSearchPolicy[N], P]
"""
A policy for trees with effects `N` is a pair of a search policy
handling `N` along with an inner policy object of type `P`.
"""


class AbstractSearchPolicy[N: tr.Node](Protocol):
    def __call__[P, T](
        self, tree: "Tree[N, P, T]", env: PolicyEnv, policy: P
    ) -> Stream[T]: ...


class AbstractPromptingPolicy(Protocol):
    def __call__[T](
        self, query: AttachedQuery[T], env: PolicyEnv
    ) -> Stream[T]: ...


#####
##### Opaque Spaces
#####


@dataclass(frozen=True)
class OpaqueSpace[P, T](Space[T]):
    stream: Callable[[PolicyEnv, P], Stream[T]]
    _source: NestedTree[Any, Any, T] | AttachedQuery[T]

    def source(self) -> NestedTree[Any, Any, T] | AttachedQuery[T]:
        return self._source

    def tags(self) -> Sequence[tr.Tag]:
        return self._source.tags()

    @staticmethod
    def from_query[P1, T1](
        query: AbstractQuery[T1],
        get_policy: Callable[[P1], AbstractPromptingPolicy],
    ) -> "tr.Builder[OpaqueSpace[P1, T1]]":
        def build(spawner: tr.QuerySpawner) -> OpaqueSpace[P1, T1]:
            attached = spawner(query)
            return OpaqueSpace(
                (lambda env, pol: get_policy(pol)(attached, env)), attached
            )

        return lambda _, spawner: build(spawner)

    @staticmethod
    def from_strategy[N: Node, P1, P2, T1](
        strategy: tr.StrategyComp[N, P2, T1],
        get_policy: Callable[[P1], tuple[AbstractSearchPolicy[N], P2]],
    ) -> "tr.Builder[OpaqueSpace[P1, T1]]":
        def build(spawner: tr.NestedTreeSpawner) -> OpaqueSpace[P1, T1]:
            nested = spawner(strategy)

            def stream(env: PolicyEnv, policy: P1) -> Stream[T1]:
                tree = nested.spawn_tree()
                search_pol, inner_pol = get_policy(policy)
                return search_pol(tree, env, inner_pol)

            return OpaqueSpace(stream, nested)

        return lambda spawner, _: build(spawner)

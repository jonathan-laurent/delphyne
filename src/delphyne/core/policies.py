"""
Policies and opaque spaces.
"""

from abc import ABC
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from delphyne.core import trees as tr
from delphyne.core.environments import PolicyEnv
from delphyne.core.queries import AbstractQuery
from delphyne.core.streams import Stream
from delphyne.core.trees import AttachedQuery, NestedTree, Node, Space, Tree

#####
##### Search and Prompting Policies
#####


N_po = TypeVar("N_po", bound=Node, contravariant=True)
P_po = TypeVar("P_po", covariant=True)


class AbstractPolicy(Generic[N_po, P_po], ABC):
    """
    A policy for trees with effects `N` gathers a search policy handling
    `N` along with an inner policy object of type `P`.
    """

    @property
    def search(self) -> "AbstractSearchPolicy[N_po]": ...
    @property
    def inner(self) -> P_po: ...


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


T = TypeVar("T", covariant=True)
P = TypeVar("P", contravariant=True)


@dataclass(frozen=True)
class OpaqueSpace(Generic[P, T], Space[T]):
    stream: Callable[[PolicyEnv, P], Stream[T]]
    _source: NestedTree[Any, Any, T] | AttachedQuery[T]
    _tags: Sequence[tr.Tag]

    def source(self) -> NestedTree[Any, Any, T] | AttachedQuery[T]:
        return self._source

    def tags(self) -> Sequence[tr.Tag]:
        return self._tags

    @staticmethod
    def from_query[P1, T1](
        query: AbstractQuery[T1],
        get_policy: Callable[[P1, Sequence[tr.Tag]], AbstractPromptingPolicy],
    ) -> "tr.SpaceBuilder[OpaqueSpace[P1, T1]]":
        def build(
            spawner: tr.QuerySpawner, tags: Sequence[tr.Tag]
        ) -> OpaqueSpace[P1, T1]:
            attached = spawner(query)
            return OpaqueSpace(
                stream=(lambda env, pol: get_policy(pol, tags)(attached, env)),
                _source=attached,
                _tags=tags,
            )

        return tr.SpaceBuilder(
            build=lambda _, spawner, tags: build(spawner, tags),
            tags=query.default_tags(),
        )

    @staticmethod
    def from_strategy[N: Node, P1, P2, T1](
        strategy: tr.StrategyComp[N, P2, T1],
        get_policy: Callable[[P1, Sequence[tr.Tag]], AbstractPolicy[N, P2]],
    ) -> "tr.SpaceBuilder[OpaqueSpace[P1, T1]]":
        def build(
            spawner: tr.NestedTreeSpawner, tags: Sequence[tr.Tag]
        ) -> OpaqueSpace[P1, T1]:
            nested = spawner(strategy)

            def stream(env: PolicyEnv, policy: P1) -> Stream[T1]:
                tree = nested.spawn_tree()
                sub = get_policy(policy, tags)
                return sub.search(tree, env, sub.inner)

            return OpaqueSpace(stream, nested, tags)

        return tr.SpaceBuilder(
            build=lambda spawner, _, tags: build(spawner, tags),
            tags=strategy.default_tags(),
        )


type Opaque[P, T] = tr.SpaceBuilder[OpaqueSpace[P, T]]

"""
Definition of Opaque Spaces
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, override

import delphyne.core as dp
from delphyne.core.trees import NestedTreeSpawner, QuerySpawner
from delphyne.stdlib.environments import PolicyEnv
from delphyne.stdlib.policies import Policy, PromptingPolicy, Stream

# Pyright somehow fails to correctly infer the variance of
# the type arguments of `OpaqueSpace`, so we manually specify them.
P = TypeVar("P", contravariant=True)
T = TypeVar("T", covariant=True)


@dataclass(frozen=True)
class OpaqueSpace(Generic[P, T], dp.Space[T]):
    """
    A space defined by a mapping from the ambient inner policy to a
    search stream.

    Opaque spaces can be defined from strategy instances
    (`StrategyInstance`) or from queries (`Query`) via the `using`
    method. Crucially, policies are unaware of how an opaque space was
    created, preserving abstraction.

    Type Parameters:
        P: Type parameter for the ambient inner policy type.
        T: Type parameter for the element type.

    Attributes:
        stream: Maps the ambient inner policy to a search stream.
    """

    stream: Callable[[PolicyEnv, P], Stream[T]]
    _source: dp.NestedTree[Any, Any, T] | dp.AttachedQuery[T]
    _tags: Sequence[dp.Tag]

    @override
    def source(self) -> dp.NestedTree[Any, Any, T] | dp.AttachedQuery[T]:
        return self._source

    @override
    def tags(self) -> Sequence[dp.Tag]:
        return self._tags

    @staticmethod
    def from_query[P1, T1](
        query: dp.AbstractQuery[T1],
        get_policy: Callable[[P1, Sequence[dp.Tag]], PromptingPolicy],
    ) -> "dp.SpaceBuilder[OpaqueSpace[P1, T1]]":
        """
        Create an opaque space from a query.

        The `Query.using` method is a more ergonomic wrapper.
        """

        def build(
            spawner: QuerySpawner, tags: Sequence[dp.Tag]
        ) -> OpaqueSpace[P1, T1]:
            attached = spawner(query)
            return OpaqueSpace(
                stream=(lambda env, pol: get_policy(pol, tags)(attached, env)),
                _source=attached,
                _tags=tags,
            )

        return dp.SpaceBuilder(
            build=lambda _, spawner, tags: build(spawner, tags),
            tags=query.default_tags(),
        )

    @staticmethod
    def from_strategy[N: dp.Node, P1, P2, T1](
        strategy: dp.StrategyComp[N, P2, T1],
        get_policy: Callable[[P1, Sequence[dp.Tag]], Policy[N, P2]],
    ) -> "dp.SpaceBuilder[OpaqueSpace[P1, T1]]":
        """
        Create an opaque space from a strategy instance.

        The `StrategyInstance.using` method is a more ergonomic wrapper.
        """

        def build(
            spawner: NestedTreeSpawner, tags: Sequence[dp.Tag]
        ) -> OpaqueSpace[P1, T1]:
            nested = spawner(strategy)

            def stream(env: PolicyEnv, policy: P1) -> Stream[T1]:
                tree = nested.spawn_tree()
                sub = get_policy(policy, tags)
                return sub.search(tree, env, sub.inner)

            return OpaqueSpace(stream, nested, tags)

        return dp.SpaceBuilder(
            build=lambda spawner, _, tags: build(spawner, tags),
            tags=strategy.default_tags(),
        )


type Opaque[P, T] = dp.SpaceBuilder[OpaqueSpace[P, T]]
"""
A convenience type alias for an opaque space builder.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import delphyne.core as dp
from delphyne.core.trees import NestedTreeSpawner, QuerySpawner
from delphyne.stdlib.policies import Policy, PromptingPolicy, SearchStream

T = TypeVar("T", covariant=True)
P = TypeVar("P", contravariant=True)


@dataclass(frozen=True)
class OpaqueSpace(Generic[P, T], dp.Space[T]):
    stream: Callable[[dp.PolicyEnv, P], SearchStream[T]]
    _source: dp.NestedTree[Any, Any, T] | dp.AttachedQuery[T]
    _tags: Sequence[dp.Tag]

    def source(self) -> dp.NestedTree[Any, Any, T] | dp.AttachedQuery[T]:
        return self._source

    def tags(self) -> Sequence[dp.Tag]:
        return self._tags

    @staticmethod
    def from_query[P1, T1](
        query: dp.AbstractQuery[T1],
        get_policy: Callable[[P1, Sequence[dp.Tag]], PromptingPolicy],
    ) -> "dp.SpaceBuilder[OpaqueSpace[P1, T1]]":
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
        def build(
            spawner: NestedTreeSpawner, tags: Sequence[dp.Tag]
        ) -> OpaqueSpace[P1, T1]:
            nested = spawner(strategy)

            def stream(env: dp.PolicyEnv, policy: P1) -> SearchStream[T1]:
                tree = nested.spawn_tree()
                sub = get_policy(policy, tags)
                return sub.search(tree, env, sub.inner)

            return OpaqueSpace(stream, nested, tags)

        return dp.SpaceBuilder(
            build=lambda spawner, _, tags: build(spawner, tags),
            tags=strategy.default_tags(),
        )


type Opaque[P, T] = dp.SpaceBuilder[OpaqueSpace[P, T]]

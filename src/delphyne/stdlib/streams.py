"""
Utilities to work with streams.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Never, Protocol, overload

import delphyne.core as dp
from delphyne.stdlib.policies import PromptingPolicy, SearchPolicy

#####
##### Stream transformers
#####


class _StreamTransformerFn(Protocol):
    def __call__[T](self, stream: dp.Stream[T]) -> dp.Stream[T]: ...


@dataclass
class StreamTransformer:
    trans: _StreamTransformerFn

    def __call__[T](self, stream: dp.Stream[T]) -> dp.Stream[T]:
        return self.trans(stream)

    @overload
    def __matmul__[N: dp.Node](
        self, other: SearchPolicy[N]
    ) -> SearchPolicy[N]: ...

    @overload
    def __matmul__(self, other: PromptingPolicy) -> PromptingPolicy: ...

    def __matmul__[N: dp.Node](
        self, other: SearchPolicy[N] | PromptingPolicy
    ) -> SearchPolicy[N] | PromptingPolicy:
        if isinstance(other, SearchPolicy):
            return self.compose_with_search_policy(other)
        else:
            return self.compose_with_prompting_policy(other)

    def compose_with_search_policy[N: dp.Node](
        self, other: SearchPolicy[N]
    ) -> SearchPolicy[N]:
        def policy[P, T](
            tree: dp.Tree[N, P, T], env: dp.PolicyEnv, policy: P
        ) -> dp.Stream[T]:
            stream = other(tree, env, policy)
            return self.trans(stream)

        return SearchPolicy(policy)

    def compose_with_prompting_policy(
        self, other: PromptingPolicy
    ) -> PromptingPolicy:
        def policy[P, T](
            query: dp.AttachedQuery[T], env: dp.PolicyEnv
        ) -> dp.Stream[T]:
            stream = other(query, env)
            return self.trans(stream)

        return PromptingPolicy(policy)


class _ParametricStreamTransformerFn[**A](Protocol):
    def __call__[T](
        self, stream: dp.Stream[T], *args: A.args, **kwargs: A.kwargs
    ) -> dp.Stream[T]: ...


def stream_transformer[**A](
    f: _ParametricStreamTransformerFn[A],
) -> Callable[A, StreamTransformer]:
    def parametric(*args: A.args, **kwargs: A.kwargs) -> StreamTransformer:
        def transformer[T](stream: dp.Stream[T]) -> dp.Stream[T]:
            return f(stream, *args, **kwargs)

        return StreamTransformer(transformer)

    return parametric


#####
##### Standard Stream Transformers
#####


@stream_transformer
async def with_budget[T](
    stream: dp.Stream[T], budget: dp.BudgetLimit
) -> dp.Stream[T]:
    total_spent = dp.Budget.zero()
    async for msg in stream:
        match msg:
            case dp.Spent(spent):
                total_spent = total_spent + spent
            case dp.Barrier(pred):
                if not (total_spent + pred <= budget):
                    return
            case _:
                pass
        yield msg


@stream_transformer
async def take[T](stream: dp.Stream[T], num_generated: int) -> dp.Stream[T]:
    count = 0
    assert num_generated > 0
    async for msg in stream:
        if isinstance(msg, dp.Yield):
            count += 1
        yield msg
        if count >= num_generated:
            return


#####
##### Stream Utilities
#####


async def bind_stream[A, B](
    stream: dp.Stream[A], f: Callable[[dp.Tracked[A]], dp.Stream[B]]
) -> dp.Stream[B]:
    async for msg in stream:
        if not isinstance(msg, dp.Yield):
            yield msg
            continue
        async for new_msg in f(msg.value):
            yield new_msg


@dataclass
class ElementStore[T](Exception):
    value: dp.Tracked[T] | None = None


async def take_one[T](
    stream: dp.Stream[T], store: ElementStore[T]
) -> dp.Stream[Never]:
    async for msg in stream:
        if isinstance(msg, dp.Yield):
            store.value = msg.value
            return
        yield msg


async def collect[T](
    stream: dp.Stream[T],
    budget: dp.BudgetLimit | None = None,
    num_generated: int | None = None,
) -> tuple[Sequence[dp.Tracked[T]], dp.Budget]:
    if budget is not None:
        stream = with_budget(budget)(stream)
    if num_generated is not None:
        stream = take(num_generated)(stream)
    total = dp.Budget.zero()
    elts: list[dp.Tracked[T]] = []
    async for msg in stream:
        if isinstance(msg, dp.Yield):
            elts.append(msg.value)
        if isinstance(msg, dp.Spent):
            total = total + msg.budget
    return elts, total

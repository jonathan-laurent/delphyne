"""
Utilities to work with streams.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import delphyne.core as dp
from delphyne.stdlib.models import NUM_REQUESTS_BUDGET

#####
##### Stream transformers
#####


class _StreamTransformerFn(Protocol):
    def __call__[T](self, stream: dp.Stream[T]) -> dp.Stream[T]: ...


@dataclass
class StreamTransformer:
    trans: _StreamTransformerFn

    def __matmul__[N: dp.Node](
        self, other: dp.SearchPolicy[N]
    ) -> dp.SearchPolicy[N]:
        def policy[P, T](
            tree: dp.Tree[N, P, T], env: dp.PolicyEnv, policy: P
        ) -> dp.Stream[T]:
            stream = other(tree, env, policy)
            return self.trans(stream)

        return dp.SearchPolicy(policy)


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


async def _with_budget[T](
    budget: dp.BudgetLimit, stream: dp.Stream[T]
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
def with_budget[T](stream: dp.Stream[T], num_requests: int) -> dp.Stream[T]:
    return _with_budget(
        dp.BudgetLimit({NUM_REQUESTS_BUDGET: num_requests}), stream
    )


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

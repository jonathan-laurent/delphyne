"""
Utilities to work with streams.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from delphyne.core import trees as tr
from delphyne.core.environment import PolicyEnv
from delphyne.core.streams import Barrier, Budget, BudgetLimit, Spent, Stream
from delphyne.core.trees import Node, SearchPolicy
from delphyne.stdlib.models import NUM_REQUESTS_BUDGET

#####
##### Stream transformers
#####


class _StreamTransformerFn(Protocol):
    def __call__[T](self, stream: Stream[T]) -> Stream[T]: ...


@dataclass
class StreamTransformer:
    trans: _StreamTransformerFn

    def __matmul__[N: Node](self, other: SearchPolicy[N]) -> SearchPolicy[N]:
        def policy[P, T](
            tree: tr.Tree[N, P, T], env: PolicyEnv, policy: P
        ) -> Stream[T]:
            stream = other(tree, env, policy)
            return self.trans(stream)

        return SearchPolicy(policy)


class _ParametricStreamTransformerFn[**A](Protocol):
    def __call__[T](
        self, stream: Stream[T], *args: A.args, **kwargs: A.kwargs
    ) -> Stream[T]: ...


def stream_transformer[**A](
    f: _ParametricStreamTransformerFn[A],
) -> Callable[A, StreamTransformer]:
    def parametric(*args: A.args, **kwargs: A.kwargs) -> StreamTransformer:
        def transformer[T](stream: Stream[T]) -> Stream[T]:
            return f(stream, *args, **kwargs)

        return StreamTransformer(transformer)

    return parametric


#####
##### Utilities
#####


async def _with_budget[T](budget: BudgetLimit, stream: Stream[T]) -> Stream[T]:
    total_spent = Budget.zero()
    async for msg in stream:
        match msg:
            case Spent(spent):
                total_spent = total_spent + spent
            case Barrier(pred):
                if not (total_spent + pred <= budget):
                    return
            case _:
                pass
        yield msg


@stream_transformer
def with_budget[T](stream: Stream[T], num_requests: int) -> Stream[T]:
    return _with_budget(
        BudgetLimit({NUM_REQUESTS_BUDGET: num_requests}), stream
    )

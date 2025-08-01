"""
Utilities to work with streams.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import delphyne.core as dp

#####
##### Search Streams
#####


@dataclass(frozen=True)
class SearchStream[T](dp.AbstractSearchStream[T]):
    _generate: Callable[[], dp.Stream[T]]

    def gen(self) -> dp.Stream[T]:
        return self._generate()

    ## Transforming the stream

    def with_budget(self, budget: dp.BudgetLimit):
        return SearchStream(lambda: stream_with_budget(self.gen(), budget))

    def take(self, num_generated: int):
        return SearchStream(lambda: stream_take(self.gen(), num_generated))

    ## Collecting values

    def first(self) -> dp.StreamGen[dp.SearchValue[T] | None]:
        return stream_take_one(self.gen())

    def all(self) -> dp.StreamGen[Sequence[dp.SearchValue[T]]]:
        return stream_take_all(self.gen())

    def collect(
        self,
        budget: dp.BudgetLimit | None = None,
        num_generated: int | None = None,
    ):
        if budget is not None:
            self = self.with_budget(budget)
        if num_generated is not None:
            self = self.take(num_generated)
        return stream_collect(self.gen())

    ## Combinators

    def bind[T2](
        self, f: Callable[[dp.SearchValue[T]], dp.Stream[T2]]
    ) -> "SearchStream[T2]":
        return SearchStream(lambda: stream_bind(self.gen(), f))


#####
##### Stream transformers
#####


class _StreamTransformerFn(Protocol):
    def __call__[T](
        self,
        stream: SearchStream[T],
        env: dp.PolicyEnv,
    ) -> dp.Stream[T]: ...


class _ParametricStreamTransformerFn[**A](Protocol):
    def __call__[T](
        self,
        stream: SearchStream[T],
        env: dp.PolicyEnv,
        *args: A.args,
        **kwargs: A.kwargs,
    ) -> dp.Stream[T]: ...


@dataclass
class StreamTransformer:
    trans: _StreamTransformerFn

    def __call__[T](
        self,
        stream: SearchStream[T],
        env: dp.PolicyEnv,
    ) -> SearchStream[T]:
        return SearchStream(lambda: self.trans(stream, env))

    def __matmul__(self, other: "StreamTransformer") -> "StreamTransformer":
        if not isinstance(other, StreamTransformer):  # type: ignore[reportUnnecessaryInstance]
            return NotImplemented

        def transformer[T](
            stream: SearchStream[T],
            env: dp.PolicyEnv,
        ) -> dp.Stream[T]:
            return self(other(stream, env), env).gen()

        return StreamTransformer(transformer)


def stream_transformer[**A](
    f: _ParametricStreamTransformerFn[A],
) -> Callable[A, StreamTransformer]:
    def parametric(*args: A.args, **kwargs: A.kwargs) -> StreamTransformer:
        def transformer[T](
            stream: SearchStream[T],
            env: dp.PolicyEnv,
        ) -> dp.Stream[T]:
            return f(stream, env, *args, **kwargs)

        return StreamTransformer(transformer)

    return parametric


#####
##### Streams Combinators
#####


class _StreamCombinatorFn(Protocol):
    def __call__[T](
        self,
        streams: Sequence[SearchStream[T]],
        probs: Sequence[float],
        env: dp.PolicyEnv,
    ) -> dp.Stream[T]: ...


@dataclass
class StreamCombinator:
    combine: _StreamCombinatorFn

    def __call__[T](
        self,
        streams: Sequence[SearchStream[T]],
        probs: Sequence[float],
        env: dp.PolicyEnv,
    ) -> SearchStream[T]:
        return SearchStream(lambda: self.combine(streams, probs, env))

    def __rmatmul__(self, other: StreamTransformer) -> "StreamCombinator":
        if not isinstance(other, StreamTransformer):  # type: ignore[reportUnnecessaryInstance]
            return NotImplemented

        def combinator[T](
            streams: Sequence[SearchStream[T]],
            probs: Sequence[float],
            env: dp.PolicyEnv,
        ) -> dp.Stream[T]:
            return other(self(streams, probs, env), env).gen()

        return StreamCombinator(combinator)


#####
##### Stream Utilities
#####


def stream_bind[A, B](
    stream: dp.Stream[A], f: Callable[[dp.SearchValue[A]], dp.Stream[B]]
) -> dp.Stream[B]:
    for msg in stream:
        if not isinstance(msg, dp.Yield):
            yield msg
            continue
        for new_msg in f(msg.value):
            yield new_msg


def stream_take_one[T](
    stream: dp.Stream[T],
) -> dp.StreamGen[dp.SearchValue[T] | None]:
    for msg in stream:
        if isinstance(msg, dp.Yield):
            return msg.value
        yield msg
    return None


def stream_take_all[T](
    stream: dp.Stream[T],
) -> dp.StreamGen[Sequence[dp.SearchValue[T]]]:
    res: list[dp.SearchValue[T]] = []
    for msg in stream:
        if isinstance(msg, dp.Yield):
            res.append(msg.value)
            continue
        yield msg
    return res


def stream_with_budget[T](
    stream: dp.Stream[T], budget: dp.BudgetLimit
) -> dp.Stream[T]:
    """
    See `with_budget` for a version wrapped as a stream transformer.
    """
    total_spent = dp.Budget.zero()
    for msg in stream:
        match msg:
            case dp.Spent(spent):
                total_spent = total_spent + spent
            case dp.Barrier(pred):
                if not (total_spent + pred <= budget):
                    return
            case _:
                pass
        yield msg


def stream_take[T](stream: dp.Stream[T], num_generated: int) -> dp.Stream[T]:
    """
    See `take` for a version wrapped as a stream transformer.
    """
    count = 0
    assert num_generated > 0
    for msg in stream:
        if isinstance(msg, dp.Yield):
            count += 1
        yield msg
        if count >= num_generated:
            return


def stream_collect[T](
    stream: dp.Stream[T],
) -> tuple[Sequence[dp.SearchValue[T]], dp.Budget]:
    total = dp.Budget.zero()
    elts: list[dp.SearchValue[T]] = []
    for msg in stream:
        if isinstance(msg, dp.Yield):
            elts.append(msg.value)
        if isinstance(msg, dp.Spent):
            total = total + msg.budget
    return elts, total


type _StreamBuilder[T] = Callable[[], dp.Stream[T]]


def stream_sequence[T](
    stream_builders: Sequence[_StreamBuilder[T]],
) -> dp.Stream[T]:
    for s in stream_builders:
        yield from s()


def stream_or_else[T](
    main: _StreamBuilder[T], fallback: _StreamBuilder[T]
) -> dp.Stream[T]:
    some_successes = False
    for msg in main():
        if isinstance(msg, dp.Yield):
            some_successes = True
        yield msg
    if not some_successes:
        for msg in fallback():
            yield msg


#####
##### Standard Stream Transformers
#####


@stream_transformer
def with_budget[T](
    stream: SearchStream[T],
    env: dp.PolicyEnv,
    budget: dp.BudgetLimit,
):
    return stream_with_budget(stream.gen(), budget)


@stream_transformer
def take[T](
    stream: SearchStream[T],
    env: dp.PolicyEnv,
    num_generated: int,
):
    return stream_take(stream.gen(), num_generated)


@stream_transformer
def loop[T](
    stream: SearchStream[T],
    env: dp.PolicyEnv,
    n: int | None = None,
) -> dp.Stream[T]:
    """
    Stream transformer that repeatedly respawns the underlying stream,
    up to an (optional) limit.
    """
    i = 0
    while (n is None) or (i < n):
        i += 1
        yield from stream.gen()

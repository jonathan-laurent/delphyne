"""
Utilities to work with streams.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import delphyne.core as dp
from delphyne.core.streams import Barrier, Spent

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

    def take(self, num_generated: int, strict: bool = True):
        return SearchStream(
            lambda: stream_take(self.gen(), num_generated, strict)
        )

    ## Collecting values

    def first(self) -> dp.StreamGen[dp.Solution[T] | None]:
        return stream_take_one(self.gen())

    def all(self) -> dp.StreamGen[Sequence[dp.Solution[T]]]:
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
        self, f: Callable[[dp.Solution[T]], dp.Stream[T2]]
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
        if not isinstance(other, StreamTransformer):  # pyright: ignore[reportUnnecessaryIsInstance]
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
        if not isinstance(other, StreamTransformer):  # pyright: ignore[reportUnnecessaryIsInstance]
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


# def _wait_balance(pending_barriers: int):
#     pass


def stream_bind[A, B](
    stream: dp.Stream[A], f: Callable[[dp.Solution[A]], dp.Stream[B]]
) -> dp.Stream[B]:
    generated: list[dp.Solution[A]] = []
    num_pending = 0
    for msg in stream:
        match msg:
            case dp.Solution():
                generated.append(msg)
            case Barrier():
                num_pending += 1
                if generated:
                    # We don't allow new spending before `generated` is
                    # emptied.
                    msg.allow = False
                yield msg
            case Spent():
                num_pending -= 1
                yield msg
        while generated:
            yield from f(generated.pop(0))
    assert not generated
    assert num_pending == 0


def stream_take_one[T](
    stream: dp.Stream[T],
) -> dp.StreamGen[dp.Solution[T] | None]:
    for msg in stream:
        if isinstance(msg, dp.Solution):
            return msg
        yield msg
    return None


def stream_take_all[T](
    stream: dp.Stream[T],
) -> dp.StreamGen[Sequence[dp.Solution[T]]]:
    res: list[dp.Solution[T]] = []
    for msg in stream:
        if isinstance(msg, dp.Solution):
            res.append(msg)
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
    # Map the id of a barrier to the frozen budget.
    pending: dict[int, dp.Budget] = {}

    for msg in stream:
        match msg:
            case Barrier(pred):
                bound = (
                    total_spent
                    + pred
                    + sum(pending.values(), start=dp.Budget.zero())
                )
                if not (bound <= budget):
                    msg.allow = False
                pending[id(msg)] = (
                    msg.budget if msg.allow else dp.Budget.zero()
                )
            case Spent(spent):
                total_spent = total_spent + spent
                matching = id(msg.barrier)
                assert matching in pending
                del pending[matching]
            case _:
                pass
        yield msg
    assert not pending


def stream_take[T](
    stream: dp.Stream[T], num_generated: int, strict: bool = True
) -> dp.Stream[T]:
    """
    See `take` for a version wrapped as a stream transformer.
    """
    count = 0
    assert num_generated > 0
    for msg in stream:
        if isinstance(msg, dp.Solution):
            count += 1
        if isinstance(msg, Barrier) and count >= num_generated:
            msg.allow = False
        if not (
            strict and isinstance(msg, dp.Solution) and count > num_generated
        ):
            yield msg


def stream_collect[T](
    stream: dp.Stream[T],
) -> tuple[Sequence[dp.Solution[T]], dp.Budget]:
    total = dp.Budget.zero()
    elts: list[dp.Solution[T]] = []
    for msg in stream:
        if isinstance(msg, dp.Solution):
            elts.append(msg)
        if isinstance(msg, Spent):
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
        if isinstance(msg, dp.Solution):
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


#####
##### Utils
#####


@dataclass(frozen=True)
class SpendingDeclined:
    pass


def spend_on[T](
    f: Callable[[], tuple[T, dp.Budget]], /, estimate: dp.Budget
) -> dp.StreamGen[T | SpendingDeclined]:
    barrier = Barrier(estimate, allow=True)
    yield barrier
    if barrier.allow:
        value, spent = f()
        yield Spent(budget=spent, barrier=barrier)
        return value
    else:
        yield Spent(budget=dp.Budget.zero(), barrier=barrier)
        return SpendingDeclined()

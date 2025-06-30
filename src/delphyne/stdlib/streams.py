"""
Utilities to work with streams.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, overload

import delphyne.core as dp
from delphyne.stdlib.policies import PromptingPolicy, SearchPolicy

#####
##### Stream transformers
#####


class _StreamTransformerFn(Protocol):
    """
    A stream transformer takes as an argument a stream builder instead
    of a stream so that it can instantiate the stream several times,
    allowing to implement `loop`. Also, having access to the policy
    environment can be useful (e.g. to access particular configuration
    or template files).
    """

    def __call__[T](
        self,
        stream_builder: Callable[[], dp.Stream[T]],
        env: dp.PolicyEnv,
    ) -> dp.Stream[T]: ...


class _ParametricStreamTransformerFn[**A](Protocol):
    def __call__[T](
        self,
        stream_builder: Callable[[], dp.Stream[T]],
        env: dp.PolicyEnv,
        *args: A.args,
        **kwargs: A.kwargs,
    ) -> dp.Stream[T]: ...


class _PureStreamTransformerFn(Protocol):
    """
    Special case of _pure_ stream transformer functions that do not need
    to replicate the source stream or access the policy environment.
    """

    def __call__[T](self, stream: dp.Stream[T]) -> dp.Stream[T]: ...


class _PureParametricStreamTransformerFn[**A](Protocol):
    def __call__[T](
        self,
        stream: dp.Stream[T],
        *args: A.args,
        **kwargs: A.kwargs,
    ) -> dp.Stream[T]: ...


@dataclass
class StreamTransformer:
    """
    Wraps a stream transformer function (`_StreamTransformerFn`) so that
    it can compose nicely with search policies using the `@` operator.
    """

    trans: _StreamTransformerFn

    def __call__[T](
        self,
        stream_builder: Callable[[], dp.Stream[T]],
        env: dp.PolicyEnv,
    ) -> dp.Stream[T]:
        return self.trans(stream_builder, env)

    @staticmethod
    def pure(
        fn: _PureStreamTransformerFn,
    ) -> "StreamTransformer":
        def pure_transformer[T](
            stream_builder: Callable[[], dp.Stream[T]],
            env: dp.PolicyEnv,
        ) -> dp.Stream[T]:
            return fn(stream_builder())

        return StreamTransformer(pure_transformer)

    @staticmethod
    def pure_parametric[**A](
        fn: _PureParametricStreamTransformerFn[A],
    ) -> Callable[A, "StreamTransformer"]:
        def parametric(*args: A.args, **kwargs: A.kwargs) -> StreamTransformer:
            def pure_transformer[T](
                stream_builder: Callable[[], dp.Stream[T]],
                env: dp.PolicyEnv,
            ) -> dp.Stream[T]:
                return fn(stream_builder(), *args, **kwargs)

            return StreamTransformer(pure_transformer)

        return parametric

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
            return self.trans(lambda: other(tree, env, policy), env)

        return SearchPolicy(policy)

    def compose_with_prompting_policy(
        self, other: PromptingPolicy
    ) -> PromptingPolicy:
        def policy[P, T](
            query: dp.AttachedQuery[T], env: dp.PolicyEnv
        ) -> dp.Stream[T]:
            return self.trans(lambda: other(query, env), env)

        return PromptingPolicy(policy)


def stream_transformer[**A](
    f: _ParametricStreamTransformerFn[A],
) -> Callable[A, StreamTransformer]:
    def parametric(*args: A.args, **kwargs: A.kwargs) -> StreamTransformer:
        def transformer[T](
            stream_builder: Callable[[], dp.Stream[T]],
            env: dp.PolicyEnv,
        ) -> dp.Stream[T]:
            return f(stream_builder, env, *args, **kwargs)

        return StreamTransformer(transformer)

    return parametric


#####
##### Stream Utilities
#####


def bind_stream[A, B](
    stream: dp.Stream[A], f: Callable[[dp.Tracked[A]], dp.Stream[B]]
) -> dp.Stream[B]:
    for msg in stream:
        if not isinstance(msg, dp.Yield):
            yield msg
            continue
        for new_msg in f(msg.value):
            yield new_msg


def take_one_with_meta[T](
    stream: dp.Stream[T],
) -> dp.StreamGen[tuple[dp.Tracked[T], dp.SearchMetadata | None] | None]:
    for msg in stream:
        if isinstance(msg, dp.Yield):
            return (msg.value, msg.meta)
        yield msg
    return None


def take_one[T](stream: dp.Stream[T]) -> dp.StreamGen[dp.Tracked[T] | None]:
    for msg in stream:
        if isinstance(msg, dp.Yield):
            return msg.value
        yield msg
    return None


def take_all[T](stream: dp.Stream[T]) -> dp.StreamGen[Sequence[dp.Tracked[T]]]:
    res: list[dp.Tracked[T]] = []
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


def collect_with_metadata[T](
    stream: dp.Stream[T],
    budget: dp.BudgetLimit | None = None,
    num_generated: int | None = None,
) -> tuple[Sequence[tuple[dp.Tracked[T], dp.SearchMetadata]], dp.Budget]:
    if budget is not None:
        stream = stream_with_budget(stream, budget)
    if num_generated is not None:
        stream = stream_take(stream, num_generated)
    total = dp.Budget.zero()
    elts: list[tuple[dp.Tracked[T], dp.SearchMetadata]] = []
    for msg in stream:
        if isinstance(msg, dp.Yield):
            elts.append((msg.value, msg.meta))
        if isinstance(msg, dp.Spent):
            total = total + msg.budget
    return elts, total


def collect[T](
    stream: dp.Stream[T],
    budget: dp.BudgetLimit | None = None,
    num_generated: int | None = None,
) -> tuple[Sequence[dp.Tracked[T]], dp.Budget]:
    res, spent = collect_with_metadata(stream, budget, num_generated)
    return [elt[0] for elt in res], spent


def stream_squash[T](stream: dp.Stream[Sequence[T]]) -> dp.Stream[Sequence[T]]:
    # TODO: implementing this requires being able to convert a list of
    # tracked object into a tracked object.
    assert False


def stream_sequence[T](
    stream_builders: Sequence[Callable[[], dp.Stream[T]]],
) -> dp.Stream[T]:
    for s in stream_builders:
        yield from s()


#####
##### Standard Stream Transformers
#####


with_budget = StreamTransformer.pure_parametric(stream_with_budget)

take = StreamTransformer.pure_parametric(stream_take)


@stream_transformer
def loop[T](
    stream_builder: Callable[[], dp.Stream[T]],
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
        yield from stream_builder()

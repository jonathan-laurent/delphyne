"""
Utilities to work with streams.
"""

import itertools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import delphyne.core as dp
from delphyne.core.streams import Barrier, BarrierId, Spent

#####
##### Search Streams
#####


@dataclass(frozen=True)
class Stream[T](dp.AbstractStream[T]):
    _generate: Callable[[], dp.StreamGen[T]]

    def gen(self) -> dp.StreamGen[T]:
        return self._generate()

    ## Collecting all elements

    def collect(
        self,
        budget: dp.BudgetLimit | None = None,
        num_generated: int | None = None,
    ) -> tuple[Sequence[dp.Solution[T]], dp.Budget]:
        if budget is not None:
            self = self.with_budget(budget)
        if num_generated is not None:
            self = self.take(num_generated)
        return stream_collect(self.gen())

    ## Transforming the stream

    def with_budget(self, budget: dp.BudgetLimit):
        return Stream(lambda: stream_with_budget(self.gen(), budget))

    def take(self, num_generated: int, strict: bool = True):
        return Stream(lambda: stream_take(self.gen(), num_generated, strict))

    def loop(
        self, n: int | None = None, *, stop_on_reject: bool = True
    ) -> "Stream[T]":
        it = itertools.count() if n is None else range(n)
        return Stream(
            lambda: stream_sequence(
                (self.gen for _ in it), stop_on_reject=stop_on_reject
            )
        )

    def bind[T2](
        self, f: Callable[[dp.Solution[T]], dp.StreamGen[T2]]
    ) -> "Stream[T2]":
        return Stream(lambda: stream_bind(self.gen(), f))

    ## Monadic Methods

    def first(self) -> dp.StreamContext[dp.Solution[T] | None]:
        return stream_take_one(self.gen())

    def all(self) -> dp.StreamContext[Sequence[dp.Solution[T]]]:
        return stream_take_all(self.gen())

    def next(
        self,
    ) -> dp.StreamContext[
        "tuple[Sequence[dp.Solution[T]], dp.Budget, Stream[T] | None]"
    ]:
        gen, budg, rest = yield from stream_next(self.gen())
        new_rest = None if rest is None else Stream(lambda: rest)
        return gen, budg, new_rest

    ## Static Methods

    @staticmethod
    def sequence[U](
        streams: Iterable["Stream[U]"], *, stop_on_reject: bool = True
    ) -> "Stream[U]":
        return Stream(
            lambda: stream_sequence(
                (s.gen for s in streams), stop_on_reject=stop_on_reject
            )
        )

    @staticmethod
    def parallel[U](streams: Sequence["Stream[U]"]) -> "Stream[U]":
        return Stream(lambda: stream_parallel([s.gen() for s in streams]))


#####
##### Stream transformers
#####


class _StreamTransformerFn(Protocol):
    def __call__[T](
        self,
        stream: Stream[T],
        env: dp.PolicyEnv,
    ) -> dp.StreamGen[T]: ...


class _ParametricStreamTransformerFn[**A](Protocol):
    def __call__[T](
        self,
        stream: Stream[T],
        env: dp.PolicyEnv,
        *args: A.args,
        **kwargs: A.kwargs,
    ) -> dp.StreamGen[T]: ...


@dataclass
class StreamTransformer:
    trans: _StreamTransformerFn

    def __call__[T](
        self,
        stream: Stream[T],
        env: dp.PolicyEnv,
    ) -> Stream[T]:
        return Stream(lambda: self.trans(stream, env))

    def __matmul__(self, other: "StreamTransformer") -> "StreamTransformer":
        if not isinstance(other, StreamTransformer):  # pyright: ignore[reportUnnecessaryIsInstance]
            return NotImplemented

        def transformer[T](
            stream: Stream[T],
            env: dp.PolicyEnv,
        ) -> dp.StreamGen[T]:
            return self(other(stream, env), env).gen()

        return StreamTransformer(transformer)


def stream_transformer[**A](
    f: _ParametricStreamTransformerFn[A],
) -> Callable[A, StreamTransformer]:
    def parametric(*args: A.args, **kwargs: A.kwargs) -> StreamTransformer:
        def transformer[T](
            stream: Stream[T],
            env: dp.PolicyEnv,
        ) -> dp.StreamGen[T]:
            return f(stream, env, *args, **kwargs)

        return StreamTransformer(transformer)

    return parametric


#####
##### Streams Combinators
#####


class _StreamCombinatorFn(Protocol):
    def __call__[T](
        self,
        streams: Sequence[Stream[T]],
        probs: Sequence[float],
        env: dp.PolicyEnv,
    ) -> dp.StreamGen[T]: ...


@dataclass
class StreamCombinator:
    combine: _StreamCombinatorFn

    def __call__[T](
        self,
        streams: Sequence[Stream[T]],
        probs: Sequence[float],
        env: dp.PolicyEnv,
    ) -> Stream[T]:
        return Stream(lambda: self.combine(streams, probs, env))

    def __rmatmul__(self, other: StreamTransformer) -> "StreamCombinator":
        if not isinstance(other, StreamTransformer):  # pyright: ignore[reportUnnecessaryIsInstance]
            return NotImplemented

        def combinator[T](
            streams: Sequence[Stream[T]],
            probs: Sequence[float],
            env: dp.PolicyEnv,
        ) -> dp.StreamGen[T]:
            return other(self(streams, probs, env), env).gen()

        return StreamCombinator(combinator)


#####
##### Standard Stream Transformers
#####


@stream_transformer
def with_budget[T](
    stream: Stream[T],
    env: dp.PolicyEnv,
    budget: dp.BudgetLimit,
):
    return stream_with_budget(stream.gen(), budget)


@stream_transformer
def take[T](
    stream: Stream[T],
    env: dp.PolicyEnv,
    num_generated: int,
    strict: bool = True,
):
    return stream_take(stream.gen(), num_generated, strict)


@stream_transformer
def loop[T](
    stream: Stream[T],
    env: dp.PolicyEnv,
    n: int | None = None,
    *,
    stop_on_reject: bool = True,
) -> dp.StreamGen[T]:
    """
    Stream transformer that repeatedly respawns the underlying stream,
    up to an (optional) limit.
    """

    return stream.loop(n, stop_on_reject=stop_on_reject).gen()


#####
##### Basic Operations on Streams
#####


@dataclass(frozen=True)
class SpendingDeclined:
    pass


def spend_on[T](
    f: Callable[[], tuple[T, dp.Budget]], /, estimate: dp.Budget
) -> dp.StreamContext[T | SpendingDeclined]:
    barrier = Barrier(estimate)
    yield barrier
    if barrier.allow:
        value, spent = f()
        yield Spent(budget=spent, barrier_id=barrier.id)
        return value
    else:
        yield Spent(budget=dp.Budget.zero(), barrier_id=barrier.id)
        return SpendingDeclined()


def stream_bind[A, B](
    stream: dp.StreamGen[A], f: Callable[[dp.Solution[A]], dp.StreamGen[B]]
) -> dp.StreamGen[B]:
    generated: list[dp.Solution[A]] = []
    num_pending = 0
    for msg in stream:
        match msg:
            case dp.Solution():
                generated.append(msg)
            case Barrier():
                num_pending += 1
                yield msg
                if generated:
                    # We don't allow new spending before `generated` is
                    # emptied.
                    msg.allow = False
            case Spent():
                num_pending -= 1
                yield msg
        if num_pending == 0:
            while generated:
                yield from f(generated.pop(0))
    assert not generated
    assert num_pending == 0


def stream_take_one[T](
    stream: dp.StreamGen[T],
) -> dp.StreamContext[dp.Solution[T] | None]:
    num_pending = 0
    generated: list[dp.Solution[T]] = []
    for msg in stream:
        match msg:
            case Barrier():
                num_pending += 1
                yield msg
                if generated:
                    msg.allow = False
            case Spent():
                num_pending -= 1
                yield msg
            case dp.Solution():
                generated.append(msg)
        if generated and num_pending == 0:
            break
    assert num_pending == 0
    return None if not generated else generated[0]


def stream_take_all[T](
    stream: dp.StreamGen[T],
) -> dp.StreamContext[Sequence[dp.Solution[T]]]:
    res: list[dp.Solution[T]] = []
    for msg in stream:
        if isinstance(msg, dp.Solution):
            res.append(msg)
            continue
        yield msg
    return res


def stream_with_budget[T](
    stream: dp.StreamGen[T], budget: dp.BudgetLimit
) -> dp.StreamGen[T]:
    """
    See `with_budget` for a version wrapped as a stream transformer.
    """
    total_spent = dp.Budget.zero()
    # Map the id of a barrier to the frozen budget.
    pending: dict[BarrierId, dp.Budget] = {}

    for msg in stream:
        yield msg
        match msg:
            case Barrier(pred):
                bound = (
                    total_spent
                    + pred
                    + sum(pending.values(), start=dp.Budget.zero())
                )
                if not (bound <= budget):
                    msg.allow = False
                pending[msg.id] = msg.budget if msg.allow else dp.Budget.zero()
            case Spent(spent):
                total_spent = total_spent + spent
                assert msg.barrier_id in pending
                del pending[msg.barrier_id]
            case _:
                pass
    assert not pending


def stream_take[T](
    stream: dp.StreamGen[T], num_generated: int, strict: bool = True
) -> dp.StreamGen[T]:
    """
    See `take` for a version wrapped as a stream transformer.
    """
    count = 0
    num_pending = 0
    if not (num_generated > 0):
        return
    for msg in stream:
        match msg:
            case Barrier():
                yield msg
                if count >= num_generated:
                    msg.allow = False
                num_pending += 1
            case Spent():
                yield msg
                num_pending -= 1
            case dp.Solution():
                count += 1
                if not (strict and count > num_generated):
                    yield msg
        if num_pending == 0 and count >= num_generated:
            break
    assert num_pending == 0


def stream_collect[T](
    stream: dp.StreamGen[T],
) -> tuple[Sequence[dp.Solution[T]], dp.Budget]:
    total = dp.Budget.zero()
    elts: list[dp.Solution[T]] = []
    for msg in stream:
        if isinstance(msg, dp.Solution):
            elts.append(msg)
        if isinstance(msg, Spent):
            total = total + msg.budget
    return elts, total


type _StreamBuilder[T] = Callable[[], dp.StreamGen[T]]


def stream_or_else[T](
    main: _StreamBuilder[T], fallback: _StreamBuilder[T]
) -> dp.StreamGen[T]:
    some_successes = False
    for msg in main():
        if isinstance(msg, dp.Solution):
            some_successes = True
        yield msg
    if not some_successes:
        for msg in fallback():
            yield msg


def monitor_acceptance[T](
    stream: dp.StreamGen[T], on_accept: Callable[[], None]
) -> dp.StreamGen[T]:
    for msg in stream:
        # It is important to check `allow` AFTER yielding the message
        # because we are interested in whether the FINAL client accepts
        # the request.
        yield msg
        if isinstance(msg, Barrier):
            if msg.allow:
                on_accept()


def stream_sequence[T](
    streams: Iterable[_StreamBuilder[T]], *, stop_on_reject: bool = True
) -> dp.StreamGen[T]:
    for mk in streams:
        accepted = False

        def on_accept():
            nonlocal accepted
            accepted = True

        yield from monitor_acceptance(mk(), on_accept)
        if stop_on_reject and not accepted:
            break


def _stream_cons[T](
    elt: dp.Solution[T] | Spent | Barrier, stream: dp.StreamGen[T]
) -> dp.StreamGen[T]:
    yield elt
    yield from stream


def stream_next[T](
    stream: dp.StreamGen[T],
) -> dp.StreamContext[
    tuple[Sequence[dp.Solution[T]], dp.Budget, dp.StreamGen[T] | None]
]:
    total_spent = dp.Budget.zero()
    num_pending = 0
    done: bool = False  # We want to see at least one barrier
    generated: list[dp.Solution[T]] = []
    while True:
        msg = next(stream, None)
        match msg:
            case None:
                assert num_pending == 0
                return generated, total_spent, None
            case Barrier():
                if done:
                    if num_pending == 0:
                        return (
                            generated,
                            total_spent,
                            _stream_cons(msg, stream),
                        )
                    else:
                        yield msg
                        msg.allow = False
                else:
                    yield msg
                num_pending += 1
                done = True
            case Spent():
                yield msg
                num_pending -= 1
            case dp.Solution():
                generated.append(msg)


#####
##### Parralel Streams
#####


type _StreamElt[T] = dp.Solution[T] | Barrier | Spent


def stream_parallel[T](streams: Sequence[dp.StreamGen[T]]) -> dp.StreamGen[T]:
    import threading
    from queue import Queue
    from threading import Event

    # Each worker can push a pair of a message to transmit and of an
    # event to set whenever the client responded to the message. When
    # the worker is done, it pushes `None`.
    queue: Queue[tuple[_StreamElt[T], Event] | None | Exception] = Queue()

    # Number of workers that are still active.
    rem = len(streams)

    # Lock for protecting access to `progressed` and `sleeping`.
    lock = threading.Lock()
    # Whether progress was made since the last time `sleeping` was reset
    # (in the form of a new `Sent` message being sent to the client).
    # WHen set to true, one can retry all sleeping barriers. Otherwise,
    # one must decline at least one.
    progressed: bool = False
    # Each sleeping worker adds an element to this list, which is a
    # queue indicating whether or not the barrier element should be
    # forcibly declined.
    sleeping: list[Queue[bool]] = []

    def progress_made() -> None:
        nonlocal progressed
        with lock:
            progressed = True

    def sleep() -> bool:
        resp = Queue[bool]()
        with lock:
            sleeping.append(resp)
        check_sleeping()
        return resp.get()

    def check_sleeping() -> None:
        nonlocal progressed
        if len(sleeping) != rem:
            return
        with lock:
            for i, q in enumerate(sleeping):
                q.put(True if not progressed and i == 0 else False)
            sleeping.clear()
            progressed = False

    def send(msg: _StreamElt[T]) -> None:
        ev = Event()
        queue.put((msg, ev))
        # We wait this event to be sure that the message was received by
        # the client and `msg.allow` is set.
        ev.wait()

    def worker(stream: dp.StreamGen[T]):
        try:
            for msg in stream:
                send(msg)
                if isinstance(msg, Spent):
                    progress_made()
                if isinstance(msg, Barrier) and not msg.allow:
                    while not msg.allow:
                        force_cancel = sleep()
                        if force_cancel:
                            break
                        send(Spent(dp.Budget.zero(), msg.id))
                        msg.allow = True
                        send(msg)
            queue.put(None)
        except Exception as e:
            queue.put(e)

    # Launch all workers
    for s in streams:
        thread = threading.Thread(target=worker, args=(s,))
        thread.start()

    # Forward messages from workers until all of them are done.
    rem = len(streams)
    while rem > 0:
        elt = queue.get()
        if elt is None:
            rem -= 1
            check_sleeping()
        elif isinstance(elt, Exception):
            raise elt
        else:
            msg, ev = elt
            yield msg
            ev.set()

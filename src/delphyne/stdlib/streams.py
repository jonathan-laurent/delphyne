"""
Search streams and stream combinators.
"""

import itertools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, override

import delphyne.core as dp
from delphyne.core.streams import Barrier, BarrierId, Spent
from delphyne.stdlib.environments import PolicyEnv

#####
##### Search Streams
#####


@dataclass(frozen=True)
class Stream[T](dp.AbstractStream[T]):
    """
    A search stream produced by a search policy or prompting policy.

    This class inherits `AbstractStream` and supports various methods
    and combinators for assembling streams, while guaranteeing adherence
    to the [search stream protocol][delphyne.core.streams].

    Attributes:
        _generate: A zeroary function that produces a stream generator.
    """

    _generate: Callable[[], dp.StreamGen[T]]

    @override
    def gen(self) -> dp.StreamGen[T]:
        return self._generate()

    ## Collecting all elements

    def collect(
        self,
        budget: dp.BudgetLimit | None = None,
        num_generated: int | None = None,
    ) -> tuple[Sequence[dp.Solution[T]], dp.Budget]:
        """
        Exhaust a stream and collect all generated solutions.

        Attributes:
            budget (optional): Budget limit (see `with_budget` method).
            num_generated (optional): Number of solutions to generate
                (see `take` method).

        Returns:
            A sequence of solutions along with the total spent budget.

        !!! warning
            This function must only be used at top-level and *not*
            within the definition of a search stream generator (in which
            case the consumed resources won't be accounted for in the
            parent stream). See `Stream.all`.
        """
        if budget is not None:
            self = self.with_budget(budget)
        if num_generated is not None:
            self = self.take(num_generated)
        return stream_collect(self.gen())

    ## Transforming the stream

    def with_budget(self, budget: dp.BudgetLimit):
        """
        Return an identical stream that denies all spending requests
        once a given amount of budget is spent.

        **Guarantees**: if all budget spending over-estimates passed to
        `spend_on` are accurate, the given budget limit is rigorously
        respected. If not, the spent amount may exceed the budget by an
        amount of `Delta * N`, where `Delta` is the maximum estimation
        error and `N` is the concurrency level of the stream (1 if
        `Stream.parallel` is never used).
        """
        return Stream(lambda: stream_with_budget(self.gen(), budget))

    def take(self, num_generated: int, strict: bool = True):
        """
        Return an identical stream that terminates once a given number
        of solution is generated. If `strict` is set to `False`, more
        solutions can be returned, provided that no additional budget
        must be spent for generating them.
        """
        return Stream(lambda: stream_take(self.gen(), num_generated, strict))

    def loop(
        self, n: int | None = None, *, stop_on_reject: bool = True
    ) -> "Stream[T]":
        """
        Repeatedly execute a stream.

        Arguments:
            n (optional): Number of times to repeat the stream. By
                default, the stream is repeated indefinitely.
            stop_on_reject: If set to `True` (default), the resulting
                stream stops after the first iteration during which no
                spending request was granted. This guarantees
                termination, even if `n` is `None`.
        """
        it = itertools.count() if n is None else range(n)
        return Stream(
            lambda: stream_sequence(
                (self.gen for _ in it), stop_on_reject=stop_on_reject
            )
        )

    def bind[U](
        self, f: Callable[[dp.Solution[T]], dp.StreamGen[U]]
    ) -> "Stream[U]":
        """
        Apply a function to all generated solutions of a stream and
        concatenate the resulting streams.

        This is analogous to the `concat_map` funtion on lists:

            def concat_map(f, xs):
                return [y for x in xs for y in f(x)]
        """
        return Stream(lambda: stream_bind(self.gen(), f))

    ## Monadic Methods

    def first(self) -> dp.StreamContext[dp.Solution[T] | None]:
        """
        Obtain the first solution from a stream, or return `None` if the
        stream terminates without yielding any solution.
        """
        return stream_first(self.gen())

    def all(self) -> dp.StreamContext[Sequence[dp.Solution[T]]]:
        """
        Obtain all solutions from a stream.
        """
        return stream_all(self.gen())

    def next(
        self,
    ) -> dp.StreamContext[
        "tuple[Sequence[dp.Solution[T]], dp.Budget, Stream[T] | None]"
    ]:
        """
        Make an atomic attempt to obtain a solution from the stream,
        stopping right before a second spending request is made.

        Return a sequence of generated solutions, the total spent
        budget, and the remaining stream, if any.
        """
        gen, budg, rest = yield from stream_next(self.gen())
        new_rest = None if rest is None else Stream(lambda: rest)
        return gen, budg, new_rest

    ## Static Methods

    @staticmethod
    def sequence[U](
        streams: Iterable["Stream[U]"], *, stop_on_reject: bool = True
    ) -> "Stream[U]":
        """
        Concatenate all streams from a possibly infinite collection.

        If `stop_on_reject` is set to `True` (default), then the
        resulting stream is stopped as soon as one stream in the
        collection terminates without a single spending request being
        granted. This allows guaranteeing termination, even if an
        infinite collection of streams is passed.
        """
        return Stream(
            lambda: stream_sequence(
                (s.gen for s in streams), stop_on_reject=stop_on_reject
            )
        )

    @staticmethod
    def parallel[U](streams: Sequence["Stream[U]"]) -> "Stream[U]":
        """
        Run all streams of a sequence in separate threads, possibly
        interleaving the resulting solutions.
        """
        return Stream(lambda: stream_parallel([s.gen() for s in streams]))

    @staticmethod
    def or_else[U](main: "Stream[U]", fallback: "Stream[U]") -> "Stream[U]":
        """
        Run the `main` stream and, if it does not yield any solution,
        run the `fallback` stream.
        """
        return Stream(lambda: stream_or_else(main.gen, fallback.gen))


#####
##### Stream transformers
#####


class _StreamTransformerFn(Protocol):
    def __call__[T](
        self,
        stream: Stream[T],
        env: PolicyEnv,
    ) -> dp.StreamGen[T]: ...


class _ParametricStreamTransformerFn[**A](Protocol):
    def __call__[T](
        self,
        stream: Stream[T],
        env: PolicyEnv,
        *args: A.args,
        **kwargs: A.kwargs,
    ) -> dp.StreamGen[T]: ...


@dataclass
class StreamTransformer:
    """
    Wrapper for a function that maps a stream to another one, possibly
    depending on the global policy environment. Can be composed with
    policies, search policies and other stream transformers using the
    `@` operator.

    Attributes:
        trans: The wrapped stream transformer function.
    """

    trans: _StreamTransformerFn

    def __call__[T](
        self,
        stream: Stream[T],
        env: PolicyEnv,
    ) -> Stream[T]:
        return Stream(lambda: self.trans(stream, env))

    def __matmul__(self, other: "StreamTransformer") -> "StreamTransformer":
        """
        Compose this transformer with another one.
        """
        if not isinstance(other, StreamTransformer):  # pyright: ignore[reportUnnecessaryIsInstance]
            return NotImplemented

        def transformer[T](
            stream: Stream[T],
            env: PolicyEnv,
        ) -> dp.StreamGen[T]:
            return self(other(stream, env), env).gen()

        return StreamTransformer(transformer)


def stream_transformer[**A](
    f: _ParametricStreamTransformerFn[A],
) -> Callable[A, StreamTransformer]:
    """
    Convenience decorator for creating parametric stream transformers
    (i.e., functions that return stream transformers).

    See `take` for an example.

    Attributes:
        f: A function that takes a stream, a policy environment, and
            additional parameters, and returns a stream generator.

    Returns:
        A function that takes the additional parameters of `f` and
        returns a `StreamTransformer` object.
    """

    def parametric(*args: A.args, **kwargs: A.kwargs) -> StreamTransformer:
        def transformer[T](
            stream: Stream[T],
            env: PolicyEnv,
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
        env: PolicyEnv,
    ) -> dp.StreamGen[T]: ...


@dataclass
class StreamCombinator:
    combine: _StreamCombinatorFn

    def __call__[T](
        self,
        streams: Sequence[Stream[T]],
        probs: Sequence[float],
        env: PolicyEnv,
    ) -> Stream[T]:
        return Stream(lambda: self.combine(streams, probs, env))

    def __rmatmul__(self, other: StreamTransformer) -> "StreamCombinator":
        if not isinstance(other, StreamTransformer):  # pyright: ignore[reportUnnecessaryIsInstance]
            return NotImplemented

        def combinator[T](
            streams: Sequence[Stream[T]],
            probs: Sequence[float],
            env: PolicyEnv,
        ) -> dp.StreamGen[T]:
            return other(self(streams, probs, env), env).gen()

        return StreamCombinator(combinator)


#####
##### Standard Stream Transformers
#####


@stream_transformer
def with_budget[T](
    stream: Stream[T],
    env: PolicyEnv,
    budget: dp.BudgetLimit,
):
    """
    Stream transformer version of `Stream.with_budget`.
    """
    return stream_with_budget(stream.gen(), budget)


@stream_transformer
def take[T](
    stream: Stream[T],
    env: PolicyEnv,
    num_generated: int,
    strict: bool = True,
):
    """
    Stream transformer version of `Stream.take`.
    """
    return stream_take(stream.gen(), num_generated, strict)


@stream_transformer
def loop[T](
    stream: Stream[T],
    env: PolicyEnv,
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
    """
    Sentinel value indicating that a spending request was declined.
    """

    pass


def spend_on[T](
    f: Callable[[], tuple[T, dp.Budget]], /, estimate: dp.Budget
) -> dp.StreamContext[T | SpendingDeclined]:
    """
    Perform a computation that requires spending some resources.

    Attributes:
        f: A zeroary function that returns the computation result, along
            with the budget spent on the computation.
        estimate: An over-estimate of the budget that is consumed by the
            computation. This estimate is allowed to be inaccurate. See
            `Stream.with_budget` for the provided guarantees.

    Returns:
        In a stream context, the value returned by the computation or an
        instance of `SpendingDeclined` if the spending request was
        declined.
    """
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
    """
    See `Stream.bind` for documentation.
    """
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


def stream_first[T](
    stream: dp.StreamGen[T],
) -> dp.StreamContext[dp.Solution[T] | None]:
    """
    See `Stream.first` for documentation.
    """
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


def stream_all[T](
    stream: dp.StreamGen[T],
) -> dp.StreamContext[Sequence[dp.Solution[T]]]:
    """
    See `Stream.all` for documentation.
    """
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
    See `Stream.with_budget` for documentation.
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
    See `Stream.take` for documentation.
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
    """
    See `Stream.collect` for documentation.
    """
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
    """
    See `Stream.or_else` for documentation.
    """
    some_successes = False
    for msg in main():
        if isinstance(msg, dp.Solution):
            some_successes = True
        yield msg
    if not some_successes:
        for msg in fallback():
            yield msg


def _monitor_acceptance[T](
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
    """
    See `Stream.sequence` for documentation.
    """
    for mk in streams:
        accepted = False

        def on_accept():
            nonlocal accepted
            accepted = True

        yield from _monitor_acceptance(mk(), on_accept)
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
    """
    See `Stream.next` for documentation.
    """
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


class _WorkerDone(Exception):
    pass


def stream_parallel[T](streams: Sequence[dp.StreamGen[T]]) -> dp.StreamGen[T]:
    """
    See `Stream.parallel` for documentation.
    """

    # Handling the trivial cases where there are less than two streams.
    if not streams:
        return
    if len(streams) == 1:
        yield from streams[0]
        return

    import threading
    from queue import Queue
    from threading import Event

    # Each worker can push a pair of a message to transmit and of an
    # event to set whenever the client responded to the message. When
    # the worker is done, it pushes `None`.
    queue: Queue[tuple[_StreamElt[T], Event] | None | Exception] = Queue()

    # Number of workers that are still active.
    rem = len(streams)

    # Flag to set when all workers are done.
    done = False

    # Lock for protecting access to `progressed` and `sleeping`.
    lock = threading.Lock()
    # Whether progress was made since the last time `sleeping` was reset
    # (in the form of a new `Sent` message being sent to the client).
    # When set to true, one can retry all sleeping barriers. Otherwise,
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
        force_cancel_resp = Queue[bool]()
        with lock:
            sleeping.append(force_cancel_resp)
        check_sleeping()
        return force_cancel_resp.get()

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
        if done:
            raise _WorkerDone()

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
        except _WorkerDone:
            queue.put(None)
        except Exception as e:
            queue.put(e)

    # Launch all workers
    for s in streams:
        thread = threading.Thread(target=worker, args=(s,))
        thread.start()

    # Forward messages from workers until all of them are done.
    rem = len(streams)
    exn = None  # Exception raised by a worker, if any.
    while rem > 0:
        elt = queue.get()
        if elt is None:
            rem -= 1
            check_sleeping()
        elif isinstance(elt, Exception):
            exn = elt
            done = True
            rem -= 1
            check_sleeping()
        else:
            msg, ev = elt
            try:
                if not done:
                    yield msg
            except GeneratorExit:
                # The generator was garbage collected and must exit
                done = True
            ev.set()
    if exn is not None:
        raise exn

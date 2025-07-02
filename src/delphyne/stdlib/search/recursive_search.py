"""
Recursive Search

A generic search algorithm that leverages stream combinators attached as
metadata on Branch and Join nodes.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import delphyne.core as dp
from delphyne.core.streams import Stream, Yield
from delphyne.stdlib.nodes import Branch, Failure, Join
from delphyne.stdlib.policies import log, search_policy
from delphyne.stdlib.queries import ProbInfo
from delphyne.stdlib.streams import StreamBuilder, take_one, take_one_with_meta

#####
##### Meta Annotations
#####


@dataclass
class GiveUp:
    """
    Instruct to treat a Branch node as a failure node.
    """

    pass


@dataclass
class VisitOne:
    """
    Annotation for a branching node, indicating that only one child must
    be visited.
    """

    pass


class StreamBuilderCombinator(Protocol):
    def __call__[T](
        self, streams: Sequence[StreamBuilder[T]], distr: Sequence[float]
    ) -> dp.Stream[T]: ...


@dataclass
class FiniteStreamDistr:
    """
    Extract one candidate annotated with `ProbInfo`, and combine
    children streams according to the provided distribution.
    """

    combine: StreamBuilderCombinator


#####
##### Main Algorithm
#####


@search_policy
def recursive_search[P, T](
    tree: dp.Tree[Branch | Join | Failure, P, T],
    env: dp.PolicyEnv,
    policy: P,
) -> Stream[T]:
    match tree.node:
        case dp.Success(x):
            yield Yield(x)
        case Failure():
            return
        case Branch(cands):
            assert tree.node.meta is not None
            meta = tree.node.meta(policy)
            cands_space = cands.stream(env, policy)
            if isinstance(meta, GiveUp):
                return
            elif isinstance(meta, VisitOne):
                elt = yield from take_one(cands_space)
                if elt is None:
                    return
                yield from recursive_search()(tree.child(elt), env, policy)
            elif isinstance(meta, FiniteStreamDistr):
                res = yield from take_one_with_meta(cands.stream(env, policy))
                if res is None:
                    log(env, "classifier_failure", loc=tree)
                    return
                _, pinfo = res
                assert isinstance(pinfo, ProbInfo), "Missing logprobs."
                distr = pinfo.distr
                probs = [x[1] for x in distr]
                streams = [
                    lambda: recursive_search()(tree.child(x[0]), env, policy)
                    for x in distr
                ]
                yield from meta.combine(streams, probs)
            else:
                assert False, f"Unknown metadata: {meta}"
        case Join(subs):
            # TODO: allow customizing the behavior, and in particular
            # generating candidates in parallel.
            elts: list[dp.Tracked[Any]] = []
            for s in subs:
                substream = recursive_search()(s.spawn_tree(), env, policy)
                elt = yield from take_one(substream)
                if elt is None:
                    return
                elts.append(elt)
            yield from recursive_search()(tree.child(elts), env, policy)


#####
##### Useful Combinators
#####


def combine_via_repeated_sampling(
    max_attempts: int | None = None,
) -> StreamBuilderCombinator:
    def combine[T](
        streams: Sequence[StreamBuilder[T]], distr: Sequence[float]
    ) -> dp.Stream[T]:
        i = 0
        while max_attempts is None or i < max_attempts:
            i += 1
            builder = random.choices(streams, weights=distr)[0]
            yield from builder()

    return combine

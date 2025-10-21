"""
Recursive Search

A generic search algorithm that leverages stream combinators attached as
metadata on Branch and Join nodes.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import delphyne.core as dp
from delphyne.core.streams import StreamGen
from delphyne.stdlib.environments import PolicyEnv
from delphyne.stdlib.nodes import Branch, Fail, Join, NodeMeta
from delphyne.stdlib.policies import search_policy, unsupported_node
from delphyne.stdlib.queries import ProbInfo
from delphyne.stdlib.streams import Stream, StreamCombinator

#####
##### Meta Annotations
#####


@dataclass
class GiveUp(NodeMeta):
    """
    Instruct to treat a Branch node as a failure node.
    """

    pass


@dataclass
class VisitOne(NodeMeta):
    """
    Annotation for a branching node, indicating that only one child must
    be visited.
    """

    pass


@dataclass
class CombineStreamDistr(NodeMeta):
    """
    Extract one candidate annotated with `ProbInfo`, and combine
    children streams according to the provided distribution.
    """

    combine: StreamCombinator


@dataclass
class OneOfEachSequentially(NodeMeta):
    """
    Most basic behaviour for a Join node, where one element of each
    subspace is extracted sequentially and the resulting child visited
    once.
    """

    pass


#####
##### Main Algorithm
#####


@search_policy
def recursive_search[P, T](
    tree: dp.Tree[Branch | Join | Fail, P, T],
    env: PolicyEnv,
    policy: P,
) -> StreamGen[T]:
    match tree.node:
        case dp.Success(x):
            yield dp.Solution(x)
        case Fail():
            return
        case Branch(cands):
            assert tree.node.meta is not None
            meta = tree.node.meta(policy)
            cands_space = cands.stream(env, policy)
            if isinstance(meta, GiveUp):
                return
            elif isinstance(meta, VisitOne):
                elt = yield from cands_space.first()
                if elt is None:
                    return
                rec = recursive_search()(tree.child(elt.tracked), env, policy)
                yield from rec
            elif isinstance(meta, CombineStreamDistr):
                res = yield from cands.stream(env, policy).first()
                if res is None:
                    env.info("classifier_failure", loc=tree)
                    return
                pinfo = res.meta
                assert isinstance(pinfo, ProbInfo), "Missing logprobs."
                distr = pinfo.distr
                probs = [x[1] for x in distr]
                streams = [
                    recursive_search()(tree.child(x[0]), env, policy)
                    for x in distr
                ]
                yield from meta.combine(streams, probs, env)
            else:
                assert False, f"Unsupported behavior for `Branch`: {meta}"
        case Join(subs):
            assert tree.node.meta is not None
            meta = tree.node.meta(policy)
            if isinstance(meta, OneOfEachSequentially):
                elts: list[dp.Tracked[Any]] = []
                for s in subs:
                    substream = recursive_search()(s.spawn_tree(), env, policy)
                    elt = yield from substream.first()
                    if elt is None:
                        return
                    elts.append(elt.tracked)
                yield from recursive_search()(tree.child(elts), env, policy)
            else:
                assert False, f"Unknown behavior for `Join`: {meta}"
        case _:
            unsupported_node(tree.node)


#####
##### Useful Combinators
#####


def combine_via_repeated_sampling(
    max_attempts: int | None = None,
) -> StreamCombinator:
    def combine[T](
        streams: Sequence[Stream[T]],
        probs: Sequence[float],
        env: PolicyEnv,
    ) -> dp.StreamGen[T]:
        i = 0
        while max_attempts is None or i < max_attempts:
            i += 1
            stream = env.random.choices(streams, weights=probs)[0]
            yield from stream

    return StreamCombinator(combine)

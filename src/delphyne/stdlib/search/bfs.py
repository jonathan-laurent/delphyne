"""
An implementation of best-first search.
"""

import heapq
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import delphyne.core as dp
from delphyne.core import refs
from delphyne.stdlib.nodes import Failure, spawn_node
from delphyne.stdlib.policies import search_policy


@dataclass(frozen=True)
class BFSFactor(dp.Node):
    """
    A node that allows computing a confidence score in the [0, 1]
    interval. This confidence can be computed by a query or a dedicated
    strategy but only one element will be generated from the resulting
    space.
    """

    confidence: dp.OpaqueSpace[Any, float]

    def navigate(self) -> dp.Navigation:
        return ()
        yield


@dataclass(frozen=True)
class BFSBranch(dp.Node):
    """
    A BFS Branching Node.

    Confidence priors represent penalty factors to be applied to the
    node's total confidence depending on the number of candidates that
    were generated already. Element `i` of `self.confidence_priors`
    indicates the penalty to apply if `i` candidates have been generated
    already (the last element of this list repeats ad infinitum).
    """

    cands: dp.OpaqueSpace[Any, Any]
    confidence_priors: Callable[[Any], Sequence[float]]

    def navigate(self) -> dp.Navigation:
        return (yield self.cands)

    def primary_space(self):
        return self.cands


def bfs_branch[P, T](
    cands: dp.Builder[dp.OpaqueSpace[P, T]],
    *,
    confidence_priors: Callable[[P], Sequence[float]],
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[BFSBranch, P, T]:
    ret = yield spawn_node(
        BFSBranch, cands=cands, confidence_priors=confidence_priors
    )
    return cast(T, ret)


def bfs_factor[P](
    confidence: dp.Builder[dp.OpaqueSpace[P, float]]
) -> dp.Strategy[BFSFactor, P, float]:  # fmt: skip
    ret = yield spawn_node(BFSFactor, confidence=confidence)
    return cast(float, ret)


@dataclass(frozen=True)
class NodeState:
    # Instead of `children`, all that is really needed here is a count
    # of the number of children. We provide children ids for easier
    # debugging. Also, the children list must sometimes be mutated and
    # `heapq` forces this dataclass to be frozen so this is also
    # convenient (`heapq` is being overly conservative here because
    # mutability is not an issue as long as the part that is relevant
    # for comparison is immutable).
    children: list[refs.GlobalNodePath]  # can be mutated
    confidence: float
    confidence_priors: Sequence[float]
    stream: dp.Stream[Any]
    node: BFSBranch  # equal to tree.node, with a more precise type
    tree: dp.Tree[BFSBranch | BFSFactor | Failure, Any, Any]

    def confidence_prior(self) -> float:
        if not self.confidence_priors:
            return 1
        i = len(self.children)
        return self.confidence_priors[min(i, len(self.confidence_priors) - 1)]


@dataclass(frozen=True, order=True)
class PriorityItem:
    # Python's heapq module puts the element with the smallest value on
    # top and uses lexicographic ordering. We want the element with
    # _highest confidence_ on top, or in case of a tie, the one that has
    # been in the queue for the longest time.
    neg_confidence: float
    insertion_id: int
    node_state: NodeState  # comparison must never reach this point


@search_policy
async def bfs[P, T](
    tree: dp.Tree[BFSBranch | BFSFactor | Failure, P, T],
    env: dp.PolicyEnv,
    policy: P,
) -> dp.Stream[T]:
    """
    Best First Search Algorithm.

    Nodes can be branching nodes or factor nodes. Factor nodes feature a
    confidence score in the [0, 1] interval. The total confidence of any
    node in the tree is the product of all confidence factors found on
    the path from the root to this node. The algorithm stores all
    visited branching nodes in a priority queue. At every step, it picks
    the node with highest confidence and spends an atomic amount of
    effort trying to generate a new child. If it succeeds, the first
    descendant branching node is added to the tree and the algorithm
    continues.

    Also, the total confidence of each branching node is multiplied by
    an additional penalty factor that depends on how many children have
    been generated already (see `confidence_priors`).
    """
    # `counter` is used to assign ids that are used to solve ties in the
    # priority queue (the older element gets priority).
    counter = 0
    pqueue: list[PriorityItem] = []  # a heap

    async def push_fresh_node(
        tree: dp.Tree[BFSBranch | BFSFactor | Failure, Any, Any],
        confidence: float,
    ) -> dp.Stream[T]:
        match tree.node:
            case dp.Success():
                yield dp.Yield(tree.node.success)
            case Failure():
                pass
            case BFSFactor():
                conf_stream = tree.node.confidence.stream(env, policy)
                factor = None
                async for conf_msg in conf_stream:
                    if isinstance(conf_msg, dp.Yield):
                        factor = conf_msg.value
                        break
                    else:
                        yield msg
                if factor is not None:
                    confidence *= factor.value
                    push = push_fresh_node(tree.child(()), confidence)
                    async for push_msg in push:
                        yield push_msg
            case BFSBranch():
                state = NodeState(
                    children=[],
                    confidence=confidence,
                    confidence_priors=tree.node.confidence_priors(policy),
                    stream=tree.node.cands.stream(env, policy),
                    node=tree.node,
                    tree=tree,
                )
                nonlocal counter
                counter += 1
                item_confidence = confidence * state.confidence_prior()
                item = PriorityItem(-item_confidence, counter, state)
                heapq.heappush(pqueue, item)

    def reinsert_node(state: NodeState) -> None:
        nonlocal counter
        counter += 1
        item_confidence = state.confidence * state.confidence_prior()
        item = PriorityItem(-item_confidence, counter, state)
        heapq.heappush(pqueue, item)

    # Pust the root into the queue.
    async for resp in push_fresh_node(tree, 1.0):
        yield resp
    while pqueue:
        state = heapq.heappop(pqueue).node_state
        try:
            # We make an atomic attempt at generating a new candidate
            while True:
                msg = await anext(state.stream)
                if isinstance(msg, dp.Yield):
                    cand = msg.value
                    break
                yield msg
                if isinstance(msg, dp.Barrier):
                    cand = None
                    break
        except StopAsyncIteration:
            # No need to put the node back in the queue
            continue
        if cand is not None:
            child = state.tree.child(cand)
            state.children.append(child.ref)
            async for resp in push_fresh_node(child, 1):
                yield resp
        # We put the node back into the queue
        reinsert_node(state)

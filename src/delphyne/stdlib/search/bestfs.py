"""
An implementation of best-first search.
"""

import heapq
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import delphyne.core as dp
from delphyne.core import refs
from delphyne.stdlib.nodes import Branch, Factor, Failure, Value
from delphyne.stdlib.policies import search_policy


@dataclass(frozen=True)
class NodeState:
    # Instead of `children`, all that is really needed here is a count
    # of the number of children. We provide children ids for easier
    # debugging. Also, the children list must sometimes be mutated and
    # `heapq` forces this dataclass to be frozen so this is also
    # convenient (`heapq` is being overly conservative here because
    # mutability is not an issue as long as the part that is relevant
    # for comparison is immutable).
    depth: int
    children: list[refs.GlobalNodePath]  # can be mutated
    confidence: float
    stream: dp.Stream[Any]
    node: Branch  # equal to tree.node, with a more precise type
    tree: dp.Tree[Branch | Factor | Value | Failure, Any, Any]


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
async def best_first_search[P, T](
    tree: dp.Tree[Branch | Factor | Failure, P, T],
    env: dp.PolicyEnv,
    policy: P,
    child_confidence_prior: Callable[[int, int], float],
    max_depth: int | None = None,
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
    been generated already, using the `child_confidence_prior` argument.
    This argument is a function that takes as its first argument the
    depth of the current branching node (0 for the root, only
    incrementing when meeting other branching nodes) and as its second
    argument how many children have been generated so far. It returns
    the additional penalty to be added.

    The `max_depth` parameter indicates the maximum depth a branch node
    can have. The root has depth 0 and and only branch nodes count
    towards increasing the depth.
    """
    # `counter` is used to assign ids that are used to solve ties in the
    # priority queue (the older element gets priority).
    counter = 0
    pqueue: list[PriorityItem] = []  # a heap

    async def push_fresh_node(
        tree: dp.Tree[Branch | Factor | Value | Failure, Any, Any],
        confidence: float,
        depth: int,
    ) -> dp.Stream[T]:
        match tree.node:
            case dp.Success():
                yield dp.Yield(tree.node.success)
            case Failure():
                pass
            case Factor() | Value():
                eval_stream = tree.node.eval.stream(env, policy)
                eval = None
                async for eval_msg in eval_stream:
                    if isinstance(eval_msg, dp.Yield):
                        eval = eval_msg.value.value
                        break
                    else:
                        yield msg
                if eval is not None:
                    if isinstance(tree.node, Value):
                        confidence = tree.node.value(policy)(eval)
                    else:
                        confidence *= tree.node.factor(policy)(eval)
                    push = push_fresh_node(tree.child(()), confidence, depth)
                    async for push_msg in push:
                        yield push_msg
            case Branch():
                if max_depth is not None and depth > max_depth:
                    return
                state = NodeState(
                    depth=depth,
                    children=[],
                    confidence=confidence,
                    stream=tree.node.cands.stream(env, policy),
                    node=tree.node,
                    tree=tree,
                )
                nonlocal counter
                counter += 1
                prior = child_confidence_prior(depth, 0)
                item_confidence = confidence * prior
                item = PriorityItem(-item_confidence, counter, state)
                heapq.heappush(pqueue, item)

    def reinsert_node(state: NodeState) -> None:
        nonlocal counter
        counter += 1
        prior = child_confidence_prior(state.depth, len(state.children))
        item_confidence = state.confidence * prior
        item = PriorityItem(-item_confidence, counter, state)
        heapq.heappush(pqueue, item)

    # Pust the root into the queue.
    async for resp in push_fresh_node(tree, 1.0, 0):
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
            async for resp in push_fresh_node(child, 1, state.depth + 1):
                yield resp
        # We put the node back into the queue
        reinsert_node(state)

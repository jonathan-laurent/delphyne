"""
An implementation of best-first search.
"""

import heapq
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import delphyne.core as dp
from delphyne.core import refs
from delphyne.stdlib.environments import PolicyEnv
from delphyne.stdlib.nodes import Branch, Factor, Fail, Value
from delphyne.stdlib.policies import search_policy, unsupported_node
from delphyne.stdlib.streams import Stream


@dataclass(frozen=True)
class _NodeState:
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
    stream: list[Stream[Any] | None]  # using one element (mutated)
    node: Branch  # equal to tree.node, with a more precise type
    tree: dp.Tree[Branch | Factor | Value | Fail, Any, Any]
    next_actions: list[dp.Tracked[Any]]  # can be mutated


@dataclass(frozen=True, order=True)
class _PriorityItem:
    # Python's heapq module puts the element with the smallest value on
    # top and uses lexicographic ordering. We want the element with
    # _highest confidence_ on top, or in case of a tie, the one that has
    # been in the queue for the longest time.
    neg_confidence: float
    insertion_id: int
    node_state: _NodeState  # comparison must never reach this point


@search_policy
def best_first_search[P, T](
    tree: dp.Tree[Branch | Factor | Value | Fail, P, T],
    env: PolicyEnv,
    policy: P,
    child_confidence_prior: Callable[[int, int], float],
    max_depth: int | None = None,
) -> dp.StreamGen[T]:
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
    pqueue: list[_PriorityItem] = []  # a heap

    def push_fresh_node(
        tree: dp.Tree[Branch | Factor | Value | Fail, Any, Any],
        confidence: float,
        depth: int,
    ) -> dp.StreamGen[T]:
        match tree.node:
            case dp.Success():
                yield dp.Solution(tree.node.success)
            case Fail():
                pass
            case Factor() | Value():
                if isinstance(tree.node, Value):
                    penalty_fun = tree.node.value(policy)
                else:
                    penalty_fun = tree.node.factor(policy)
                # Evaluate metrics if a penalty function is provided
                if penalty_fun is not None:
                    eval_stream = tree.node.eval.stream(env, policy)
                    eval = yield from eval_stream.first()
                    # If we failed to evaluate the metrics, we give up.
                    if eval is None:
                        return
                    if isinstance(tree.node, Value):
                        confidence = penalty_fun(eval.tracked.value)
                    else:
                        confidence *= penalty_fun(eval.tracked.value)
                yield from push_fresh_node(tree.child(None), confidence, depth)
            case Branch():
                if max_depth is not None and depth > max_depth:
                    return
                state = _NodeState(
                    depth=depth,
                    children=[],
                    confidence=confidence,
                    stream=[tree.node.cands.stream(env, policy)],
                    node=tree.node,
                    tree=tree,
                    next_actions=[],
                )
                nonlocal counter
                counter += 1
                prior = child_confidence_prior(depth, 0)
                item_confidence = confidence * prior
                item = _PriorityItem(-item_confidence, counter, state)
                heapq.heappush(pqueue, item)
            case _:
                unsupported_node(tree.node)

    def reinsert_node(state: _NodeState) -> None:
        nonlocal counter
        counter += 1
        prior = child_confidence_prior(state.depth, len(state.children))
        item_confidence = state.confidence * prior
        item = _PriorityItem(-item_confidence, counter, state)
        heapq.heappush(pqueue, item)

    # Put the root into the queue.
    yield from push_fresh_node(tree, 1.0, 0)
    while pqueue:
        state = heapq.heappop(pqueue).node_state
        if not state.next_actions:
            if not state.stream[0]:
                # No more actions to take, we do not put the node back.
                continue
            generated, _, next = yield from state.stream[0].next()
            state.next_actions.extend([a.tracked for a in generated])
            state.stream[0] = next
        if state.next_actions:
            cand = state.next_actions.pop(0)
            child = state.tree.child(cand)
            state.children.append(child.ref)
            yield from push_fresh_node(child, 1, state.depth + 1)
        # We put the node back into the queue
        reinsert_node(state)

"""
An implementation of best-first search
"""

import heapq
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from delphyne.core.refs import NodeId
from delphyne.core.trees import Navigation, Node, Strategy, Success, Tree
from delphyne.stdlib.dsl import GeneratorConvertible, convert_to_generator
from delphyne.stdlib.generators import GenEnv, Generator, GenResponse, GenRet
from delphyne.stdlib.nodeclasses import nodeclass
from delphyne.stdlib.nodes import Failure
from delphyne.stdlib.search_envs import HasSearchEnv


@nodeclass(frozen=True)
class BFSFactor[P](Node):
    confidence: Generator[P, float]

    def navigate(self) -> Navigation:
        return ()
        yield


@nodeclass(frozen=True)
class BFSBranch[P](Node):
    gen: Generator[P, Any]
    confidence_priors: Callable[[P], Sequence[float]]
    label: str | None

    def navigate(self) -> Navigation:
        return (yield self.gen)


def bfs_branch[P: HasSearchEnv, T](
    gen: GeneratorConvertible[P, T],
    *,
    confidence_priors: Callable[[P], Sequence[float]],
    label: str | None = None,
    param_type: type[P] | None = None
) -> Strategy[BFSBranch[P], T]:  # fmt: skip
    ret = yield BFSBranch(convert_to_generator(gen), confidence_priors, label)
    return cast(T, ret)


def bfs_factor[P: HasSearchEnv](
    confidence: GeneratorConvertible[P, float]
) -> Strategy[BFSFactor[P], float]:  # fmt: skip
    ret = yield BFSFactor(convert_to_generator(confidence))
    return cast(float, ret)


type BFSNode[P] = BFSBranch[P] | BFSFactor[P] | Failure


type BFS[P, T] = Strategy[BFSNode[P], T]


@dataclass(frozen=True)
class NodeState[P]:
    children: list[NodeId]  # can be mutated
    confidence: float
    confidence_priors: Sequence[float]
    gen: GenRet[Any]
    node: BFSBranch[P]  # equal to tree.node, with a more precise type
    tree: Tree[BFSNode[P], Any]

    def confidence_prior(self) -> float:
        if not self.confidence_priors:
            return 1
        i = len(self.children)
        return self.confidence_priors[min(i, len(self.confidence_priors) - 1)]


@dataclass(frozen=True, order=True)
class PriorityItem[P]:
    neg_confidence: float
    insertion_id: int
    node_state: NodeState[P]  # comparison must never reach this point


async def bfs[P, T](
    env: GenEnv,
    tree: Tree[BFSNode[P], T],
    params: P,
) -> GenRet[T]:  # fmt: skip

    # `counter` is used to assign ids that are used to solve ties in the
    # priority queue
    counter = 0
    pqueue: list[PriorityItem[P]] = []  # a heap

    async def push_fresh_node(
        tree: Tree[BFSNode[P], Any], confidence: float
    ) -> GenRet[T]:
        match tree.node:
            case Success():
                yield GenResponse([tree.node.success])
            case Failure():
                yield GenResponse([])
            case BFSFactor():
                conf_gen = tree.node.confidence(env, tree, params)
                async for conf_resp in conf_gen:
                    if not conf_resp.items:
                        yield GenResponse([])
                    else:
                        confidence *= conf_resp.items[0].value
                        break
                async for resp in push_fresh_node(tree.child(()), confidence):
                    yield resp
            case BFSBranch():
                state = NodeState(
                    children=[],
                    confidence=confidence,
                    confidence_priors=tree.node.confidence_priors(params),
                    gen=tree.node.gen(env, tree, params),
                    node=tree.node,
                    tree=tree,
                )
                nonlocal counter
                counter += 1
                item_confidence = confidence * state.confidence_prior()
                item = PriorityItem(-item_confidence, counter, state)
                heapq.heappush(pqueue, item)
                yield GenResponse([])

    def reinsert_node(state: NodeState[P]) -> None:
        nonlocal counter
        counter += 1
        item_confidence = state.confidence * state.confidence_prior()
        item = PriorityItem(-item_confidence, counter, state)
        heapq.heappush(pqueue, item)

    async for resp in push_fresh_node(tree, 1.0):
        yield resp
    while pqueue:
        state = heapq.heappop(pqueue).node_state
        try:
            # We only try one step of generating a new solution
            gen_resp = await anext(state.gen)
            if not gen_resp.items:
                yield GenResponse([])
        except StopAsyncIteration:
            # No need to put the node back in the queue
            continue
        for gen_item in gen_resp.items:
            child = state.tree.child(gen_item)
            state.children.append(child.node_id)
            async for resp in push_fresh_node(child, 1):
                yield resp
        # We put the node back in the queue
        reinsert_node(state)

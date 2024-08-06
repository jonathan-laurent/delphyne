"""
The basic DFS search policy.
"""

from abc import ABC, abstractmethod

from delphyne.core.trees import Success, Tree
from delphyne.stdlib.generators import (
    BudgetCounter,
    GenEnv,
    GenResponse,
    GenRet,
)
from delphyne.stdlib.nodes import Branch, BranchingStrategyNode, Failure


class HasMaxDepth(ABC):
    @abstractmethod
    def get_max_depth(self) -> int | None:
        pass


def _max_depth(params: object) -> int | None:
    if isinstance(params, HasMaxDepth):
        return params.get_max_depth()
    return None


async def dfs[P, T](
    env: GenEnv,
    tree: Tree[BranchingStrategyNode[P], T],
    params: P,
    depth: int = 0
) -> GenRet[T]:  # fmt: skip
    match tree.node:
        case Success(v):
            yield GenResponse([v])
        case Failure():
            yield GenResponse([])
        case Branch():
            if (maxd := _max_depth(params)) is not None and depth >= maxd:
                return
            i = 0
            n = tree.node.max_branching(params)
            gen_counter = BudgetCounter(tree.node.max_gen_budget(params))
            gen_env = env.with_counter(gen_counter)
            async for gen_ret in tree.node.gen(gen_env, tree, params):
                if not gen_ret.items:
                    yield gen_ret
                for c in gen_ret.items:
                    i += 1
                    if n is not None and i > n:
                        return
                    ccounter = BudgetCounter(tree.node.max_cont_budget(params))
                    cont_env = env.with_counter(ccounter)
                    recursive = dfs(cont_env, tree.child(c), params, depth + 1)
                    async for ret in recursive:
                        yield ret
                        if not cont_env.budget_left():
                            break
                if not gen_env.budget_left():
                    return

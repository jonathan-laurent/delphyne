"""
Depth-First Search Algorithm
"""

from delphyne.core.environments import PolicyEnv
from delphyne.core.streams import Solution, Stream
from delphyne.core.trees import Success, Tree
from delphyne.stdlib.nodes import Branch, Fail
from delphyne.stdlib.policies import search_policy, unsupported_node


@search_policy
def dfs[P, T](
    tree: Tree[Branch | Fail, P, T],
    env: PolicyEnv,
    policy: P,
    max_depth: int | None = None,
    max_branching: int | None = None,
) -> Stream[T]:
    """
    Depth-first search

    If set, `max_depth` defines the maximum number of branching nodes
    that can be traversed in a path to success.
    """
    assert max_branching is None or max_branching > 0
    match tree.node:
        case Success(x):
            yield Solution(x)
        case Fail():
            pass
        case Branch(cands):
            if max_depth is not None and max_depth <= 0:
                return
            cands = cands.stream(env, policy)
            if max_branching is not None:
                cands = cands.take(max_branching, strict=True)
            yield from cands.bind(
                lambda a: dfs(
                    max_depth=max_depth - 1 if max_depth is not None else None,
                    max_branching=max_branching,
                )(tree.child(a.tracked), env, policy).gen()
            ).gen()
        case _:
            unsupported_node(tree.node)

"""
Depth-First Search Algorithm
"""

from delphyne.core.streams import Solution, StreamGen
from delphyne.core.trees import Success, Tree
from delphyne.stdlib.environments import PolicyEnv
from delphyne.stdlib.nodes import Branch, Fail
from delphyne.stdlib.policies import search_policy, unsupported_node
from delphyne.stdlib.streams import Stream


@search_policy
def dfs[P, T](
    tree: Tree[Branch | Fail, P, T],
    env: PolicyEnv,
    policy: P,
    max_depth: int | None = None,
    max_branching: int | None = None,
) -> StreamGen[T]:
    """
    The Standard Depth-First Search Algorithm.

    Whenever a branching node is encountered, branching candidates are
    lazily enumerated and the corresponding child recursively searched.

    Attributes:
        max_depth (optional): maximum number of branching nodes
            that can be traversed in a path to success.
        max_branching (optional): maximum number of children explored at
            each branching node.
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


@search_policy
def par_dfs[P, T](
    tree: Tree[Branch | Fail, P, T],
    env: PolicyEnv,
    policy: P,
) -> StreamGen[T]:
    """
    Parallel Depth-First Search.

    Whenever a branching node is encountered, all branching candidates
    are computed at once and the associated children are explored in
    parallel.
    """
    match tree.node:
        case Success(x):
            yield Solution(x)
        case Fail():
            pass
        case Branch(cands):
            cands = yield from cands.stream(env, policy).all()
            yield from Stream.parallel(
                [par_dfs()(tree.child(a.tracked), env, policy) for a in cands]
            ).gen()
        case _:
            unsupported_node(tree.node)

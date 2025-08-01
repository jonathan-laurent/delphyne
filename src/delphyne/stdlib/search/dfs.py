"""
Depth-First Search Algorithm
"""

from delphyne.core.environments import PolicyEnv
from delphyne.core.streams import SearchValue, Stream, Yield
from delphyne.core.trees import Success, Tree
from delphyne.stdlib.nodes import Branch, Fail
from delphyne.stdlib.policies import search_policy


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
            yield Yield(SearchValue(x))
        case Fail():
            pass
        case Branch(cands):
            if max_depth is not None and max_depth <= 0:
                return
            branches_explored = 0
            for cands_msg in cands.stream(env, policy).gen():
                if not isinstance(cands_msg, Yield):
                    yield cands_msg
                    continue
                child = tree.child(cands_msg.value.value)
                rec = dfs(
                    max_depth=max_depth - 1 if max_depth is not None else None,
                    max_branching=max_branching,
                )
                yield from rec(child, env, policy).gen()
                branches_explored += 1
                if (
                    max_branching is not None
                    and branches_explored >= max_branching
                ):
                    break

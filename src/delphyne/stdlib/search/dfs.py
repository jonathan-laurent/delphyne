"""
Depth-First Search Algorithm
"""

from delphyne.core.environments import PolicyEnv
from delphyne.core.streams import Stream, Yield
from delphyne.core.trees import Success, Tree
from delphyne.stdlib.nodes import Branch, Failure
from delphyne.stdlib.policies import search_policy


@search_policy
def dfs[P, T](
    tree: Tree[Branch | Failure, P, T],
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
            yield Yield(x)
        case Failure():
            pass
        case Branch(cands):
            if max_depth is not None and max_depth <= 0:
                return
            for i, cands_msg in enumerate(cands.stream(env, policy)):
                if not isinstance(cands_msg, Yield):
                    yield cands_msg
                    continue
                child = tree.child(cands_msg.value)
                max_depth = max_depth - 1 if max_depth is not None else None
                yield from dfs(max_depth)(child, env, policy)
                if max_branching is not None and i + 1 >= max_branching:
                    break

"""
Some policies leveraging classification.
"""

import numpy as np

import delphyne as dp
from delphyne.stdlib.nodes import Branch
from delphyne.stdlib.policies import log, search_policy
from delphyne.stdlib.queries import ProbInfo
from delphyne.stdlib.streams import take_one_with_meta


@search_policy
def sample_and_proceed[N: dp.Node, P, T](
    tree: dp.Tree[Branch | N, P, T],
    env: dp.PolicyEnv,
    policy: P,
    proceed_with: dp.AbstractSearchPolicy[Branch | N],
) -> dp.Stream[T]:
    match tree.node:
        case dp.Success(x):
            yield dp.Yield(x)
        case dp.Branch(cands):
            res = yield from take_one_with_meta(cands.stream(env, policy))
            if res is None:
                log(env, "classifier_failure", loc=tree)
                return
            _, meta = res
            assert isinstance(meta, ProbInfo), "Missing logprobs."
            distr = meta.distr
            values = [x[0] for x in distr]
            probs = [x[1] for x in distr]
            k = np.random.choice(len(values), p=probs)
            selected = values[k]
            yield from proceed_with(tree.child(selected), env, policy)
        case _:
            assert False, "Expected branching or success node at the root."

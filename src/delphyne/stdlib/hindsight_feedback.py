"""
Defining the standard `Hindsight` effect for hindsight feedback.
"""

from dataclasses import dataclass
from typing import Any, Never

import delphyne.core as dp
import delphyne.stdlib.policies as pol


@dataclass
class Hindsight(dp.Node):
    """
    The standard `Hindsight` effect.
    """

    pass

    def navigate(self) -> dp.Navigation:
        return None
        yield


@pol.contextual_tree_transformer
def elim_hindsight(
    env: pol.PolicyEnv,
    policy: Any,
) -> pol.PureTreeTransformerFn[Hindsight, Never]:
    """
    Eliminate the `Hindsight` effect.

    This transformer populates the `hindsight_feedback` field of
    `PolicyEnv`.
    """

    def transform[N: dp.Node, P, T](
        tree: dp.Tree[Hindsight | N, P, T],
    ) -> dp.Tree[N, P, T]:
        if isinstance(tree.node, Hindsight):
            return transform(tree.child(None))
        return tree.transform(tree.node, transform)

    return transform

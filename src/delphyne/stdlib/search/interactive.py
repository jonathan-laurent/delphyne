from collections.abc import Callable, Sequence
from typing import Any

import delphyne.core as dp

# import delphyne.stdlib.queries as dq
from delphyne.stdlib import models as md

# from delphyne.stdlib.nodes import Failure, fail, spawn_node
# from delphyne.stdlib.policies import search_policy
# from delphyne.stdlib.strategies import strategy

# @strategy(name="interact")
# def _interact() -> dp.Strategy[]:
#     pass


def interact[P, A, B, T: md.AbstractTool[Any]](
    follow_up: Callable[[dp.AnswerPrefix], dp.OpaqueSpace[P, A]],
    process: Callable[[A], B | dp.Error],
    tools: Sequence[int] = (),
) -> dp.OpaqueSpace[P, B]:
    assert False

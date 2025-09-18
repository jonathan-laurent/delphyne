"""
Defining the standard `Hindsight` effect for hindsight feedback.
"""

from dataclasses import dataclass
from typing import Any, Literal, Never, overload

import delphyne.core as dp
import delphyne.stdlib.policies as pol
from delphyne.stdlib.environments import HindsightFeedback
from delphyne.stdlib.nodes import spawn_node


@dataclass
class Hindsight(dp.Node):
    """
    The standard `Hindsight` effect.

    This effect allows annotating the tree with feedback about what the
    answer to a particular query *should have been*.
    """

    query_name: str
    query_args: dict[str, object]
    hindsight_answer: dp.Answer

    def navigate(self) -> dp.Navigation:
        return None
        yield


@overload
def hindsight[T](
    query: dp.AbstractQuery[T],
    feedback: T,
) -> dp.Strategy[Hindsight, object, None]: ...


@overload
def hindsight(
    query: dp.AbstractQuery[Any],
    feedback: Any,
    *,
    as_parsed_answer: Literal[False],
) -> dp.Strategy[Hindsight, object, None]: ...


def hindsight(
    query: dp.AbstractQuery[Any],
    feedback: Any,
    *,
    as_parsed_answer: bool = True,
) -> dp.Strategy[Hindsight, object, None]:
    """
    Report some hindsight feedback.

    See `Hindsight`.

    Arguments:
        query: The query for which we provide feedback.
        feedback: The feedback to provide.
        as_parsed_answer: If `True`, `feedback` is assumed to be
            a parsed answer (of type `T` if the query is of type
            `AbstractQuery[T]`). This argument is only used for type
            checking and is ignored at runtime.
    """

    answer = query.hindsight_answer(feedback)
    if answer is None:
        raise ValueError(
            "Could not obtained an answer from the provided feedback "
            f"for query of type {type(query)}:\n\n"
            f"{feedback}"
        )
    parsed = query.parse_answer(answer)
    if isinstance(parsed, dp.ParseError):
        raise ValueError(
            "Could not parse the hindsight answer generated "
            f"for query of type {type(query)}:\n\n"
            f"{answer}\n\n"
            f"Parse error:\n\n"
            f"{parsed}"
        )
    yield spawn_node(
        Hindsight,
        query_name=query.query_name(),
        query_args=query.serialize_args(),
        hindsight_answer=answer,
    )
    return None


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
            node_id = env.tracer.global_node_id(tree.ref).id
            feedback = HindsightFeedback(
                query=tree.node.query_name,
                args=tree.node.query_args,
                answer=tree.node.hindsight_answer,
            )
            env.add_hindsight_feedback(node_id, feedback)
            return transform(tree.child(None))
        return tree.transform(tree.node, transform)

    return transform

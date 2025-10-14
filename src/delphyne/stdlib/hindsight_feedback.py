"""
Defining the standard `Feedback` effect for hindsight feedback.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Generic, Never, TypeVar

import delphyne.core as dp
import delphyne.stdlib.nodes as nd
import delphyne.stdlib.policies as pol

#####
##### Feedback Messages
#####


T_co = TypeVar("T_co", covariant=True, contravariant=False)


@dataclass(frozen=True)
class HindsightMessage(Generic[T_co]):
    pass


@dataclass(frozen=True)
class GoodValue(HindsightMessage[Never]):
    pass


@dataclass(frozen=True)
class BadValue(HindsightMessage[Never]):
    error: dp.Error


@dataclass(frozen=True)
class Shortcut[T](HindsightMessage[T]):
    value: T


@dataclass(frozen=True)
class AttachedFeedback[T]:
    msg: HindsightMessage[T]
    dst: nd.TypedSpaceRef[T]


def send[T](
    msg: HindsightMessage[T], to: nd.TypedSpaceRef[T], /
) -> AttachedFeedback[T]:
    return AttachedFeedback(msg, to)


#####
##### Feedback Nodes
#####


@dataclass(frozen=True)
class Feedback(nd.Skippable):
    """
    The standard `Feedback` effect.
    """


@dataclass(frozen=True)
class ThrowFeedback(Feedback):
    """
    Feedback source.
    """

    label: str
    messages: Iterable[AttachedFeedback[Any]]


@dataclass(frozen=True)
class BackpropagateFeedback(Feedback):
    """
    Handler for backpropagating feedback.
    """

    label: str
    back: Callable[[HindsightMessage[Any]], Iterable[AttachedFeedback[Any]]]


#####
##### Triggers
#####


def feedback(
    label: str, messages: Iterable[AttachedFeedback[Any]]
) -> dp.Strategy[Feedback, object, None]:
    yield nd.spawn_node(ThrowFeedback, label=label, messages=messages)
    return None


def backward[T](
    label: str,
    res: T,
    back: Callable[[HindsightMessage[T]], Iterable[AttachedFeedback[Any]]],
) -> dp.Strategy[Feedback, object, None]:
    yield nd.spawn_node(BackpropagateFeedback, label=label, back=back)
    return None


#####
##### Transformers
#####


@pol.contextual_tree_transformer
def elim_feedback(
    env: pol.PolicyEnv,
    policy: Any,
) -> pol.PureTreeTransformerFn[Feedback, Never]:
    """
    Eliminate the `Feedback` effect, by removing all feedback nodes.
    """

    def transform[N: dp.Node, P, T](
        tree: dp.Tree[Feedback | N, P, T],
    ) -> dp.Tree[N, P, T]:
        if isinstance(tree.node, Feedback):
            return transform(tree.child(None))
        return tree.transform(tree.node, transform)

    return transform

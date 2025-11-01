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
class ValueFeedback(Generic[T_co]):
    pass


@dataclass(frozen=True)
class GoodValue(ValueFeedback[Never]):
    pass


@dataclass(frozen=True)
class BadValue(ValueFeedback[Never]):
    error: dp.Error


@dataclass(frozen=True)
class BetterValue[T](ValueFeedback[T]):
    value: T


@dataclass(frozen=True)
class BadValueAlso[T](ValueFeedback[T]):
    # This is for BadValue what BetterValue is for GoodValue.
    # This communicates that a specific answer is another wrong answer,
    # with a justification.
    value: T
    error: dp.Error


@dataclass(frozen=True)
class AttachedFeedback[T]:
    msg: ValueFeedback[T]
    dst: nd.TypedSpaceElementRef[T]


def send[T](
    msg: ValueFeedback[T], to: nd.TypedSpaceElementRef[T], /
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
    back: Callable[[ValueFeedback[Any]], Iterable[AttachedFeedback[Any]]]


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
    back: Callable[[ValueFeedback[T]], Iterable[AttachedFeedback[Any]]],
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

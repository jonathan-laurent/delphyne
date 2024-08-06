"""
Abstract tree interface.
"""

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from typing import Any, Iterable, Never, cast

from delphyne.core.queries import Query
from delphyne.core.refs import ChoiceArgRef, ChoiceRef, NodeId
from delphyne.core.tracing import Outcome, Tracer, Value, value_ref
from delphyne.utils.typing import NoTypeInfo, TypeAnnot


#####
##### Abstract tree interface
#####


type Navigation = Generator[Choice[Any], Outcome[Any], Value]
"""
Nodes have a `navigate` method that yields choices and expects
corresponding outcomes until it outputs an action.
"""


class Node(ABC):

    def get_label(self) -> str | None:
        if (choice := self.primary_choice()) is not None:
            return choice.label()
        return None

    def type_name(self) -> str:
        return self.__class__.__name__

    def valid_action(self, action: object) -> bool:
        """
        This method can be implemented optionally to validate actions.
        """
        return True

    def leaf_node(self) -> bool:
        return False

    @abstractmethod
    def navigate(self) -> Navigation:
        """
        See `Navigation` for more details.
        """
        pass

    def primary_choice(self) -> "Choice[object] | None":
        """
        Choice that is inferred by default in a choice outcome reference
        (see `ChoiceOutcomeRef`). For example, `aggregate([gen{''},
        gen{'foo bar'}])` can be abbreviated into `aggregate(['', 'foo
        bar'])` if `gen` is the primary choice for the current node.
        """
        return None

    def base_choices(self) -> "Iterable[Choice[object]]":
        """
        Choices that are always displayed in the UI, even if they do not
        appear in the trace. By default, the primary choice is treated
        as the only base choice (when specified).
        """
        if (choice := self.primary_choice()) is not None:
            return (choice,)
        return ()

    def summary_message(self) -> str | None:
        return None


@dataclass(frozen=True)
class Success[T](Node):
    success: Outcome[T]

    def leaf_node(self) -> bool:
        return True

    def valid_action(self, action: object) -> bool:
        return False

    def navigate(self) -> Navigation:
        assert False


type Strategy[N: Node, T] = Generator[N, object, T]


type StrategyComp[N: Node, T] = Callable[[], Strategy[N, T]]


class Tree[N: Node, T](ABC):
    """
    Abstract interface for trees.

    This interface denotes a tree that exposes nodes of type M, which is
    wrapped in such a way to expose nodes of type N <: M. The type
    parameter `M` along with the `wrap` method are needed to support
    tree wrappers in the presence of embedded sub-trees.
    """

    @property
    @abstractmethod
    def node(self) -> N | Success[T]:
        pass

    @abstractmethod
    def child(self, action: Value) -> "Tree[N, T]":
        pass

    @abstractmethod
    def spawn[L: Node, U](
        self, strategy: StrategyComp[L, U], origin: ChoiceRef
    ) -> "Tree[L, U]":  # fmt: skip
        pass

    @property
    @abstractmethod
    def node_id(self) -> NodeId:
        pass

    @property
    @abstractmethod
    def tracer(self) -> Tracer:
        pass

    @abstractmethod
    def return_type(self) -> TypeAnnot[T] | NoTypeInfo:
        pass


#####
##### Abstract choices
#####


@dataclass(frozen=True)
class StrategyInstance[N: Node, T]:
    """
    For inspection purposes, it is better to avoid passing anonymous
    functions. Rather, top-level functions should be used, combined with
    `functools.partial`.
    """

    strategy: StrategyComp[N, T]


@dataclass(frozen=True)
class FiniteChoice[T]:
    choices: Sequence[tuple[str, T]]
    qualified_hints_only: bool = True


type ChoiceSource[T] = (
    Query[Never, T] | StrategyInstance[Node, T] | FiniteChoice[T]
)


class Choice[T](ABC):

    @abstractmethod
    def label(self) -> str | None:
        pass

    @abstractmethod
    def source(self) -> ChoiceSource[T]:
        pass

    @abstractmethod
    def return_type(self) -> TypeAnnot[T] | NoTypeInfo:
        pass

    @abstractmethod
    def set_origin(self, origin: ChoiceRef) -> None:
        pass

    @abstractmethod
    def get_origin(self) -> ChoiceRef:
        pass


#####
##### Subchoices
#####


type ChoiceArg = int | Value


def choice_arg_ref(arg: ChoiceArg) -> ChoiceArgRef:
    if isinstance(arg, int):
        return arg
    return value_ref(arg)


def subchoice[F: Callable[..., Choice[object]]](method: F) -> F:
    """Decorator for wrapping subchoice methods.

    The wrapped method calls `set_origin` on the returned choice.
    Ideally, we would use the type system to enforce that the method
    only accepts arguments
    """

    @functools.wraps(method)
    def wrapped(self: Node, *args: ChoiceArg):
        label = method.__name__
        args_ref = tuple(choice_arg_ref(a) for a in args)
        choice_ref = (label, args_ref)
        choice = method(self, *args)
        choice.set_origin(choice_ref)
        return choice

    return cast(F, wrapped)

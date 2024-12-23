"""
The core tree datastructure.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from delphyne.core import inspect
from delphyne.core.queries import AbstractQuery
from delphyne.core.refs import Assembly, GlobalNodePath, ValueRef
from delphyne.utils.typing import NoTypeInfo, TypeAnnot


####
#### Tracked values
####


@dataclass
class Tracked[T]:
    """
    A tracked value, which associates a value with a reference.

    The `node` field does not appear in Orakell and is a global path
    (from the global origin) to the node the value is attached too.
    Having this field is useful to check at runtime that a tracked value
    passed as an argument to `child` is attached to the current node.
    """

    value: T
    ref: ValueRef
    node: GlobalNodePath
    type_annot: TypeAnnot[T] | NoTypeInfo


type Value = Assembly[Tracked[Any]]
"""
A dynamic assembly of tracked values.
"""


####
#### Choices
####


type Tag = str
"""
Tags for better navigation in traces and in the demo language.
"""


class Space[T](ABC):
    """
    Abstract type for a space.
    """

    @abstractmethod
    def tags(self) -> Sequence[Tag]:
        pass

    @abstractmethod
    def source(self) -> "StrategyComp[Any, T] | AbstractQuery[Any, T]":
        pass

    @abstractmethod
    def element_type(self) -> TypeAnnot[T] | NoTypeInfo:
        pass


####
#### Nodes
####


class Node(ABC):
    """
    Abstract type for a node.
    """

    def extra_tags(self) -> Sequence[Tag]:
        return []

    def tags(self) -> Sequence[Tag]:
        # TODO
        return self.extra_tags()

    def effect_name(self) -> str:
        return self.__class__.__name__

    def valid_action(self, action: object) -> bool:
        """
        This method can be implemented optionally to validate actions.
        """
        return True

    def leaf_node(self) -> bool:
        return False

    @abstractmethod
    def navigate(self) -> "Navigation":
        """
        See `Navigation` for more details.
        """
        pass

    def primary_choice(self) -> "Space[object] | None":
        """
        Choice that is inferred by default in a choice outcome reference
        (see `SpaceElementRef`). For example, `compare([cands{''},
        cands{'foo bar'}])` can be abbreviated into `compare(['', 'foo
        bar'])` if `cands` is the primary choice for the current node.
        """
        return None

    def base_choices(self) -> "Iterable[Space[object]]":
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


type Navigation = Generator[Space[Any], Tracked[Any], Value]
"""
Nodes have a `navigate` method that yields choices and expects
corresponding outcomes until it outputs an action.
"""


@dataclass(frozen=True)
class Success[T](Node):
    success: Tracked[T]

    def leaf_node(self) -> bool:
        return True

    def valid_action(self, action: object) -> bool:
        return False

    def navigate(self) -> Navigation:
        assert False


####
#### The Tree Type
####

type Strategy[N: Node, P, T] = Generator[N, object, T]


@dataclass
class StrategyComp[N: Node, T]:
    """
    A strategy computation also stores metadata for debugging.
    """

    comp: Callable[..., Strategy[N, Any, T]]
    args: list[Any]
    kwargs: dict[str, Any]

    def __call__(self) -> Strategy[N, Any, T]:
        return self.comp(*self.args, **self.kwargs)

    def strategy_name(self) -> str | None:
        return inspect.function_name(self.comp)

    def tags(self) -> Sequence[Tag]:
        return [name] if (name := self.strategy_name()) is not None else []

    def return_type(self) -> TypeAnnot[T] | NoTypeInfo:
        ret_type = inspect.function_return_type(self.comp)
        if isinstance(ret_type, NoTypeInfo):
            return NoTypeInfo()
        return inspect.return_type_of_strategy_type(ret_type)


@dataclass(frozen=True)
class EmbeddedTree[T]:
    _comp: StrategyComp[Any, T]
    tree: "Callable[[], Tree[Any, T]]"

    @staticmethod
    def make(comp: StrategyComp[Any, T]) -> "EmbeddedTree[T]":
        assert False


@dataclass(frozen=True)
class Tree[N: Node, T]:
    node: N | Success[T]
    child: "Callable[[Value], Tree[N, T]]"
    ref: GlobalNodePath


####
#### Reifying Strategies into Trees
####


def _reify[N: Node, T](strategy: StrategyComp[N, T]) -> Tree[N, T]:
    assert False

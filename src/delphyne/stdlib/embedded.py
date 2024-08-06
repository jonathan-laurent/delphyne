"""
Embedded subtrees.
"""

from dataclasses import dataclass, field

from delphyne.core import inspect as std_inspect
from delphyne.core.refs import ChoiceRef
from delphyne.core.trees import Choice, Node, StrategyComp, Tree
from delphyne.utils.misc import Cell
from delphyne.utils.typing import NoTypeInfo, TypeAnnot


@dataclass(frozen=True)
class EmbeddedSubtree[N: Node, T](Choice[T]):
    strategy: StrategyComp[N, T]
    origin: Cell[ChoiceRef | None] = field(default_factory=lambda: Cell(None))

    def label(self) -> str | None:
        return std_inspect.underlying_strategy_name(self.strategy)

    def __call__[M: Node, U](self, parent: Tree[M, U]) -> Tree[N, T]:
        assert self.origin.content is not None
        return parent.spawn(self.strategy, self.origin.content)

    def return_type(self) -> TypeAnnot[T] | NoTypeInfo:
        return std_inspect.underlying_strategy_return_type(self.strategy)

"""
Standard Nodes.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Never, cast

from delphyne.core.trees import Choice, Navigation, Node, Strategy, subchoice
from delphyne.stdlib.dsl import GeneratorConvertible, convert_to_generator
from delphyne.stdlib.embedded import EmbeddedSubtree
from delphyne.stdlib.generators import Budget, Generator
from delphyne.stdlib.search_envs import Params


#####
##### Failure
#####


@dataclass(frozen=True)
class Failure(Node):
    message: str

    def valid_action(self, action: object) -> bool:
        return False

    def leaf_node(self) -> bool:
        return True

    def navigate(self) -> Navigation:
        assert False
        yield

    def summary_message(self) -> str:
        return self.message


def fail(msg: str) -> Strategy[Failure, Never]:
    yield Failure(msg)
    assert False


def ensure(prop: bool, msg: str = "") -> Strategy[Failure, None]:
    if not prop:
        yield Failure(msg)


#####
##### Subs
#####


@dataclass(frozen=True)
class Subs(Node):
    _subs: Sequence[EmbeddedSubtree[Any, Any]]

    __match_args__ = ()

    def __len__(self) -> int:
        return len(self._subs)

    @subchoice
    def sub(self, i: int) -> EmbeddedSubtree[Any, Any]:
        return self._subs[i]

    def valid_action(self, action: object) -> bool:
        n = len(self._subs)
        return isinstance(action, tuple) and len(cast(Any, action)) == n

    def base_choices(self) -> Iterable[Choice[object]]:
        return tuple(self.sub(i) for i in range(len(self)))


#####
##### Branch node
#####


@dataclass(frozen=True)
class Branch[P](Node):
    _gen: Generator[P, Any]
    max_branching: Callable[[P], int | None]
    max_gen_budget: Callable[[P], Budget]
    max_cont_budget: Callable[[P], Budget]

    __match_args__ = ()

    @property
    @subchoice
    def gen(self) -> Generator[P, Any]:
        return self._gen

    def navigate(self) -> Navigation:
        return (yield self.gen)

    def primary_choice(self):
        return self.gen


@dataclass(frozen=True)
class Run[P](Branch[P]):
    pass


def branch[P: Params, T](
    gen: GeneratorConvertible[P, T],
    max_branching: Callable[[P], int | None] | None = None,
    *,
    max_gen_budget: Callable[[P], Budget] | None = None,
    max_cont_budget: Callable[[P], Budget] | None = None,
    param_type: type[P] | None = None
) -> Strategy[Branch[P], T]:  # fmt: skip
    # The param_type argument is only useful to help pyright infer the
    # expected type of `max_gen_budget` et al. when a lambda is
    # provided.
    if max_branching is None:
        max_branching = lambda _: None
    if max_gen_budget is None:
        max_gen_budget = lambda _: Budget.limit()
    if max_cont_budget is None:
        max_cont_budget = lambda _: Budget.limit()
    ret = yield Branch(
        convert_to_generator(gen),
        max_branching,
        max_gen_budget,
        max_cont_budget,
    )
    return cast(T, ret)


def run[P: Params, T](
    gen: GeneratorConvertible[P, T],
    max_budget: Callable[[P], Budget] | None = None,
    *,
    param_type: type[P] | None = None
) -> Strategy[Run[P], T]:  # fmt: skip
    """
    Run a generator to completion. This creates a degenerate branch node
    with a max branching parameter of 1.
    """
    if max_budget is None:
        max_budget = lambda _: Budget.limit()
    ret = yield Run(
        convert_to_generator(gen),
        max_branching=lambda _: 1,
        max_gen_budget=max_budget,
        max_cont_budget=lambda _: Budget.limit(),
    )
    return cast(T, ret)


#####
##### Debug
#####


@dataclass(frozen=True)
class DebugLog(Node):
    message: str

    def valid_action(self, action: object) -> bool:
        return action is None

    def summary_message(self) -> str:
        return self.message


#####
##### Basic branching strategies
#####


type BranchingStrategyNode[P] = Branch[P] | Failure


type Branching[P, T] = Strategy[BranchingStrategyNode[P], T]

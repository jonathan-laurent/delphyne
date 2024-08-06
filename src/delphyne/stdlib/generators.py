"""
Generators: interface and implementations.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from delphyne.core import inspect as dpy_inspect
from delphyne.core import refs, trees
from delphyne.core.parse import ParseError
from delphyne.core.queries import AnyQuery, Prompt, Query
from delphyne.core.refs import AnswerId, ChoiceRef
from delphyne.core.tracing import Outcome, QueryOrigin
from delphyne.core.trees import ChoiceSource, Node, StrategyComp, Tree
from delphyne.utils.misc import Cell
from delphyne.utils.typing import NoTypeInfo, TypeAnnot


#####
##### Representing budgets
#####


@dataclass(frozen=True)
class Budget:
    num_requests: float
    num_context_tokens: float
    num_generated_tokens: float
    price: float

    @staticmethod
    def limit(
        num_requests: float | None = None,
        num_context_tokens: float | None = None,
        num_generated_tokens: float | None = None,
        price: float | None = None,
    ) -> "Budget":
        if num_requests is None:
            num_requests = np.inf
        if num_context_tokens is None:
            num_context_tokens = np.inf
        if num_generated_tokens is None:
            num_generated_tokens = np.inf
        if price is None:
            price = np.inf
        return Budget(
            num_requests, num_context_tokens, num_generated_tokens, price
        )

    @staticmethod
    def spent(
        num_requests: float | None = None,
        num_context_tokens: float | None = None,
        num_generated_tokens: float | None = None,
        price: float | None = None,
    ) -> "Budget":
        if num_requests is None:
            num_requests = 0
        if num_context_tokens is None:
            num_context_tokens = 0
        if num_generated_tokens is None:
            num_generated_tokens = 0
        if price is None:
            price = 0
        return Budget(
            num_requests, num_context_tokens, num_generated_tokens, price
        )

    @staticmethod
    def of_vec(vec: NDArray[np.float64]):
        return Budget(*vec)

    def vec(self) -> NDArray[np.float64]:
        return np.array(list(self.__dict__.values()))

    def __add__(self, other: "Budget"):
        return Budget.of_vec(self.vec() + other.vec())

    def __sub__(self, other: "Budget"):
        return Budget.of_vec(self.vec() - other.vec())

    def nonnegative(self) -> bool:
        return bool(np.all(self.vec() >= 0))

    def __ge__(self, other: "Budget"):
        return bool(np.all(self.vec() >= other.vec()))

    def __le__(self, other: "Budget"):
        return bool(np.all(self.vec() <= other.vec()))

    @staticmethod
    def zero():
        return Budget(0, 0, 0, 0)


#####
##### Abstract Generator Interface
#####


@dataclass(frozen=False)
class BudgetCounter:
    """
    These counters are mutated in place when budget is spent.

    Note: we cannot use a single `remaining` field because the limit can
    be +oo, in which case we would not be able to track spending.
    """

    limit: Budget
    spent: Budget = field(default_factory=Budget.zero)

    def can_spend(self, budget: Budget) -> bool:
        return self.spent + budget <= self.limit


type GenMode = Literal["lazy", "eager"]


@dataclass
class GenEnv:
    """
    The counters and mode variables belong to the generator who created
    them. However, counters can still be mutated by descendants.
    """

    counters: Sequence[BudgetCounter]
    mode: GenMode

    def budget_left(self) -> bool:
        return all(c.can_spend(Budget.zero()) for c in self.counters)

    def with_counter(self, counter: BudgetCounter) -> "GenEnv":
        return GenEnv([*self.counters, counter], self.mode)

    def can_spend(self, budget: Budget) -> bool:
        return all(c.can_spend(budget) for c in self.counters)

    def declare_spent(self, budget: Budget) -> None:
        for c in self.counters:
            c.spent += budget


@dataclass(frozen=True)
class GenResponse[T]:
    items: Sequence[T]
    min_required_next: Budget | None = None


type GenRet[T] = AsyncIterator[GenResponse[Outcome[T]]]


type ArbitraryTree = Tree[Node, object]


class Generator[P, T](trees.Choice[T], ABC):
    @abstractmethod
    def __call__(
        self, env: GenEnv, parent: ArbitraryTree, params: P
    ) -> GenRet[T]:
        pass


#####
##### Standard generators
#####


class SearchPolicy[P, N: Node](Protocol):
    # fmt: off
    def __call__[T](
        self, env: GenEnv, tree: Tree[N, T], params: P
    ) -> GenRet[T]:
        ...
    # fmt: on


@dataclass(frozen=True)
class ExecuteStrategy[P, N: Node, T](Generator[P, T]):
    strategy: StrategyComp[N, T]
    search_policy: SearchPolicy[P, N]
    origin: Cell[ChoiceRef | None] = field(default_factory=lambda: Cell(None))

    def __call__(
        self, env: GenEnv, parent: ArbitraryTree, params: P
    ) -> GenRet[T]:
        assert self.origin.content is not None
        sub_tree = parent.spawn(self.strategy, self.origin.content)
        return self.search_policy(env, sub_tree, params)

    def label(self) -> str | None:
        return dpy_inspect.underlying_strategy_name(self.strategy)

    def source(self) -> trees.StrategyInstance[N, T]:
        return trees.StrategyInstance(self.strategy)

    def return_type(self) -> TypeAnnot[T] | NoTypeInfo:
        return dpy_inspect.underlying_strategy_return_type(self.strategy)

    def set_origin(self, origin: refs.ChoiceRef) -> None:
        self.origin.content = origin

    def get_origin(self) -> refs.ChoiceRef:
        assert self.origin.content is not None
        return self.origin.content


type ModelExecutor = Callable[
    [AnyQuery, Prompt, int], Awaitable[tuple[Sequence[str | None], Budget]]
]
"""
The first query argument is used for mocking.
"""


type ExampleProvider = Callable[[AnyQuery], Sequence[tuple[AnyQuery, str]]]


type CostEstimator = Callable[[Prompt], Budget]


type PromptHook = Callable[[Prompt, AnswerId], None]


@dataclass(frozen=True)
class ExecuteQuery[P, T](Generator[P, T]):
    query: Query[P, T]
    estimator: Callable[[P], CostEstimator]
    executor: Callable[[P], ModelExecutor]
    example_provider: Callable[[P], ExampleProvider]
    prompt_hook: Callable[[P], PromptHook]
    origin: Cell[ChoiceRef | None] = field(default_factory=lambda: Cell(None))

    def source(self) -> Query[P, T]:
        return self.query

    def label(self) -> str | None:
        return self.query.name()

    async def __call__(
        self, env: GenEnv, parent: ArbitraryTree, params: P
    ) -> GenRet[T]:
        examples = self.example_provider(params)(self.query)
        prompt = self.query.create_prompt(params, examples)
        cost_estimate = self.estimator(params)(prompt)
        executor = self.executor(params)
        origin = QueryOrigin(parent.node_id, self.get_origin())
        parent.tracer.declare_query(origin)
        while True:
            # TODO: we do not check the mode and are lazy here...
            if not env.can_spend(cost_estimate):
                yield GenResponse([], cost_estimate)
                continue
            answers, budget = await executor(self.query, prompt, 1)
            env.declare_spent(budget)
            outcomes: list[Outcome[T]] = []
            for answer in answers:
                if answer is None:
                    continue
                parsed_answer = self.query.parse_answer(answer)
                if isinstance(parsed_answer, ParseError):
                    continue  # We do not want to add syntax errors in the trace.
                parsed_answer = cast(T, parsed_answer)
                aid = parent.tracer.fresh_or_cached_answer_id(answer, origin)
                self.prompt_hook(params)(prompt, aid)
                ref = refs.ChoiceOutcomeRef(self.get_origin(), aid)
                outcomes.append(
                    Outcome(parsed_answer, ref, self.return_type())
                )
            yield GenResponse(outcomes)

    def set_origin(self, origin: refs.ChoiceRef) -> None:
        self.origin.content = origin

    def get_origin(self) -> refs.ChoiceRef:
        assert self.origin.content is not None
        return self.origin.content

    def return_type(self) -> TypeAnnot[T] | NoTypeInfo:
        return self.query.return_type()


@dataclass
class GeneratorAdaptor[P, Q, T](Generator[P, T]):
    generator: Generator[Q, T]
    adapt: Callable[[P], Q]

    def __call__(
        self, env: GenEnv, parent: ArbitraryTree, params: P
    ) -> GenRet[T]:
        return self.generator(env, parent, self.adapt(params))

    def label(self) -> str | None:
        return self.generator.label()

    def source(self) -> ChoiceSource[T]:
        return self.generator.source()

    def return_type(self) -> TypeAnnot[T] | NoTypeInfo:
        return self.generator.return_type()

    def set_origin(self, origin: ChoiceRef) -> None:
        self.generator.set_origin(origin)

    def get_origin(self) -> ChoiceRef:
        return self.generator.get_origin()

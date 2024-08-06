"""
Strategy decorator.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, overload

from delphyne.core.inspect import FunctionWrapper
from delphyne.core.queries import Query
from delphyne.core.refs import Assembly
from delphyne.core.tracing import Outcome, drop_refs
from delphyne.core.trees import Node, Strategy, StrategyComp
from delphyne.stdlib.generators import (
    ExecuteQuery,
    ExecuteStrategy,
    Generator,
    GeneratorAdaptor,
    SearchPolicy,
)
from delphyne.stdlib.search_envs import HasSearchEnv


#####
##### @strategy decorator
#####


@dataclass
class WrappedStrategy[P, N: Node, T](FunctionWrapper[[], Strategy[N, T]]):
    strategy: StrategyComp[N, T]
    search_policy: SearchPolicy[P, N] | None

    def __post_init__(self):
        functools.update_wrapper(self, self.strategy)

    def __call__(self) -> Strategy[N, T]:
        return self.strategy()

    def wrapped(self):
        return self.strategy

    def __getitem__[Q](
        self, policy: SearchPolicy[Q, N], /
    ) -> "WrappedStrategy[Q, N, T]":  # fmt: skip
        return WrappedStrategy[Q, N, T](self.strategy, policy)

    @property
    def generator(self) -> ExecuteStrategy[P, N, T]:
        assert self.search_policy is not None
        return ExecuteStrategy(self.strategy, self.search_policy)

    def using[Q](self, adapter: Callable[[Q], P]) -> Generator[Q, T]:
        return GeneratorAdaptor(self.generator, adapter)


@dataclass
class WrappedParametricStrategy[**A, P, N: Node, T](
    FunctionWrapper[A, WrappedStrategy[P, N, T]]
):
    strategy: Callable[A, Strategy[N, T]]
    search_policy: SearchPolicy[P, N] | None

    def __post_init__(self):
        functools.update_wrapper(self, self.strategy)

    def __call__(
        self, *args: A.args, **kwargs: A.kwargs
    ) -> WrappedStrategy[P, N, T]:
        wrapped = functools.partial(self.strategy, *args, **kwargs)
        return WrappedStrategy(wrapped, self.search_policy)

    def wrapped(self):
        return self.strategy

    def __getitem__[Q](
        self, policy: SearchPolicy[Q, N], /
    ) -> "WrappedParametricStrategy[A, Q, N, T]":  # fmt: skip
        return WrappedParametricStrategy(self.strategy, policy)


type _StrategyDecorator[**A, P, N: Node, T] = Callable[
    [Callable[A, Strategy[N, T]]],
    WrappedParametricStrategy[A, P, N, T],
]


# fmt: off

@overload
def strategy[**A, N: Node, T]() -> _StrategyDecorator[A, Any, N, T]:
    ...

@overload
def strategy[**A, P, N: Node, T](
    search_policy: SearchPolicy[P, N]
) -> _StrategyDecorator[A, P, N, T]:
    ...

def strategy[**A, P, N: Node, T](
    search_policy: SearchPolicy[P, N] | None = None
) -> _StrategyDecorator[A, P, N, T]:  # fmt: skip
    return lambda f: WrappedParametricStrategy(f, search_policy)

# fmt: on


#####
##### Automated conversions
#####


type GeneratorConvertible[P, T] = (
    Generator[P, T] | WrappedStrategy[P, Any, T] | Query[P, T]
)


def convert_to_generator[P: HasSearchEnv, T](
    arg: GeneratorConvertible[P, T], /
) -> Generator[P, T]:  # fmt: skip
    match arg:
        case WrappedStrategy():
            return arg.generator
        case Query():
            return ExecuteQuery[P, T](
                arg,
                lambda p: p.env.estimate_cost,
                lambda p: p.env.execute_prompt,
                lambda p: p.env.collect_examples,
                lambda p: p.env.prompt_hook,
            )
        case _:
            return arg


def convert_to_parametric_generator[P: HasSearchEnv, T](
    f: Callable[..., GeneratorConvertible[P, T]], /
) -> Callable[[Assembly[Outcome[Any]]], Generator[P, T]]:  # fmt: skip

    def parametric_gen(*args: Assembly[Outcome[Any]]) -> Generator[P, T]:
        args_new = [drop_refs(arg) for arg in args]
        return convert_to_generator(f(*args_new))

    return parametric_gen

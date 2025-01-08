"""
Defining the `@strategy` decorator.
"""

# pyright: basic

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, overload

import delphyne.core as dp


@dataclass(frozen=True)
class StrategyInstance[N: dp.Node, P, T](dp.StrategyComp[N, P, T]):
    def using[P2](
        self,
        get_policy: Callable[[P2], tuple[dp.AbstractSearchPolicy[N], P]],
        inner_policy_type: type[P2] | None = None,
    ) -> dp.Builder[dp.OpaqueSpace[P2, T]]:
        return dp.OpaqueSpace[P2, T].from_strategy(self, get_policy)

    # Pyright seems to be treating __getitem__ differently and does
    # worse inference than for using. Same for operators like &, @...

    # def __getitem__[P2](
    #     self, get_policy: Callable[[P2], tuple[SearchPolicy[N], P]]
    # ) -> Builder[OpaqueSpace[P2, T]]:
    #     return self.using(get_policy)


def search[N: dp.Node, P, P2, T](
    strategy: StrategyInstance[N, P, T],
    get_policy: Callable[[P2], tuple[dp.AbstractSearchPolicy[N], P]],
    inner_policy_type: type[P2] | None = None,
) -> dp.Builder[dp.OpaqueSpace[P2, T]]:
    return strategy.using(get_policy, inner_policy_type)


class _StrategyDecorator(Protocol):
    def __call__[**A, N: dp.Node, P, T](
        self,
        f: Callable[A, dp.Strategy[N, P, T]],
    ) -> Callable[A, StrategyInstance[N, P, T]]: ...


@overload
def strategy[**A, N: dp.Node, P, T](
    f: Callable[A, dp.Strategy[N, P, T]],
) -> Callable[A, StrategyInstance[N, P, T]]: ...


@overload
def strategy(*, name: str | None) -> _StrategyDecorator: ...


def strategy(*args: Any, **kwargs: Any) -> Any:
    if not kwargs and len(args) == 1 and callable(args[0]):
        f: Any = args[0]
        return lambda *args, **kwargs: StrategyInstance(f, args, kwargs)
    elif not args:
        return lambda f: lambda *args, **kwargs: StrategyInstance(
            f, args, kwargs, name=kwargs.get("name")
        )
    assert False, "Wrong use of @strategy."


# def strategy[**A, N: dp.Node, P, T](
#     f: Callable[A, dp.Strategy[N, P, T]],
# ) -> Callable[A, StrategyInstance[N, P, T]]:
#     def wrapped(
#         *args: A.args, **kwargs: A.kwargs
#     ) -> StrategyInstance[N, P, T]:
#         return StrategyInstance(f, args, kwargs)

#     return wrapped

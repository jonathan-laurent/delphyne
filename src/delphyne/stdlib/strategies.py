"""
Defining the `@strategy` decorator.
"""

from collections.abc import Callable
from dataclasses import dataclass

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


def strategy[**A, N: dp.Node, P, T](
    f: Callable[A, dp.Strategy[N, P, T]],
) -> Callable[A, StrategyInstance[N, P, T]]:
    def wrapped(
        *args: A.args, **kwargs: A.kwargs
    ) -> StrategyInstance[N, P, T]:
        return StrategyInstance(f, args, kwargs)

    return wrapped

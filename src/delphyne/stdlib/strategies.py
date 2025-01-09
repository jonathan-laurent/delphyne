"""
Defining the `@strategy` decorator.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, overload

import delphyne.core as dp


@dataclass(frozen=True)
class StrategyInstance[N: dp.Node, P, T](dp.StrategyComp[N, P, T]):
    def using[P2](
        self,
        get_policy: Callable[[P2], dp.Policy[N, P]],
        inner_policy_type: type[P2] | None = None,
    ) -> dp.Builder[dp.OpaqueSpace[P2, T]]:
        return dp.OpaqueSpace[P2, T].from_strategy(self, get_policy)

    # Pyright seems to be treating __getitem__ differently and does
    # worse inference than for using. Same for operators like &, @...

    def __call__[P2](
        self,
        inner_policy_type: type[P2],
        get_policy: Callable[[P2], dp.Policy[N, P]],
    ) -> dp.Builder[dp.OpaqueSpace[P2, T]]:
        return self.using(get_policy)

    def run_toplevel(
        self,
        env: dp.PolicyEnv,
        policy: dp.Policy[N, P],
        monitor: dp.TreeMonitor = dp.TreeMonitor(),
    ) -> dp.Stream[T]:
        tree = dp.reify(self, monitor)
        return policy[0](tree, env, policy[1])


def search[N: dp.Node, P, P2, T](
    strategy: StrategyInstance[N, P, T],
    get_policy: Callable[[P2], dp.Policy[N, P]],
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


def strategy(*dec_args: Any, **dec_kwargs: Any) -> Any:
    # Using functools.wraps is important so that the object loader can
    # get the type hints to properly instantiate arguments.
    if not dec_kwargs and len(dec_args) == 1 and callable(dec_args[0]):
        f: Any = dec_args[0]
        return functools.wraps(f)(
            lambda *args, **kwargs: StrategyInstance(f, args, kwargs)
        )
    elif not dec_args:
        return lambda f: functools.wraps(f)(  # type: ignore
            lambda *args, **kwargs: StrategyInstance(  # type: ignore
                f,  # type: ignore
                args,
                kwargs,
                name=dec_kwargs.get("name"),  # type: ignore
            )
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

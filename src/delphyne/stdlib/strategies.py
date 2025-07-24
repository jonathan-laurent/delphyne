"""
Standard Wrappers for Strategy Computations and Functions.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, overload

import delphyne.core as dp


@dataclass(frozen=True)
class StrategyInstance[N: dp.Node, P, T](dp.StrategyComp[N, P, T]):
    """
    A subclass of [`StrategyComp`][delphyne.core.StrategyComp] that adds
    convenience features such as the `using` method to build opaque
    spaces.
    """

    def using[P2](
        self,
        get_policy: Callable[[P2], dp.Policy[N, P]],
        inner_policy_type: type[P2] | None = None,
    ) -> dp.Opaque[P2, T]:
        """
        Turn a strategy instance into an opaque space by providing a
        mapping from the ambient inner policy to an appropriate policy.

        The optional `inner_policy_type` argument is ignored at runtime
        and can be used to help type checkers infer the type of the
        ambient inner policy.

        ??? info "Design Note"
            Using operators such as `&` or `@` instead of using does not
            work well since some type checkers (e.g., Pyright) perform
            worse inference when using those instead of a standard
            method.
        """
        return dp.OpaqueSpace[P2, T].from_strategy(self, get_policy)

    def run_toplevel(
        self,
        env: dp.PolicyEnv,
        policy: dp.Policy[N, P],
        monitor: dp.TreeMonitor = dp.TreeMonitor(),
    ) -> dp.Stream[T]:
        """
        Utility method to reify a strategy into a tree and run it using
        a given policy.
        """
        tree = dp.reify(self, monitor)
        return policy[0](tree, env, policy[1])


class _StrategyDecorator(Protocol):
    """
    Type of the `@strategy(...)` decorator.
    """

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
    """
    Standard decorator for wrapping strategy functions into functions
    returning [`StrategyInstance`][delphyne.stdlib.StrategyInstance].
    """
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

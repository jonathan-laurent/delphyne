"""
Standard Wrappers for Strategy Computations and Functions.
"""

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import EllipsisType
from typing import Any, Protocol, cast, overload

import delphyne.core as dp
from delphyne.core import inspect
from delphyne.stdlib.policies import IPDict, dict_subpolicy
from delphyne.utils.typing import NoTypeInfo, TypeAnnot


@dataclass(frozen=True)
class StrategyInstance[N: dp.Node, P, T](dp.StrategyComp[N, P, T]):
    """
    A subclass of [`StrategyComp`][delphyne.core.StrategyComp] that adds
    convenience features such as the `using` method to build opaque
    spaces.
    """

    @overload
    def using(self, get_policy: EllipsisType, /) -> dp.Opaque[IPDict, T]: ...

    @overload
    def using[P2](
        self,
        get_policy: Callable[[P2], dp.Policy[N, P]] | EllipsisType,
        /,
        inner_policy_type: type[P2] | None = None,
    ) -> dp.Opaque[P2, T]: ...

    def using[P2](
        self,
        get_policy: Callable[[P2], dp.Policy[N, P]] | EllipsisType,
        /,
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
        if isinstance(get_policy, EllipsisType):
            return dp.OpaqueSpace[P2, T].from_strategy(
                self, cast(Any, dict_subpolicy)
            )
        return dp.OpaqueSpace[P2, T].from_strategy(
            self, lambda p, _: get_policy(p)
        )

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

    Note that this definition cannot be inlined in the return type of
    `@strategy`, without which variables such as `P` and `T` would not
    be correctly scoped and inferred (type checkers such as Pyright
    would set them to `Unknown` if they cannot be inferred from `args`
    in `@strategy(*args)(f)`).
    """

    def __call__[**A, N: dp.Node, P, T](
        self,
        f: Callable[A, dp.Strategy[N, P, T]],
    ) -> Callable[A, StrategyInstance[N, P, T]]: ...


@overload
def strategy[**A, N: dp.Node, P, T](
    f: Callable[A, dp.Strategy[N, P, T]], /
) -> Callable[A, StrategyInstance[N, P, T]]: ...


@overload
def strategy(
    *,
    name: str | None = None,
    ret: TypeAnnot[Any] | NoTypeInfo = NoTypeInfo(),
    inherit_tags: Callable[..., Sequence[dp.SpaceBuilder[Any]]] | None = None,
) -> _StrategyDecorator: ...


def strategy(*dec_args: Any, **dec_kwargs: Any) -> Any:
    """
    Standard parametric decorator for wrapping strategy functions into
    functions returning
    [`StrategyInstance`][delphyne.stdlib.StrategyInstance].

    Parameters:
        name (optional): Name of the strategy. If not provided, the
            __name__ attribute of the strategy function is used instead.
        ret (optional): Return type of the strategy function. If not
            provided, it is obtained by inspecting type annotations.
        inherit_tags (optional): A function that maps all arguments from
            the decorated strategy function to a sequence of space
            builders from which tags must be inherited.

    ??? info
        `strategy()(f)` can be shortened as `@strategy(f)`, at the cost
        of a pretty complex type signature for `strategy`.
    """

    # Using functools.wraps is important so that the object loader can
    # get the type hints to properly instantiate arguments.
    error_msg = "Invalid use of the @strategy decorator."
    # If no positional argument is provided, we have a call of the form
    # `@strategy(...)(f)`. Otherwise, we have a call of the form
    # `@strategy(f)`.
    assert len(dec_args) in [0, 1], error_msg

    # @strategy(f) case
    if not dec_kwargs and len(dec_args) == 1 and callable(dec_args[0]):
        f: Any = dec_args[0]
        name = inspect.function_name(f)
        tags = (name,) if name else ()

        def wrapped(*args: Any, **kwargs: Any):
            return StrategyInstance(
                f,
                args,
                kwargs,
                _name=None,
                _return_type=NoTypeInfo(),
                _tags=tags,
            )

        return functools.wraps(f)(wrapped)

    # @strategy(...)(f) case
    elif not dec_args:

        def decorator(f: Any):
            def wrapped(*args: Any, **kwargs: Any):
                name = dec_kwargs.get("name")
                if name is None:
                    name = inspect.function_name(f)
                tags = (name,) if name else ()
                # Inherit tags from space arguments if needed.
                inherited_fun = dec_kwargs.get("inherit_tags", None)
                if inherited_fun is not None:
                    inherited = inherited_fun(*args, **kwargs)
                    for space in inherited:
                        assert isinstance(space, dp.SpaceBuilder)
                        tags = (*tags, *space.tags)
                return StrategyInstance(
                    f,
                    args,
                    kwargs,
                    _name=name,
                    _return_type=dec_kwargs.get("ret", NoTypeInfo()),
                    _tags=tags,
                )

            return functools.wraps(f)(wrapped)

        return decorator

    assert False, error_msg

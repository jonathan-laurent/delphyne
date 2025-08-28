"""
Defining the `@strategy` decorator.
"""

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import EllipsisType
from typing import Any, Protocol, cast, overload

import delphyne.core as dp
import delphyne.stdlib.policies as pol
from delphyne.core import inspect
from delphyne.stdlib.environments import PolicyEnv
from delphyne.stdlib.opaque import Opaque, OpaqueSpace
from delphyne.stdlib.policies import IPDict, Policy, Stream
from delphyne.utils.typing import NoTypeInfo, TypeAnnot


@dataclass(frozen=True)
class StrategyInstance[N: dp.Node, P, T](dp.StrategyComp[N, P, T]):
    """
    A *strategy computation* that can be reified into a search tree,
    obtained by instantiating a strategy function.

    `StrategyInstance` is a subclass of `StrategyComp` that adds
    convenience features such as the `using` method for building opaque
    spaces. The `strategy` decorator can be used to wrap strategy
    functions so as to have them return `StrategyInstance` objects.

    Type Parameters:
        N: Signature of the strategy.
        P: Inner policy type associated with the strategy.
        T: Return type of the strategy.
    """

    @overload
    def using(self, get_policy: EllipsisType, /) -> Opaque[IPDict, T]: ...

    @overload
    def using[Pout](
        self,
        get_policy: Callable[[Pout], Policy[N, P]] | EllipsisType,
        /,
        inner_policy_type: type[Pout] | None = None,
    ) -> Opaque[Pout, T]: ...

    def using[Pout](
        self,
        get_policy: Callable[[Pout], Policy[N, P]] | EllipsisType,
        /,
        inner_policy_type: type[Pout] | None = None,
    ) -> Opaque[Pout, T]:
        """
        Turn a strategy instance into an opaque space by providing a
        mapping from the ambient inner policy to an appropriate policy.

        Attributes:
            get_policy: A function that maps the ambient inner policy to
                a policy (i.e., a pair of a search policy and of an
                inner policy) to use for answering the query.
                Alternatively, if the ellipsis value `...` is passed,
                the inner policy type is assumed to be `IPDict`, and
                sub-policies are automatically selected using tags (see
                `IPDict` documentation).
            inner_policy_type: Ambient inner policy type for the outer
                strategy from which the strategy is called. This
                information is not used at runtime but it can be
                provided to help type inference when necessary.

        Type Parameters:
            Pout: Ambient inner policy type associated with the outer
                strategy from which the strategy is called.
        """

        # Using operators such as `&` or `@` instead of using does not
        # work well since some type checkers (e.g., Pyright) perform
        # worse inference when using those instead of a standard method.
        if isinstance(get_policy, EllipsisType):
            return OpaqueSpace[Pout, T].from_strategy(
                self, cast(Any, pol.dict_subpolicy)
            )
        return OpaqueSpace[Pout, T].from_strategy(
            self, lambda p, _: get_policy(p)
        )

    def run_toplevel(
        self,
        env: PolicyEnv,
        policy: Policy[N, P],
        monitor: dp.TreeMonitor = dp.TreeMonitor(),
    ) -> Stream[T]:
        """
        Reify a strategy into a tree and run it using a given policy.
        """
        tree = dp.reify(self, monitor)
        return policy.search(tree, env, policy.inner)


class _StrategyDecorator(Protocol):
    """
    Type of the `strategy` decorator, after is optional arguments are
    instantiated.
    """

    # Note that this definition cannot be inlined in the return type of
    # `@strategy`, without which variables such as `P` and `T` would not
    # be correctly scoped and inferred (type checkers such as Pyright
    # would set them to `Unknown` if they cannot be inferred from `args`
    # in `@strategy(*args)(f)`).

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


def strategy(
    f: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    ret: TypeAnnot[Any] | NoTypeInfo = NoTypeInfo(),
    inherit_tags: Callable[..., Sequence[dp.SpaceBuilder[Any]]] | None = None,
) -> Any:
    """
    Standard parametric decorator for wrapping strategy functions into
    functions returning `StrategyInstance` objects.

    Parameters:
        name (optional): Name of the strategy. If not provided, the
            __name__ attribute of the strategy function is used instead.
            The name of the strategy is used in defining default tags
            and when visualizing traces.
        ret (optional): Return type of the strategy function. If not
            provided, it is obtained by inspecting type annotations.
            This information is used when visualizing traces and for
            serializing top-level success values when running commands.
        inherit_tags (optional): A function that maps all arguments from
            the decorated strategy function to a sequence of space
            builders from which tags must be inherited. By default,
            nothing is inherited.

    !!! info
        `strategy()(f)` can be shortened as `@strategy(f)`, hence the
        overloading of the type of `strategy`.

    ??? info "Runtime use of type annotations"
        The type annotations for the arguments and return type of a
        strategy function are leveraged at runtime in two ways:

        - To improve the rendering of values when visualizing traces
          (e.g., using YAML serialization instead of `pprint`).
        - To unserialize arguments for the top-level strategy when
          specified in JSON or YAML and serialize its return value.

        In summary, type annotations are fully optional, except when
        trying to unserialize (resp. serialize) the arguments (resp.
        return type) of a top-level strategy involving custom data types
        (and not just JSON values such as integers, strings,
        dictionaries...).
    """

    # Using functools.wraps is important so that the object loader can
    # get the type hints to properly instantiate arguments.

    # @strategy(f) case
    if f is not None:
        assert name is None
        assert isinstance(ret, NoTypeInfo)
        assert inherit_tags is None
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
    else:

        def decorator(f: Any):
            def wrapped(*args: Any, **kwargs: Any):
                nonlocal name
                if name is None:
                    name = inspect.function_name(f)
                tags = (name,) if name else ()
                # Inherit tags from space arguments if needed.
                if inherit_tags is not None:
                    inherited = inherit_tags(*args, **kwargs)
                    for space in inherited:
                        assert isinstance(space, dp.SpaceBuilder)
                        tags = (*tags, *space.tags)
                return StrategyInstance(
                    f,
                    args,
                    kwargs,
                    _name=name,
                    _return_type=ret,
                    _tags=tags,
                )

            return functools.wraps(f)(wrapped)

        return decorator

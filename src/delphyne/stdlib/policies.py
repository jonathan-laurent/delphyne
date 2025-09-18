"""
Standard policy types and wrappers
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, NoReturn, Protocol, TypeVar

import delphyne.core as dp
from delphyne.core import Node
from delphyne.stdlib.environments import PolicyEnv
from delphyne.stdlib.streams import Stream, StreamTransformer

#####
##### Policies
#####


N = TypeVar("N", bound=Node, contravariant=True)
P = TypeVar("P", covariant=True)


@dataclass(frozen=True)
class Policy(Generic[N, P], dp.AbstractPolicy[PolicyEnv, N, P]):
    """
    A pair of a search policy and of an inner policy.

    More precisely, a policy for trees with effects `N` (contravariant)
    gathers a search policy handling `N` along with an inner policy
    object of type `P` (covariant).

    Values of this type can be built concisely using the `&` operator
    defined on type `SearchPolicy`.
    """

    _search: "SearchPolicy[N]"
    _inner: P

    @property
    def search(self) -> "SearchPolicy[N]":
        return self._search

    @property
    def inner(self) -> P:
        return self._inner


#####
##### Search Policies
#####


@dataclass(frozen=True)
class SearchPolicy[N: Node](dp.AbstractSearchPolicy[PolicyEnv, N]):
    """
    A search policy takes as arguments a tree with a given signature
    (covariant type parameter `N`), a global policy environment, and an
    inner policy with appropriate type, and returns a search stream.

    `SearchPolicy` is a subclass of `AbstractSearchPolicy`, which
    provides convenience features such as support for the `@`
    composition operator (for composing search policies with stream
    transformers and tree transformers) and the `&` operator for pairing
    a search policy with an inner policy.

    Search policies can be conveniently defined using the
    `search_policy` decorator. See `dfs` for an example.
    """

    _fn: "_SearchPolicyFn[N]"

    def __call__[P, T](
        self,
        tree: "dp.Tree[N, P, T]",
        env: PolicyEnv,
        policy: P,
    ) -> Stream[T]:
        return Stream(lambda: self._fn(tree, env, policy))

    def __and__[P](self, other: P) -> "Policy[N, P]":
        """
        Pair a search policy with an inner policy to form a policy.
        """
        return Policy(self, other)

    def __rmatmul__(self, other: StreamTransformer) -> "SearchPolicy[N]":
        """
        Compose a search policy with a stream transformer.
        """
        if not isinstance(other, StreamTransformer):  # pyright: ignore[reportUnnecessaryIsInstance]
            return NotImplemented
        return self._compose_with_stream_transformer(other)

    def _compose_with_stream_transformer(
        self,
        trans: StreamTransformer,
    ) -> "SearchPolicy[N]":
        def policy[P, T](
            tree: dp.Tree[N, P, T], env: PolicyEnv, policy: P
        ) -> dp.StreamGen[T]:
            return trans(self(tree, env, policy), env).gen()

        return SearchPolicy(policy)


class _SearchPolicyFn[N: Node](Protocol):
    def __call__[P, T](
        self,
        tree: dp.Tree[N, P, T],
        env: PolicyEnv,
        policy: P,
    ) -> dp.StreamGen[T]: ...


class _ParametricSearchPolicyFn[N: Node, **A](Protocol):
    def __call__[P, T](
        self,
        tree: dp.Tree[N, P, T],
        env: PolicyEnv,
        policy: P,
        *args: A.args,
        **kwargs: A.kwargs,
    ) -> dp.StreamGen[T]: ...


class _ParametricSearchPolicy[N: Node, **A](Protocol):
    def __call__(
        self, *args: A.args, **kwargs: A.kwargs
    ) -> SearchPolicy[N]: ...


def search_policy[N: Node, **A](
    fn: _ParametricSearchPolicyFn[N, A],
) -> _ParametricSearchPolicy[N, A]:
    """
    Convenience decorator for creating parametric search policies (i.e.,
    functions that return search policies).

    See `dfs` for an example.

    Attributes:
        fn: A function that takes a tree, a policy environment, an inner
            policy, and additional parameters as arguments and returns a
            search stream generator (`SearchStreamGen`).

    Returns:
        A function that takes the additional parameters of `fn` as
        arguments and returns a search policy (`SearchPolicy`).
    """

    def parametric(*args: A.args, **kwargs: A.kwargs) -> SearchPolicy[N]:
        def policy[T](
            tree: dp.Tree[N, Any, T], env: PolicyEnv, policy: Any
        ) -> dp.StreamGen[T]:
            return fn(tree, env, policy, *args, **kwargs)

        return SearchPolicy(policy)

    return parametric


#####
##### Prompting Policies
#####


@dataclass(frozen=True)
class PromptingPolicy(dp.AbstractPromptingPolicy[PolicyEnv]):
    """
    A prompting policy takes as arguments a query (attached to a
    specific node) and a global policy environment, and returns a search
    stream (`SearchStream`).

    `PromptingPolicy` is a subclass of `AbstractPromptingPolicy`, which
    provides convenience features such as support for the `@`
    composition operator (for composing prompting policies with stream
    transformers).

    Prompting policies can be conveniently defined using the
    `prompting_policy` decorator. See the definition of `few_shot` for
    an example.
    """

    _fn: "_PromptingPolicyFn"

    def __call__[T](
        self, query: dp.AttachedQuery[T], env: PolicyEnv
    ) -> Stream[T]:
        return Stream(lambda: self._fn(query, env))

    def __rmatmul__(self, other: StreamTransformer) -> "PromptingPolicy":
        """
        Compose a prompting policy with a stream transformer.
        """
        if not isinstance(other, StreamTransformer):  # pyright: ignore[reportUnnecessaryIsInstance]
            return NotImplemented
        return self._compose_with_stream_transformer(other)

    def _compose_with_stream_transformer(
        self,
        trans: StreamTransformer,
    ) -> "PromptingPolicy":
        def policy[T](
            query: dp.AttachedQuery[T], env: PolicyEnv
        ) -> dp.StreamGen[T]:
            return trans(self(query, env), env).gen()

        return PromptingPolicy(policy)


class _PromptingPolicyFn(Protocol):
    def __call__[T](
        self,
        query: dp.AttachedQuery[T],
        env: PolicyEnv,
    ) -> dp.StreamGen[T]: ...


class _ParametricPromptingPolicyFn[**A](Protocol):
    def __call__[T](
        self,
        query: dp.AttachedQuery[T],
        env: PolicyEnv,
        *args: A.args,
        **kwargs: A.kwargs,
    ) -> dp.StreamGen[T]: ...


class _ParametricPromptingPolicy[**A](Protocol):
    def __call__(
        self, *args: A.args, **kwargs: A.kwargs
    ) -> PromptingPolicy: ...


def prompting_policy[**A](
    fn: _ParametricPromptingPolicyFn[A],
) -> _ParametricPromptingPolicy[A]:
    """
    Convenience decorator for creating parametric prompting policies
    (i.e., functions that return prompting policies).

    See the definition of `few_shot` for an example.

    Attributes:
        fn: A function that takes an attached query, a policy
            environment, and additional parameters as arguments and
            returns a search stream generator (`SearchStreamGen`).

    Returns:
        A function that takes the additional parameters of `fn` as
        arguments and returns a prompting policy (`PromptingPolicy`).
    """

    def parametric(*args: A.args, **kwargs: A.kwargs) -> PromptingPolicy:
        def policy[T](
            query: dp.AttachedQuery[T], env: PolicyEnv
        ) -> dp.StreamGen[T]:
            return fn(query, env, *args, **kwargs)

        return PromptingPolicy(policy)

    return parametric


#####
##### Tree Transformers
#####


class PureTreeTransformerFn[A: Node, B: Node](Protocol):
    """
    A function that maps any tree with signature `A | N` to a tree with
    signature `B | N`, for all `N`.
    """

    def __call__[N: Node, P, T](
        self, tree: dp.Tree[A | N, P, T]
    ) -> dp.Tree[B | N, P, T]: ...


class _ContextualTreeTransformerFn[A: Node, B: Node](Protocol):
    def __call__(
        self, env: PolicyEnv, policy: Any
    ) -> PureTreeTransformerFn[A, B]: ...


class _ParametricContextualTreeTransformerFn[A: Node, B: Node, **C](Protocol):
    def __call__(
        self, env: PolicyEnv, policy: Any, *args: C.args, **kwargs: C.kwargs
    ) -> PureTreeTransformerFn[A, B]: ...


@dataclass
class ContextualTreeTransformer[A: Node, B: Node]:
    """
    Wrapper for a function that maps trees to trees, possibly
    changing their signature. Can depend on the global policy
    environment (hence the *contextual* aspect).

    Contextual tree transformers can be composed with search policies to
    modify their accepted signature. They can be convniently defined
    using the `contextual_tree_transformer` decorator. See
    `elim_compute` and `elim_messages` for examples.

    Type Parameters:
        A: The type of nodes that the transformer removes from search
            policy signature.
        B: The type of nodes that the transformer adds to search policy
            signature (or the bottom type `Never` if no types are
            added).

    Attributes:
        fn: A function that takes a policy environment and an inner
            policy as arguments (hence the *contextual* aspect) and
            returns a pure tree transformer (`PureTreeTransformerFn`)
    """

    fn: _ContextualTreeTransformerFn[A, B]

    @staticmethod
    def pure(
        fn: PureTreeTransformerFn[A, B],
    ) -> "ContextualTreeTransformer[A, B]":
        """
        Create a contextual tree transformer from a pure tree
        transformer.
        """

        def contextual(env: PolicyEnv, policy: Any):
            return fn

        return ContextualTreeTransformer(contextual)

    def __rmatmul__[N: Node](
        self, search_policy: "SearchPolicy[B | N]"
    ) -> "SearchPolicy[A | N]":
        """
        Compose a contextual tree transformer with a search policy.
        """
        if not isinstance(search_policy, SearchPolicy):  # pyright: ignore[reportUnnecessaryIsInstance]
            return NotImplemented

        def new_search_policy[P, T](
            tree: dp.Tree[A | N, P, T],
            env: PolicyEnv,
            policy: P,
        ) -> dp.StreamGen[T]:
            new_tree = self.fn(env, policy)(tree)
            return search_policy(new_tree, env, policy).gen()

        return SearchPolicy(new_search_policy)


def contextual_tree_transformer[A: Node, B: Node, **C](
    f: _ParametricContextualTreeTransformerFn[A, B, C], /
) -> Callable[C, ContextualTreeTransformer[A, B]]:
    """
    A convenience decorator for defining contextual tree transformers.

    See the implementation of `elim_messages` for an example.

    Arguments:
        f: A function that takes a policy environment, an inner policy,
            and additional parameters as arguments and returns a pure
            tree transformer (`PureTreeTransformerFn`).

    Returns:
        A function that takes the additional parameters of `f` as
        arguments and returns a contextual tree transformer
        (`ContextualTreeTransformer`).
    """

    def parametric(*args: C.args, **kwargs: C.kwargs):
        def contextual(env: PolicyEnv, policy: Any):
            return f(env, policy, *args, **kwargs)

        return ContextualTreeTransformer(contextual)

    return parametric


#####
##### Checking consistency of strategies and policies
#####


type _ParametricPolicy[**A, N: Node, P] = Callable[A, Policy[N, P]]


def ensure_compatible[**A, N: Node, P](
    strategy: Callable[..., dp.StrategyComp[N, P, object]],
) -> Callable[[_ParametricPolicy[A, N, P]], _ParametricPolicy[A, N, P]]:
    """
    A decorator that does nothing but allows type-checkers to ensure
    that the decorated function returns a policy compatible with its
    strategy argument.
    """

    return lambda f: f


#####
##### Inner Policy Dictionaries
#####


type IPDict = Mapping[str, Policy[Any, Any] | PromptingPolicy]
"""
Type of an Inner-Policy Dictionary.

Inner-Policy dictionaries allow to define strategies in a more concise
way in exchange for less static type safety.

Normally, an *inner policy type* must be defined for every strategy, and
opaque spaces are created from queries or strategy by passing the
`using` method a mapping from the ambient inner policy to a proper
sub-policy, often in the form of an anonymous function:

```python
@dataclass class MyInnerPolicy:
    foo: PromptingPolicy
    # etc

def my_strategy() -> Strategy[Branch, MyInnerPolicy, str]:
    x = yield from branch(Foo().using(lambda p: p.foo))
    # etc
```
    
As an alternative, one can have a strategy use an inner policy
dictionary, by passing ellipses (`...`) to the `using` method:

```python
def my_strategy() -> Strategy[Branch, IPDict, str]:
    x = yield from branch(Foo().using(...))
    # etc
```

When doing so, a simple Python dictionary can be used as an inner
policy, whose keys are space tags (the same tags can be referenced in
demonstration tests). In the example above, and since a spaces induced
by a query inherits its name as a tag by default, one can define an
inner policy for `my_strategy` as:

```python
{"Foo": foo_prompting_policy, ...}
```

A conjunction of tags can also be specified, separated by `&` (without
spaces). For example, `{"tag1&tag2": pp, ...}` associates prompting
policies `pp` to spaces with both tags `tag1` and `tag2`. New tags can
be added to a space builder using the `SpaceBuilder.tagged` method.

!!! info
    If several entries of the inner policy dictionary apply for a given
    instance of `.using(...)`, a runtime error is raised.

See `tests/example_strategies:generate_number` for another example.
"""


def _dict_ip_key_match(key: str, tags: Sequence[dp.Tag]) -> bool:
    key_tags = key.split("&")
    return set(key_tags).issubset(set(tags))


def dict_subpolicy(ip: IPDict, tags: Sequence[dp.Tag]) -> Any:
    """
    Retrieve a sub-policy from an internal policy dictionary, using the
    tags of a particular space.
    """
    # TODO: add a type check to make sure we get a prompting policy or a
    # policy out of it (whatever appropriate)?
    matches = [k for k in ip if _dict_ip_key_match(k, tags)]
    if not matches:
        raise ValueError(
            f"Missing sub-policy for space with tags {tags}.\n"
            + f"Provided keys are: {list(ip)}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple sub-policies match tags {tags}.\n"
            + f"Matching keys are: {matches}\n"
            + f"Provided keys are: {list(ip)}"
        )
    return ip[matches[0]]


#####
##### Utilities
#####


def query_dependent(
    f: Callable[[dp.AbstractQuery[Any]], PromptingPolicy],
) -> PromptingPolicy:
    """
    Create a prompting policy that is dependent on the exact query being
    processed (most prompting policies do not inspect the name or
    arguments of their query argument).
    """

    def policy[T](
        query: dp.AttachedQuery[T], env: PolicyEnv
    ) -> dp.StreamGen[T]:
        query_policy = f(query.query)
        return query_policy(query, env).gen()

    return PromptingPolicy(policy)


def unsupported_node(node: dp.Node) -> NoReturn:
    """
    Raise an exception indicating that a node has an unsupported type.

    See `dfs` for an example usage.
    """
    assert False, f"Unsupported node type: {type(node)}."

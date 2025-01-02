"""
Decorators and shortcuts for the Delphyne DSL.
"""

from typing import Any, Protocol

from delphyne.core.environment import PolicyEnv
from delphyne.core.streams import StreamRet
from delphyne.core.trees import (
    AttachedQuery,
    Node,
    PromptingPolicy,
    SearchPolicy,
    Tree,
)

#####
##### Search Policies
#####


class _ParametricSearchPolicyFn[N: Node, **A](Protocol):
    def __call__[P, T](
        self,
        tree: Tree[N, P, T],
        env: PolicyEnv,
        policy: P,
        *args: A.args,
        **kwargs: A.kwargs,
    ) -> StreamRet[T]: ...


class _ParametricSearchPolicy[N: Node, **A](Protocol):
    def __call__(
        self, *args: A.args, **kwargs: A.kwargs
    ) -> SearchPolicy[N]: ...


def search_policy[N: Node, **A](
    fn: _ParametricSearchPolicyFn[N, A],
) -> _ParametricSearchPolicy[N, A]:
    """
    Wraps a search policy into a callable objects with additional helper
    features (such as the `@` operator for composing with tree
    transformers).
    """

    def parametric(*args: A.args, **kwargs: A.kwargs) -> SearchPolicy[N]:
        def policy[T](
            tree: Tree[N, Any, T], env: PolicyEnv, policy: Any
        ) -> StreamRet[T]:
            return fn(tree, env, policy, *args, **kwargs)

        return SearchPolicy(policy)

    return parametric


#####
##### Prompting Policies
#####


class _ParametricPromptingPolicyFn[**A](Protocol):
    def __call__[T](
        self,
        query: AttachedQuery[T],
        env: PolicyEnv,
        *args: A.args,
        **kwargs: A.kwargs,
    ) -> StreamRet[T]: ...


class _ParametricPromptingPolicy[**A](Protocol):
    def __call__(
        self, *args: A.args, **kwargs: A.kwargs
    ) -> PromptingPolicy: ...


def prompting_policy[**A](
    fn: _ParametricPromptingPolicyFn[A],
) -> _ParametricPromptingPolicy[A]:
    def parametric(*args: A.args, **kwargs: A.kwargs) -> PromptingPolicy:
        def policy[T](query: AttachedQuery[T], env: PolicyEnv) -> StreamRet[T]:
            return fn(query, env, *args, **kwargs)

        return PromptingPolicy(policy)

    return parametric

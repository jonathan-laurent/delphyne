"""
Abstract Types for Policies.

Instances with more features are defined in the standard library.
"""

from abc import ABC
from typing import Generic, Protocol, TypeVar

from delphyne.core.streams import AbstractStream
from delphyne.core.trees import AttachedQuery, Node, Tree

N = TypeVar("N", bound=Node, contravariant=True)
P = TypeVar("P", covariant=True)
E = TypeVar("E", contravariant=True)


class AbstractPolicy(Generic[E, N, P], ABC):
    """
    A pair of a search policy and of an inner policy.

    More preciely, a policy for trees with effects `N` (contravariant)
    gathers a search policy handling `N` along with an inner policy
    object of type `P` (covariant).
    """

    @property
    def search(self) -> "AbstractSearchPolicy[E, N]": ...
    @property
    def inner(self) -> P: ...


class AbstractSearchPolicy(Generic[E, N], Protocol):
    """
    A search policy takes as arguments a tree with a given signature
    (covariant type parameter `N`), a global policy environment, and an
    inner policy with appropriate type, and returns a search stream.
    """

    def __call__[P, T](
        self, tree: "Tree[N, P, T]", env: E, policy: P
    ) -> AbstractStream[T]: ...


class AbstractPromptingPolicy(Generic[E], Protocol):
    """
    A prompting policy takes as arguments a query (attached to a
    specific node) and a global policy environment, and returns a search
    stream.
    """

    def __call__[T](
        self, query: AttachedQuery[T], env: E
    ) -> AbstractStream[T]: ...

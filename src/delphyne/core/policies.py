"""
Abstract Types for Policies.

Instances with more features are defined in the standard library.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from delphyne.core.streams import AbstractStream
from delphyne.core.trees import AttachedQuery, Node, Tree

N = TypeVar("N", bound=Node, contravariant=True)
P = TypeVar("P", covariant=True)
E = TypeVar("E", contravariant=True)


class AbstractPolicy(Generic[E, N, P], ABC):
    """
    A policy maps a tree with a given signature (contravariant parameter
    N) and inner policy type (covariant parameter P) to a search stream.
    """

    @abstractmethod
    def __call__[T](self, tree: "Tree[N, P, T]", env: E) -> AbstractStream[T]:
        pass


class AbstractSearchPolicy(Generic[E, N], ABC):
    """
    A search policy takes as arguments a tree with a given signature
    (covariant type parameter `N`), a global policy environment, and an
    inner policy with appropriate type, and returns a search stream.
    """

    @abstractmethod
    def __call__[P, T](
        self, tree: "Tree[N, P, T]", env: E, policy: P
    ) -> AbstractStream[T]:
        pass


class AbstractPromptingPolicy(Generic[E], ABC):
    """
    A prompting policy takes as arguments a query (attached to a
    specific node) and a global policy environment, and returns a search
    stream.
    """

    @abstractmethod
    def __call__[T](
        self, query: AttachedQuery[T], env: E
    ) -> AbstractStream[T]:
        pass

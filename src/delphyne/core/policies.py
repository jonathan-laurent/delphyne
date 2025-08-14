"""
Abstract Types for Policies.

Instances with more features are defined in the standard library.
"""

from abc import ABC
from typing import Generic, Protocol, TypeVar

from delphyne.core import trees as tr
from delphyne.core.environments import PolicyEnv
from delphyne.core.streams import AbstractSearchStream
from delphyne.core.trees import AttachedQuery, Node, Tree

N_po = TypeVar("N_po", bound=Node, contravariant=True)
P_po = TypeVar("P_po", covariant=True)


class AbstractPolicy(Generic[N_po, P_po], ABC):
    """
    A pair of a search policy and of an inner policy.

    More preciely, a policy for trees with effects `N` (contravariant)
    gathers a search policy handling `N` along with an inner policy
    object of type `P` (covariant).
    """

    @property
    def search(self) -> "AbstractSearchPolicy[N_po]": ...
    @property
    def inner(self) -> P_po: ...


class AbstractSearchPolicy[N: tr.Node](Protocol):
    """
    A search policy takes as arguments a tree with a given signature
    (covariant type parameter `N`), a global policy environment, and an
    inner policy with appropriate type, and returns a search stream.
    """

    def __call__[P, T](
        self, tree: "Tree[N, P, T]", env: PolicyEnv, policy: P
    ) -> AbstractSearchStream[T]: ...


class AbstractPromptingPolicy(Protocol):
    """
    A prompting policy takes as arguments a query (attached to a
    specific node) and a global policy environment, and returns a search
    stream.
    """

    def __call__[T](
        self, query: AttachedQuery[T], env: PolicyEnv
    ) -> AbstractSearchStream[T]: ...

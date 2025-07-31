"""
Abstract Policies
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
    A policy for trees with effects `N` gathers a search policy handling
    `N` along with an inner policy object of type `P`.
    """

    @property
    def search(self) -> "AbstractSearchPolicy[N_po]": ...
    @property
    def inner(self) -> P_po: ...


class AbstractSearchPolicy[N: tr.Node](Protocol):
    def __call__[P, T](
        self, tree: "Tree[N, P, T]", env: PolicyEnv, policy: P
    ) -> AbstractSearchStream[T]: ...


class AbstractPromptingPolicy(Protocol):
    def __call__[T](
        self, query: AttachedQuery[T], env: PolicyEnv
    ) -> AbstractSearchStream[T]: ...

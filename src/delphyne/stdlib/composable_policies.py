"""
Composable search policies
"""

from collections.abc import Callable
from typing import Protocol, TypeGuard, cast

from delphyne.core.trees import Node, Success, Tree
from delphyne.stdlib.generators import (
    GenEnv,
    GenResponse,
    GenRet,
    SearchPolicy,
)


# fmt: off


type _Guard[N] = Callable[[object], TypeGuard[N]]


class _Recursor[N: Node, T](Protocol):
    def __call__(self, env: GenEnv, tree: Tree[N, T], /) -> GenRet[T]: ...


class _Visitor[N: Node](Protocol):
    def __call__[M: Node, T](
        self,
        env: GenEnv,
        node: N,
        tree: Tree[M, T],
        recurse: _Recursor[M, T],
        /
    ) -> GenRet[T]:
        ...


type PolicyElement[N: Node] = tuple[_Guard[N], _Visitor[N]]


# It would be tempting to wrap composable policies in dataclasses.
# However, pyright fails to properly infer P if composable policies are
# not functions. See https://github.com/microsoft/pyright/issues/7782.
type ComposablePolicy[P, N: Node] = (
    Callable[[P], tuple[_Guard[N], _Visitor[N]]]
)


def composed_policy[P, N: Node](
    cp: ComposablePolicy[P, N]
) -> SearchPolicy[P, N]:
    def policy[T](env: GenEnv, tree: Tree[N, T], params: P) -> GenRet[T]:
        _, visitor = cp(params)
        async def recurse[U](env: GenEnv, tree: Tree[N, U]) -> GenRet[U]:
            node = tree.node
            if isinstance(node, Success):
                node = cast(Success[U], node)
                yield GenResponse([node.success])
            else:
                async for ret in visitor(env, node, tree, recurse):
                    yield ret
        return recurse(env, tree)
    return policy


def compose[P, N: Node, M: Node](
    cpl: ComposablePolicy[P, N],
    cpr: ComposablePolicy[P, M]
) -> ComposablePolicy[P, N | M]:
    def launch(params: P) -> tuple[_Guard[N | M], _Visitor[N | M]]:
        guardl, visitl = cpl(params)
        guardr, visitr = cpr(params)
        def guard(node: object) -> TypeGuard[N | M]:
            return guardl(node) or guardr(node)
        def visit[T](
            env: GenEnv,
            node: N | M, tree:
            Tree[N | M, T], recurse:
            _Recursor[N | M, T]
        ) -> GenRet[T]:
            if guardl(node):
                return visitl(env, node, tree, recurse)
            elif guardr(node):
                return visitr(env, node, tree, recurse)
            assert False
        return guard, visit
    return launch

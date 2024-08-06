from typing import Callable, TypeGuard

from delphyne.core.trees import Node, Tree
from delphyne.stdlib.composable_policies import PolicyElement
from delphyne.stdlib.generators import GenEnv, GenResponse, GenRet
from delphyne.stdlib.nodes import Failure, Run


def handle_failure(params: object) -> PolicyElement[Failure]:

    async def visit[N: Node, T](
        env: GenEnv,
        node: Failure,
        tree: Tree[N, T],
        recurse: Callable[[GenEnv, Tree[N, T]], GenRet[T]]
    ) -> GenRet[T]:  # fmt: skip
        yield GenResponse([])

    def guard(obj: object) -> TypeGuard[Failure]:
        return isinstance(obj, Failure)

    return guard, visit


def handle_run[P](params: P) -> PolicyElement[Run[P]]:

    async def visit[N: Node, T](
        env: GenEnv,
        node: Run[P],
        tree: Tree[N, T],
        recurse: Callable[[GenEnv, Tree[N, T]], GenRet[T]]
    ) -> GenRet[T]:  # fmt: skip
        async for resp in node.gen(env, tree, params):
            if not resp.items:
                yield resp
            if resp.items:
                async for ret in recurse(env, tree.child(resp.items[0])):
                    yield ret
                return

    def guard(obj: object) -> TypeGuard[Run[P]]:
        return isinstance(obj, Run)

    return guard, visit

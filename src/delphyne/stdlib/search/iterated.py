"""
Utilities for writing blacklist enumerators.
"""

from collections.abc import Callable, Sequence
from typing import cast

from delphyne.core.tracing import Outcome
from delphyne.core.trees import Navigation, Node, Strategy, Success, Tree
from delphyne.stdlib.dsl import (
    GeneratorConvertible,
    convert_to_parametric_generator,
    strategy,
)
from delphyne.stdlib.generators import GenEnv, Generator, GenResponse, GenRet
from delphyne.stdlib.nodeclasses import nodeclass
from delphyne.stdlib.search_envs import HasSearchEnv


@nodeclass(frozen=True)
class Iterated[P, T](Node):
    next: Callable[[tuple[Outcome[T], ...]], Generator[P, T]]

    def navigate(self) -> Navigation:
        return (yield self.next(()))


async def search_iterated[P, T](
    env: GenEnv, tree: Tree[Iterated[P, T], T], params: P
) -> GenRet[T]:  # fmt: skip
    node = tree.node
    if isinstance(node, Success):
        yield GenResponse([node.success])
        return
    blacklist: list[Outcome[T]] = []
    while True:
        async for resp in node.next(tuple(blacklist))(env, tree, params):
            if not resp.items:
                yield resp
            else:
                item = resp.items[0]
                # TODO: doing this would be INCORRECT and we have to
                # have stronger measures to prevent it.
                #     yield GenResponse([item])
                # Only success nodes can yield items.
                recursive = search_iterated(env, tree.child(item), params)
                async for ret in recursive:
                    yield ret
                blacklist.append(item)
                break
        else:
            break


# TODO: pyright has a hard time here since what we would like to infer is
# \forall P. WrappedParametricStrategy[P, ...]
@strategy(search_iterated)  # pyright: ignore [reportArgumentType]
def iterated[P: HasSearchEnv, T](
    next: Callable[[Sequence[T]], GeneratorConvertible[P, T]]
) -> Strategy[Iterated[P, T], T]:  # fmt: skip
    ret = yield Iterated(convert_to_parametric_generator(next))
    return cast(T, ret)

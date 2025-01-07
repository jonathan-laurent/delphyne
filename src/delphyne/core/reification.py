"""
Reify strategies into trees using thermometer continuations.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from delphyne.core import environments as en
from delphyne.core import refs
from delphyne.core import trees as tr
from delphyne.core.refs import GlobalNodePath, SpaceName, Value
from delphyne.core.trees import Node, Strategy, StrategyComp, Success, Tree
from delphyne.utils.typing import NoTypeInfo, TypeAnnot

#####
##### Reification Interface
#####


type TreeCache = dict[refs.GlobalNodePath, Tree[Any, Any, Any]]
"""
A cache for never recomputing the same node twice.
"""


type TreeHook = Callable[[Tree[Any, Any, Any]], None]
"""
A function to be called every time a new tree node is created.
"""


@dataclass(frozen=True)
class TreeMonitor:
    """
    When `reify` is called with a monitor, it makes sure that proper
    hooks are called and proper caching is performed.
    """

    cache: TreeCache | TreeCache | None = None
    hooks: Sequence[TreeHook] = ()


def reify[N: Node, P, T](
    strategy: StrategyComp[N, P, T],
    monitor: TreeMonitor = TreeMonitor(),
) -> Tree[N, P, T]:
    """
    Reify a strategy into a tree.
    """
    return _reify(strategy, None, monitor)


def tracer_hook(tracer: en.Tracer) -> Callable[[Tree[Any, Any, Any]], None]:
    return lambda tree: tracer.trace_node(tree.ref)


#####
##### Reification Implementation
#####


type _NonEmptyGlobalPath = tuple[GlobalNodePath, refs.SpaceRef, refs.NodePath]
"""
Encodes a non-empty global node path.

More precisely, `(gr, sr, nr)` encodes `(*gr, (sr, nr))`
"""


def _nonempty_path(ref: _NonEmptyGlobalPath) -> GlobalNodePath:
    (gr, sr, nr) = ref
    return (*gr, (sr, nr))


def _reify[N: Node, P, T](
    strategy: StrategyComp[N, P, T],
    root_ref: _NonEmptyGlobalPath | None,
    monitor: TreeMonitor,
) -> Tree[N, P, T]:
    """
    Reify a strategy into a tree.

    This version is private because it exposes the `root_ref` argument.
    Outside of these modules, trees can only be reified relative to the
    global origin.
    """

    def aux(
        strategy: StrategyComp[N, P, T],
        actions: Sequence[object],
        ref: _NonEmptyGlobalPath,
        node: N | Success[T],
        cur_generator: Strategy[N, P, T],
    ) -> Tree[N, P, T]:
        generator_valid = True

        def child(action: Value) -> Tree[N, P, T]:
            action_raw = refs.drop_refs(action)
            action_ref = refs.value_ref(action)
            del action
            assert node.valid_action(action_raw), (
                "Invalid action for node of type "
                + f"{type(node)}: {action_raw}."
            )
            # Compute new references and use the cache if necessary
            gr, cr, nr = ref
            new_ref = (gr, cr, (*nr, action_ref))
            if monitor.cache is not None:
                cached = monitor.cache.get(_nonempty_path(new_ref))
                if cached is not None:
                    return cached
            # If `child` has never been called before, the generator
            # that yielded the current node is still valid and can be
            # reused. For subsequent calls, `self._cur_generator` is
            # None and the strategy is resimulated from scratch
            # following `self._path`.
            nonlocal generator_valid
            new_actions = (*actions, action_raw)
            if generator_valid:
                pre_node = _send_action(cur_generator, action_raw)
                new_gen = cur_generator
                generator_valid = False
            else:
                pre_node, new_gen = _recompute(strategy, new_actions)
            ret_type = strategy.return_type()
            new_node = _finalize_node(pre_node, new_ref, ret_type, monitor)
            return aux(strategy, new_actions, new_ref, new_node, new_gen)

        tree = Tree(node, child, _nonempty_path(ref))
        if monitor.cache is not None:
            monitor.cache[_nonempty_path(ref)] = tree
        for hook in monitor.hooks:
            hook(tree)
        return tree

    if root_ref is None:
        root_ref = ((), refs.MAIN_SPACE, ())
    pre_node, gen = _recompute(strategy, ())
    ret_type = strategy.return_type()
    node = _finalize_node(pre_node, root_ref, ret_type, monitor)
    return aux(strategy, (), root_ref, node, gen)


@dataclass
class _PreSuccess[T]:
    value: T


type _PreNode[N: Node, P, T] = tr.NodeBuilder[N, P] | _PreSuccess[T]


def _send_action[N: Node, P, T](
    gen: Strategy[N, P, T], action: object
) -> _PreNode[N, P, T]:
    try:
        return gen.send(action)
    except StopIteration as e:
        v = cast(T, e.value)
        return _PreSuccess(v)
    except Exception as e:
        raise tr.StrategyException(e)


def _recompute[N: Node, P, T](
    strategy: StrategyComp[N, P, T],
    actions: Sequence[object],
) -> tuple[_PreNode[N, P, T], Strategy[N, P, T]]:
    gen = strategy()
    mknode = _send_action(gen, None)
    for a in actions:
        mknode = _send_action(gen, a)
    return mknode, gen


def _finalize_node[N: Node, P, T](
    pre_node: _PreNode[N, P, T],
    ref: _NonEmptyGlobalPath,
    type: TypeAnnot[T] | NoTypeInfo,
    monitor: TreeMonitor,
) -> N | Success[T]:
    if not isinstance(pre_node, _PreSuccess):
        return pre_node.build_node(_BuilderExecutor(ref, monitor))
    (gr, sr, nr) = ref
    ser = refs.SpaceElementRef(sr, nr)
    return Success(refs.Tracked(pre_node.value, ser, gr, type))


#####
##### Builder Executor
#####


@dataclass(frozen=True)
class _BuilderExecutor(tr.AbstractBuilderExecutor):
    """
    Allows spawning arbitrary parametric subspaces at a given node.
    """

    _ref: "_NonEmptyGlobalPath"
    _monitor: "TreeMonitor"

    def parametric[S](
        self,
        space_name: SpaceName,
        parametric_builder: Callable[..., tr.Builder[S]],
    ) -> Callable[..., S]:
        def run_builder(*args: Any) -> S:
            args_raw = [refs.drop_refs(arg) for arg in args]
            args_ref = tuple(refs.value_ref(arg) for arg in args)
            builder = parametric_builder(*args_raw)
            gr = _nonempty_path(self._ref)
            sr = refs.SpaceRef(space_name, args_ref)

            def spawn_tree[N: Node, P, T](
                strategy: StrategyComp[N, P, T],
            ) -> tr.NestedTree[N, P, T]:
                def spawn() -> Tree[N, P, T]:
                    new_ref = (gr, sr, ())
                    if (cache := self._monitor.cache) is not None:
                        cached = cache.get(_nonempty_path(new_ref))
                        if cached is not None:
                            return cached
                    return _reify(strategy, new_ref, self._monitor)

                return tr.NestedTree(strategy, (gr, sr), spawn)

            def spawn_query[T](
                query: tr.AbstractQuery[T],
            ) -> tr.AttachedQuery[T]:
                def answer(
                    mode: refs.AnswerModeName, text: str
                ) -> tr.Tracked[T] | tr.ParseError:
                    ref = refs.SpaceElementRef(sr, refs.Answer(mode, text))
                    atype = query.answer_type()
                    parsed = query.modes()[mode].parse(atype, text)
                    if isinstance(parsed, tr.ParseError):
                        return parsed
                    return tr.Tracked(parsed, ref, gr, query.answer_type())

                return tr.AttachedQuery(query, (gr, sr), answer)

            return builder(spawn_tree, spawn_query)

        return run_builder

    def nonparametric[S](self, name: SpaceName, builder: tr.Builder[S]) -> S:
        return self.parametric(name, lambda: builder)()

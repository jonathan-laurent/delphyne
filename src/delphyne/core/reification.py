"""
Reify strategies into trees using thermometer continuations.
"""

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

from delphyne.core import refs
from delphyne.core import trees as tr
from delphyne.core.queries import AbstractQuery
from delphyne.core.refs import GlobalNodePath, SpaceName, Value
from delphyne.core.trees import Node, Strategy, StrategyComp, Success, Tree
from delphyne.utils.typing import NoTypeInfo, TypeAnnot

#####
##### Reification Interface
#####


type TreeCache = dict[refs.GlobalNodePath, Tree[Any, Any, Any]]
"""
A cache for never recomputing the same node twice.

Each encountered subtree and (recursively) nested tree is stored,
indexed by its global reference.
"""


type TreeHook = Callable[[Tree[Any, Any, Any]], None]
"""
A function to be called every time a new tree node is created.
"""


@dataclass(frozen=True)
class TreeMonitor:
    """
    A record that gathers a tree cache along with a series of hooks to
    be called on node creation.

    When no monitor is passed to `reify`, the resulting tree is a pure,
    immutable datastructure. Passing a monitor allows adding limited
    side effects in the form of caching and hooks.

    Attributes:
        cache: a cache for never recomputing the same node twice.
        hooks: functions to be called every time a new node is created.
            For example, hooks can be used to automatically produce a
            trace that keeps track of all visited nodes (see
            `tracer_hook`).
    """

    cache: TreeCache | TreeCache | None = None
    hooks: Sequence[TreeHook] = ()


def reify[N: Node, P, T](
    strategy: StrategyComp[N, P, T],
    monitor: TreeMonitor = TreeMonitor(),
    allow_noncopyable_actions: bool = False,
) -> Tree[N, P, T]:
    """
    Reify a strategy computation into a tree.

    The resulting tree raises `StrategyException` whenever the
    underlying strategy raises an uncaught exception.

    Internally, trees are implemented by having each node keep track of
    a sequence of actions leading to it. Calling `child` appends a new
    action to this path and replays the strategy computation from the
    start with the new, augmented path.

    Arguments:
        strategy: The strategy to reify.
        monitor: An optional cache, along with node creation hooks.
        allow_noncopyable_actions: Allow
            actions to be nonserializable objects (such as functions)
            that cannot be deepcopied. Allowing such actions opens the
            door to unsafe side effects corrupting the resulting tree,
            so it must be done with care. See discussion below on side
            effects.

    !!! note "On Side Effects in Strategies"
        Strategy functions are allowed to have side effects (see
        `tests/example_strategies/imperative_strategy` for example). For
        this to be sound, actions are always deepcopied before being
        sent back to strategy coroutines. This requirement can be
        weakened by setting `allow_noncopyable_actions` to `True`. In
        this case, noncopyable actions must be immutable. For example, a
        pure function can be used as an action but a closure that
        captures a mutable piece of state cannot (in which case
        computing the child of a node could affect its siblings by
        mutating some actions in their paths).

        In addition, non-copyable strategy arguments must never be
        mutated. Copyable arguments such as lists of copyable values can
        be mutated, since `reify` automaticallt performs deepcopies.
        Noncopyable arguments such as functions are allowed (e.g., to
        enable implementing higher-order strategies) but they must be
        pure and not mutate any state.

        When `allow_noncopyable_actions` is set to `False`, a dynamic
        check is performed to ensure that actions are copyable: an
        action is considered copyable if it can be serialized into JSON
        by pydantic (this is necessary since calling `deepcopy` on a
        closure returns the same closure unchanged).
    """
    return _reify(
        strategy=strategy,
        root_ref=None,
        monitor=monitor,
        allow_noncopyable_actions=allow_noncopyable_actions,
    )


#####
##### Reification Implementation
#####


type _NonEmptyGlobalPath = tuple[GlobalNodePath, refs.SpaceRef, refs.NodePath]
"""
Encodes a non-empty global node path.

More precisely, `(gr, sr, nr)` stands for `(*gr, (sr, nr))`.
"""


def _nonempty_path(ref: _NonEmptyGlobalPath) -> GlobalNodePath:
    (gr, sr, nr) = ref
    return (*gr, (sr, nr))


def _reify[N: Node, P, T](
    strategy: StrategyComp[N, P, T],
    root_ref: _NonEmptyGlobalPath | None,
    monitor: TreeMonitor,
    allow_noncopyable_actions: bool,
    enable_unsound_generator_caching: bool = False,
) -> Tree[N, P, T]:
    """
    Reify a strategy into a tree.

    This version is private because it exposes the `root_ref` argument.
    Outside of these modules, trees can only be reified relative to the
    global origin.

    When `enable_unsound_generator_caching` is set to `True`, an
    optimization is enabled that is unsound in general and that allows
    not replaying the strategy code from scratch every time a child is
    computed (but only when _actual_ branching occurs). This
    optimization is unsafe in general, even with pure strategy code, due
    the way closures capture local variables in Python. Do not use this
    option unless you know exactly what you are doing.
    """

    def aux(
        strategy: StrategyComp[N, P, T],
        actions: Sequence[object],
        ref: _NonEmptyGlobalPath,
        node: N | Success[T],
        cur_generator: Strategy[N, P, T],
    ) -> Tree[N, P, T]:
        generator_valid = enable_unsound_generator_caching

        def child(action: Value) -> Tree[N, P, T]:
            refs.check_local_value(action, _nonempty_path(ref))
            action_raw = refs.drop_refs(action)
            action_ref = refs.value_ref(action)
            # We deepcopy actions and do not allow them to contain
            # functions by default.
            # Indeed, functions may
            if not allow_noncopyable_actions:
                assert _is_copyable(action_raw), (
                    f"Noncopyable action: {action_raw}"
                )
            action_raw = deepcopy(action_raw)
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
            new_node = _finalize_node(
                pre_node, new_ref, ret_type, monitor, allow_noncopyable_actions
            )
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
    node = _finalize_node(
        pre_node, root_ref, ret_type, monitor, allow_noncopyable_actions
    )
    return aux(strategy, (), root_ref, node, gen)


def _is_copyable(obj: object) -> bool:
    """
    To know if an object can be deepcopied safely, we try and serialize
    it using pydantic. Indeed, some objects such as functions (and
    closures in particular) can be passed to `deepcopy` without causing
    an exception to be raised, but are not independent copies.
    """
    import pydantic

    adapter = pydantic.TypeAdapter[Any](Any)
    try:
        adapter.dump_json(obj)
        return True
    except Exception:
        return False


@dataclass
class _PreSuccess[T]:
    value: T


type _PreNode[N: Node, P, T] = tr.NodeBuilder[N, P] | _PreSuccess[T]


def _send_action[N: Node, P, T](
    gen: Strategy[N, P, T], action: object
) -> _PreNode[N, P, T]:
    try:
        return gen.send(deepcopy(action))
    except StopIteration as e:
        v = cast(T, e.value)
        return _PreSuccess(v)
    except Exception as e:
        raise tr.StrategyException(e)


def _recompute[N: Node, P, T](
    strategy: StrategyComp[N, P, T],
    actions: Sequence[object],
) -> tuple[_PreNode[N, P, T], Strategy[N, P, T]]:
    try:
        gen = strategy.run_generator()
    except Exception as e:
        raise tr.StrategyException(e)
    mknode = _send_action(gen, None)
    for a in actions:
        mknode = _send_action(gen, a)
    return mknode, gen


def _finalize_node[N: Node, P, T](
    pre_node: _PreNode[N, P, T],
    ref: _NonEmptyGlobalPath,
    type: TypeAnnot[T] | NoTypeInfo,
    monitor: TreeMonitor,
    allow_noncopyable_actions: bool,
) -> N | Success[T]:
    if not isinstance(pre_node, _PreSuccess):
        return pre_node.build_node(
            _BuilderExecutor(ref, monitor, allow_noncopyable_actions)
        )
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
    _allow_noncopyable_actions: bool

    def parametric[S](
        self,
        space_name: SpaceName,
        parametric_builder: Callable[..., tr.SpaceBuilder[S]],
    ) -> Callable[..., S]:
        def run_builder(*args: Any) -> S:
            for arg in args:
                refs.check_local_value(arg, _nonempty_path(self._ref))
            args_raw = [refs.drop_refs(arg) for arg in args]
            args_ref = tuple(refs.value_ref(arg) for arg in args)
            # We deepcopy the arguments to prevent them being mutated.
            # We allow some of them to contain functions (left
            # unmodified by deepcopy) since this is very useful in
            # general but such functions must _not_ be stateful.
            args_raw = [deepcopy(arg) for arg in args_raw]
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
                    return _reify(
                        strategy,
                        new_ref,
                        self._monitor,
                        self._allow_noncopyable_actions,
                    )

                return tr.NestedTree(strategy, (gr, sr), spawn)

            def spawn_query[T](
                query: tr.AbstractQuery[T],
            ) -> tr.AttachedQuery[T]:
                def parse_answer(
                    answer: refs.Answer,
                ) -> tr.Tracked[T] | tr.ParseError:
                    ref = refs.SpaceElementRef(sr, answer)
                    parsed = query.parse_answer(answer)
                    if isinstance(parsed, tr.ParseError):
                        return parsed
                    return tr.Tracked(parsed, ref, gr, query.answer_type())

                return tr.AttachedQuery(query, (gr, sr), parse_answer)

            return builder(spawn_tree, spawn_query)

        return run_builder

    def nonparametric[S](
        self, name: SpaceName, builder: tr.SpaceBuilder[S]
    ) -> S:
        return self.parametric(name, lambda: builder)()


#####
##### Standalone Queries
#####


def spawn_standalone_query[T](query: AbstractQuery[T]) -> tr.AttachedQuery[T]:
    """
    Spawn a standalone query attached to the `MAIN_SPACE` of the global
    origin. Do NOT use this for queries attached to strategies.
    """

    def parse_answer(answer: refs.Answer) -> tr.Tracked[T] | tr.ParseError:
        ref = refs.SpaceElementRef(refs.MAIN_SPACE, answer)
        parsed = query.parse_answer(answer)
        if isinstance(parsed, tr.ParseError):
            return parsed
        return tr.Tracked(parsed, ref, (), query.answer_type())

    return tr.AttachedQuery(query, ((), refs.MAIN_SPACE), parse_answer)

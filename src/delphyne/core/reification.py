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
from delphyne.core.refs import SpaceName, Value
from delphyne.core.trees import Node, Strategy, StrategyComp, Success, Tree
from delphyne.utils.typing import NoTypeInfo, TypeAnnot

#####
##### Reification Interface
#####


type TreeCache = dict[refs.GlobalNodeRef, Tree[Any, Any, Any]]
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
        root_space=refs.GlobalSpacePath(()),
        monitor=monitor,
        allow_noncopyable_actions=allow_noncopyable_actions,
    )


#####
##### Reification Implementation
#####


def _reify[N: Node, P, T](
    strategy: StrategyComp[N, P, T],
    root_space: refs.GlobalSpacePath,
    monitor: TreeMonitor,
    allow_noncopyable_actions: bool,
    enable_unsound_generator_caching: bool = False,
) -> Tree[N, P, T]:
    """
    Reify a strategy into a tree.

    This version is private because it exposes the `root_space` argument.
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
        actions_and_refs: Sequence[tuple[object, refs.GlobalNodeRef]],
        ref: refs.GlobalNodeRef,
        node: N | Success[T],
        cur_generator: Strategy[N, P, T],
    ) -> Tree[N, P, T]:
        generator_valid = enable_unsound_generator_caching

        def child(action: Value) -> Tree[N, P, T]:
            refs.check_local_value(action, ref)
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
            new_ref = ref.child(action_ref)
            if monitor.cache is not None:
                cached = monitor.cache.get(new_ref)
                if cached is not None:
                    return cached
            # If `child` has never been called before, the generator
            # that yielded the current node is still valid and can be
            # reused. For subsequent calls, `self._cur_generator` is
            # None and the strategy is resimulated from scratch
            # following `self._path`.
            nonlocal generator_valid
            new_actions_and_refs = (*actions_and_refs, (action_raw, new_ref))
            if generator_valid:
                pre_node = _send_action(cur_generator, (action_raw, new_ref))
                new_gen = cur_generator
                generator_valid = False
            else:
                pre_node, new_gen = _recompute(strategy, new_actions_and_refs)
            ret_type = strategy.return_type()
            new_node = _finalize_node(
                pre_node, new_ref, ret_type, monitor, allow_noncopyable_actions
            )
            return aux(
                strategy, new_actions_and_refs, new_ref, new_node, new_gen
            )

        tree = Tree(node, child, ref)
        if monitor.cache is not None:
            monitor.cache[ref] = tree
        for hook in monitor.hooks:
            hook(tree)
        return tree

    pre_node, gen = _recompute(strategy, ())
    ret_type = strategy.return_type()
    ref = refs.GlobalNodeRef(root_space, refs.NodePath(()))
    node = _finalize_node(
        pre_node, ref, ret_type, monitor, allow_noncopyable_actions
    )
    return aux(strategy, (), ref, node, gen)


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
    gen: Strategy[N, P, T],
    action_and_ref: tuple[object, refs.GlobalNodeRef] | None,
) -> _PreNode[N, P, T]:
    try:
        if action_and_ref is not None:
            action, ref = action_and_ref
            return gen.send((deepcopy(action), ref))
        return next(gen)  # `gen.send(None)` gives a type error in pyright
    except StopIteration as e:
        v = cast(T, e.value)
        return _PreSuccess(v)
    except Exception as e:
        raise tr.StrategyException(e)


def _recompute[N: Node, P, T](
    strategy: StrategyComp[N, P, T],
    actions_and_refs: Sequence[tuple[object, refs.GlobalNodeRef]],
) -> tuple[_PreNode[N, P, T], Strategy[N, P, T]]:
    try:
        # Exceptions may be raised even before the generator starts, if
        # the strategy functions is passed wrong arguments for example.
        gen = strategy.run_generator()
    except Exception as e:
        raise tr.StrategyException(e)
    mknode = _send_action(gen, None)
    for ar in actions_and_refs:
        mknode = _send_action(gen, ar)
    return mknode, gen


def _finalize_node[N: Node, P, T](
    pre_node: _PreNode[N, P, T],
    ref: refs.GlobalNodeRef,
    type: TypeAnnot[T] | NoTypeInfo,
    monitor: TreeMonitor,
    allow_noncopyable_actions: bool,
) -> N | Success[T]:
    if not isinstance(pre_node, _PreSuccess):
        return pre_node.build_node(
            _BuilderExecutor(ref, monitor, allow_noncopyable_actions)
        )
    parent_node, parent_space = ref.space.split()
    ser = refs.SpaceElementRef(parent_space, ref.path)
    return Success(refs.Tracked(pre_node.value, ser, parent_node, type))


#####
##### Builder Executor
#####


@dataclass(frozen=True)
class _BuilderExecutor(tr.AbstractBuilderExecutor):
    """
    Allows spawning arbitrary parametric subspaces at a given node.
    """

    _ref: refs.GlobalNodeRef
    _monitor: "TreeMonitor"
    _allow_noncopyable_actions: bool

    def parametric[S: tr.Space[Any]](
        self,
        space_name: SpaceName,
        parametric_builder: Callable[..., tr.SpaceBuilder[S]],
    ) -> Callable[..., S]:
        def run_builder(*args: Any) -> S:
            for arg in args:
                refs.check_local_value(arg, self._ref)
            args_raw = [refs.drop_refs(arg) for arg in args]
            args_ref = tuple(refs.value_ref(arg) for arg in args)
            # We deepcopy the arguments to prevent them being mutated.
            # We allow some of them to contain functions (left
            # unmodified by deepcopy) since this is very useful in
            # general but such functions must _not_ be stateful.
            args_raw = [deepcopy(arg) for arg in args_raw]
            builder = parametric_builder(*args_raw)
            sr = refs.SpaceRef(space_name, args_ref)

            def spawn_tree[N: Node, P, T](
                strategy: StrategyComp[N, P, T],
            ) -> tr.NestedTree[N, P, T]:
                new_ref = self._ref.nested_tree(sr)

                def spawn() -> Tree[N, P, T]:
                    if (cache := self._monitor.cache) is not None:
                        cached = cache.get(new_ref)
                        if cached is not None:
                            return cached
                    return _reify(
                        strategy,
                        new_ref.space,
                        self._monitor,
                        self._allow_noncopyable_actions,
                    )

                return tr.NestedTree(strategy, new_ref.space, spawn)

            def spawn_query[T](
                query: tr.AbstractQuery[T],
            ) -> tr.AttachedQuery[T]:
                space_ref = self._ref.nested_space(sr)

                def parse_answer(
                    answer: refs.Answer,
                ) -> tr.Tracked[T] | tr.ParseError:
                    parent_node, parent_space = space_ref.split()
                    ref = refs.SpaceElementRef(parent_space, answer)
                    parsed = query.parse_answer(answer)
                    if isinstance(parsed, tr.ParseError):
                        return parsed
                    return tr.Tracked(
                        parsed, ref, parent_node, query.answer_type()
                    )

                return tr.AttachedQuery(query, space_ref, parse_answer)

            return builder(spawn_tree, spawn_query)

        return run_builder

    def nonparametric[S: tr.Space[Any]](
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
        ref = refs.SpaceElementRef(None, answer)
        parsed = query.parse_answer(answer)
        if isinstance(parsed, tr.ParseError):
            return parsed
        return tr.Tracked(parsed, ref, None, query.answer_type())

    space_ref = refs.GlobalSpacePath(())
    return tr.AttachedQuery(query, space_ref, parse_answer)

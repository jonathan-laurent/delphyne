"""
Language server for Delphyne demonstration files.
"""

import functools
import importlib
import json
import pprint
import sys
import typing
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Never, Sequence, cast, get_type_hints

import pydantic

from delphyne.core import demos, inspect, parse
from delphyne.core import pprint as dpy_pprint
from delphyne.core import refs, trees
from delphyne.core.demos import (
    Demonstration,
    NodeLabel,
    TestCommandString,
    TestStep,
)
from delphyne.core.inspect import FunctionWrapper, remove_wrappers
from delphyne.core.queries import AnyQuery, ParseError, Query
from delphyne.core.refs import NodeId, SubtreeOf
from delphyne.core.strategies import StrategyException, StrategyTree
from delphyne.core.tracing import QueryOrigin, Tracer, drop_refs, value_type
from delphyne.core.trees import Node, StrategyComp, StrategyInstance, Success
from delphyne.server import feedback as fb
from delphyne.server.navigation import (
    HintResolver,
    HintReverseMap,
    Interrupted,
    InvalidSubchoice,
    NavigationInfo,
    NavigationTree,
    ReachedFailureNode,
    RefResolver,
    Stuck,
    TraceReverseMap,
)
from delphyne.utils.typing import NoTypeInfo, TypeAnnot


# Autoreload is presently unsound since we store loaded objects in the
# example cache.
AUTO_RELOAD = False


#####
##### Environment Execution Context
#####


@dataclass
class ExecutionContext:
    strategy_dirs: Sequence[Path]
    modules: Sequence[str]

    def find_object(self, name: str) -> Any:
        """
        Find a Python object by name, by going through a list of
        candidate modules.
        """
        with _append_path(self.strategy_dirs):
            for module_name in self.modules:
                try:
                    module = __import__(module_name)
                    if AUTO_RELOAD:
                        module = importlib.reload(module)
                    return getattr(module, name)
                except AttributeError:
                    pass
        assert False, f"Object not found: {name}"

    def find_and_instantiate_strategy(
        self, name: str, args: dict[str, Any]
    ) -> StrategyComp[Any, Any]:
        f = remove_wrappers(self.find_object(name))
        pargs = _parse_strategy_args(f, args)
        return functools.partial(f, **pargs)

    def find_and_instantiate_wrapped_strategy(
        self, name: str, args: dict[str, Any]
    ) -> StrategyComp[Any, Any]:
        f = self.find_object(name)
        assert isinstance(f, FunctionWrapper)  # TODO
        f = cast(Any, f)
        pargs = _parse_strategy_args(remove_wrappers(f), args)
        return f(**pargs)

    def load_query(self, name: str, args: dict[str, Any]) -> Query[Any, Any]:
        q = cast(type[Query[Any, Any]], self.find_object(name))
        return q.parse(args)


@contextmanager
def _append_path(paths: Sequence[Path]):
    sys.path = [str(p) for p in paths] + sys.path
    yield
    sys.path = sys.path[len(paths) :]


def _parse_strategy_args(
    f: Callable[..., Any], args: dict[str, Any]
) -> dict[str, Any]:
    hints = get_type_hints(f)
    pargs: dict[str, Any] = {}
    for k in args:
        if k not in hints:
            raise ValueError(f"Unknown argument: {k}")
        T = pydantic.TypeAdapter(hints[k])
        pargs[k] = T.validate_python(args[k])
    return pargs


#####
##### Reference simplification
#####


type AnyNavTree = NavigationTree[Any, Any]


@dataclass
class _RefSimplifier:
    tree: AnyNavTree
    hint_rev_map: HintReverseMap

    def action(
        self, id: NodeId, action: refs.ValueRef
    ) -> Sequence[refs.Hint] | None:
        return self.hint_rev_map.actions.get((id, action))

    def path_to(
        self, src_id: NodeId, dst_id: NodeId
    ) -> Sequence[refs.Hint] | None:
        # Compute a sequence of hints necessary to go from the source
        # node __or one of its immediate subtrees__ to the destination.
        if src_id == dst_id:
            return ()
        tracer = self.tree.tracer
        if isinstance(dst_origin := tracer.nodes[dst_id], SubtreeOf):
            if dst_origin.node == src_id:
                return ()
            return None
        before = dst_origin.node
        action = dst_origin.action
        action_hints = self.hint_rev_map.actions.get((before, action))
        if action_hints is None:
            return None
        prefix = self.path_to(src_id, before)
        if prefix is None:
            return None
        return tuple([*prefix] + [*action_hints])

    def choice_outcome_ref(
        self, id: NodeId, ref: refs.ChoiceOutcomeRef
    ) -> refs.ChoiceOutcomeRef:
        # We start by substituting answer ids and node ids
        match ref.value:
            case refs.AnswerId():
                if ref.value in self.hint_rev_map.answers:
                    hint = self.hint_rev_map.answers[ref.value]
                    hints = refs.Hints((hint,) if hint is not None else ())
                    ref = refs.ChoiceOutcomeRef(ref.choice, hints)
            case refs.NodeId():
                hints_raw = self.path_to(id, ref.value)
                if hints_raw is not None:
                    hints = tuple(hints_raw)
                    ref = refs.ChoiceOutcomeRef(ref.choice, refs.Hints(hints))
            case refs.Hints():
                pass
        # We make the choice implicit if we can
        if isinstance(ref.value, refs.Hints):
            tree = self.tree.goto(id)
            node = tree.node
            assert isinstance(node, Node)
            choice = node.primary_choice()
            if choice is not None and ref.choice == choice.get_origin():
                ref = refs.ChoiceOutcomeRef(None, ref.value)
        # If the choice ref is still explicit, we simplify it
        if ref.choice is not None:
            ref = refs.ChoiceOutcomeRef(
                self.choice_ref(id, ref.choice), ref.value
            )
        return ref

    def value_ref(self, id: NodeId, v: refs.ValueRef) -> refs.ValueRef:
        if isinstance(v, tuple):
            return tuple(self.value_ref(id, v) for v in v)
        else:
            return self.choice_outcome_ref(id, v)

    def choice_ref(self, id: NodeId, ref: refs.ChoiceRef) -> refs.ChoiceRef:
        hd, args = ref
        args_simp = tuple(
            a if isinstance(a, int) else self.value_ref(id, a) for a in args
        )
        return (hd, args_simp)


#####
##### Browsable traces
#####


def compute_browsable_trace(
    tree: NavigationTree[Node, Any], simplifier: _RefSimplifier | None = None
) -> fb.Trace:
    return _TraceTranslator(tree, simplifier).translate_trace()


def _check_valid_json(obj: object) -> bool:
    match obj:
        case int() | float() | str() | bool() | None:
            return True
        case dict():
            obj = cast(dict[object, object], obj)
            return all(
                isinstance(k, str) and _check_valid_json(v)
                for k, v in obj.items()
            )
        case tuple() | list():
            obj = cast(Sequence[object], obj)
            return all(_check_valid_json(v) for v in obj)
        case _:
            return False


def _value_repr[T](
    obj: T, typ: TypeAnnot[T] | NoTypeInfo
) -> fb.ValueRepr:  # fmt: skip
    short = pprint.pformat(obj, compact=True, sort_dicts=False)
    long = pprint.pformat(obj, compact=False, sort_dicts=False)
    value = fb.ValueRepr(short, long, False, None)
    if not isinstance(typ, NoTypeInfo):
        try:
            adapter = pydantic.TypeAdapter[T](typ)
            json = adapter.dump_python(obj)
            assert _check_valid_json(json)
            value.json = json
            value.json_provided = True
        except Exception:
            pass
    return value


class _TraceTranslator:

    def __init__(
        self,
        tree: NavigationTree[Node, Any],
        simplifier: _RefSimplifier | None = None,
    ) -> None:
        # We rely on the nodes in the trace being presented in
        # topological order.
        self.tree = tree
        self.simplifier = simplifier
        self.rev_map = TraceReverseMap.make(tree.tracer)
        # We do not use sets since converting a set to a list is
        # nondeterministic.
        self.choices: dict[NodeId, dict[refs.ChoiceRef, None]] = defaultdict(
            dict
        )
        self.choice_prop_ids: dict[
            tuple[NodeId, refs.ChoiceRef], fb.TraceNodePropertyId
        ] = {}
        self.action_ids: dict[
            tuple[NodeId, refs.ValueRef], fb.TraceActionId
        ] = {}

    def detect_choices(self) -> None:
        for origin in self.tree.tracer.nodes.values():
            id = origin.node
            for choice in _choices_in_node_origin(origin):
                self.choices[id][choice] = None
        for origin in self.tree.tracer.answer_ids.keys():
            id = origin.node
            for choice in _choices_in_choice_ref(origin.ref):
                self.choices[id][choice] = None

    def translate_strategy_instance(
        self,
        tree: NavigationTree[Any, Any],
        ref: refs.ChoiceRef,
        strategy: trees.StrategyInstance[Any, Any],
    ) -> fb.NodeProperty:
        wrapped = inspect.remove_wrappers(strategy.strategy)
        strategy_name = inspect.function_name(wrapped)
        assert strategy_name is not None
        args_raw = inspect.instantiated_args_dict(strategy.strategy)
        hints = typing.get_type_hints(wrapped)
        args = {a: _value_repr(v, hints[a]) for a, v in args_raw.items()}
        node_id = self.rev_map.subtrees[tree.node_id].get(ref)
        node_id_raw = node_id.id if node_id is not None else None
        return fb.Subtree("subtree", strategy_name, args, node_id_raw)

    def translate_query(
        self,
        tree: NavigationTree[Any, Any],
        choice: trees.Choice[Any],
        query: trees.Query[Any, Any],
    ) -> fb.NodeProperty:
        name = query.name()
        args = query.serialize_args()
        ref = choice.get_origin()
        answers: list[fb.Answer] = []
        origin = QueryOrigin(tree.node_id, ref)
        for v, aid in tree.tracer.answer_ids.get(origin, {}).items():
            parsed = query.parse_answer(v)
            parsed_repr = _value_repr(parsed, query.return_type())
            hint_str: tuple[()] | tuple[str] | None = None
            if self.simplifier is not None:
                if aid in self.simplifier.hint_rev_map.answers:
                    hint = self.simplifier.hint_rev_map.answers[aid]
                    if hint is None:
                        hint_str = ()
                    else:
                        hint_str = (hint.hint,)
            answers.append(fb.Answer(aid.id, hint_str, parsed_repr))
        return fb.Query("query", name, args, answers)

    def translate_choice(
        self, id: NodeId, ref: refs.ChoiceRef
    ) -> tuple[fb.Reference, fb.NodeProperty]:
        tree = self.tree.goto(id)
        choice = tree.basic_resolver().resolve_subchoice_ref(tree, ref)
        match (source := choice.source()):
            case trees.StrategyInstance():
                pty = self.translate_strategy_instance(tree, ref, source)
            case trees.Query():
                pty = self.translate_query(tree, choice, source)
            case trees.FiniteChoice():
                assert False
        ref_str = fb.Reference(dpy_pprint.choice_ref(ref), None)
        if self.simplifier is not None:
            simplified = self.simplifier.choice_ref(id, ref)
            ref_str.with_hints = dpy_pprint.choice_ref(simplified)
        return (ref_str, pty)

    def translate_action(
        self, src: NodeId, action: refs.ValueRef, dst: NodeId
    ) -> fb.Action:
        ref_str = fb.Reference(dpy_pprint.value_ref(action), None)
        hints_str: list[str] | None = None
        if self.simplifier is not None:
            hints = self.simplifier.action(src, action)
            ref_str.with_hints = dpy_pprint.value_ref(action)
            if hints is None:
                hints_str = None
            else:
                hints_str = [dpy_pprint.hint(h) for h in hints]
        successes: dict[fb.TraceNodeId, None] = {}
        answers: dict[fb.TraceAnswerId, None] = {}
        for out in _outcomes_in_value_ref(action):
            if isinstance(out.value, NodeId):
                successes[out.value.id] = None
            elif isinstance(out.value, refs.AnswerId):
                answers[out.value.id] = None
        tree = self.tree.goto(src)
        value = tree.basic_resolver().resolve_value_ref(tree, action)
        repr = _value_repr(drop_refs(value), value_type(value))
        return fb.Action(
            ref_str, hints_str, list(successes), list(answers), repr, dst.id
        )

    def translate_origin(self, id: NodeId) -> fb.NodeOrigin:
        if id == Tracer.ROOT_ID:
            return "root"
        match self.tree.tracer.nodes[id]:
            case refs.ChildOf(parent, action):
                action_id = self.action_ids[(parent, action)]
                return ("child", parent.id, action_id)
            case refs.SubtreeOf(parent, choice):
                prop_id = self.choice_prop_ids[(parent, choice)]
                return ("sub", parent.id, prop_id)

    def translate_node(self, id: NodeId) -> fb.Node:
        tree = self.tree.goto(id)
        node = cast(Node, tree.node)
        if isinstance(node, Success):
            node = cast(Success[Any], node)
            value = drop_refs(node.success)
            success = _value_repr(value, tree.return_type())
        else:
            success = None
        kind = node.type_name()
        summary = node.summary_message()
        leaf = node.leaf_node()
        label = node.get_label()
        prop_refs = {c.get_origin(): None for c in node.base_choices()}
        for cr in self.choices[id]:
            prop_refs[cr] = None
        for i, ref in enumerate(prop_refs):
            self.choice_prop_ids[(id, ref)] = i
        props = [self.translate_choice(id, r) for r in prop_refs]
        actions: list[fb.Action] = []
        for i, (a, dst) in enumerate(self.rev_map.children[id].items()):
            actions.append(self.translate_action(id, a, dst))
            self.action_ids[(id, a)] = i
        origin = self.translate_origin(id)
        return fb.Node(
            kind, success, summary, leaf, label, props, actions, origin
        )

    def translate_trace(self) -> fb.Trace:
        self.detect_choices()
        ids = [Tracer.ROOT_ID] + list(self.tree.tree.tracer.nodes.keys())
        trace = fb.Trace({id.id: self.translate_node(id) for id in ids})
        return trace


def _choices_in_node_origin(
    origin: refs.NodeOrigin,
) -> Iterable[refs.ChoiceRef]:
    match origin:
        case refs.ChildOf():
            yield from _choices_in_value_ref(origin.action)
        case refs.SubtreeOf():
            yield from _choices_in_choice_ref(origin.choice)


def _choices_in_value_ref(
    value: refs.ValueRef,
) -> Iterable[refs.ChoiceRef]:
    if isinstance(value, refs.ChoiceOutcomeRef):
        if value.choice is not None:
            yield from _choices_in_choice_ref(value.choice)
    else:
        for v in value:
            yield from _choices_in_value_ref(v)


def _choices_in_choice_ref(ref: refs.ChoiceRef) -> Iterable[refs.ChoiceRef]:
    yield ref
    _, args = ref
    for a in args:
        if not isinstance(a, int):
            yield from _choices_in_value_ref(a)


def _outcomes_in_value_ref(
    value: refs.ValueRef,
) -> Iterable[refs.ChoiceOutcomeRef]:
    if isinstance(value, refs.ChoiceOutcomeRef):
        yield value
        if value.choice is not None:
            yield from _outcomes_in_choice_ref(value.choice)
    else:
        for v in value:
            yield from _outcomes_in_value_ref(v)


def _outcomes_in_choice_ref(
    ref: refs.ChoiceRef,
) -> Iterable[refs.ChoiceOutcomeRef]:
    _, args = ref
    for a in args:
        if not isinstance(a, int):
            yield from _outcomes_in_value_ref(a)


#####
##### Evaluate Demonstrations
#####


@dataclass(frozen=True)
class SerializedQuery:
    """
    A representation of a query used as an index for the example cache.
    """

    name: str
    args: str

    @staticmethod
    def make(query: AnyQuery) -> "SerializedQuery":
        args = json.dumps(query.serialize_args())
        return SerializedQuery(query.name(), args)


class DemoHintResolver(HintResolver):
    def __init__(
        self, ctx: ExecutionContext, tracer: Tracer, demo: Demonstration
    ):
        self.tracer = tracer
        self.demo = demo
        self.queries: list[SerializedQuery] = []
        for i, q in enumerate(demo.queries):
            try:
                query = ctx.load_query(q.query, q.args)
                self.queries.append(SerializedQuery.make(query))
            except Exception as e:
                raise DemoHintResolver.InvalidQuery(i, e)
            # We try to parse all answers in anticipation, to avoid
            # an error later.
            for j, a in enumerate(q.answers):
                parsed = query.parse_answer(a.answer)
                if isinstance(parsed, ParseError):
                    raise DemoHintResolver.InvalidAnswer(i, j, parsed.error)
        self.answer_refs: dict[fb.TraceAnswerId, fb.DemoAnswerId] = {}
        self.query_used: list[bool] = [False] * len(self.queries)

    def __call__(
        self,
        query: Query[Never, object],
        origin: QueryOrigin,
        hint: refs.HintStr | None,
    ) -> str | None:
        serialized = SerializedQuery.make(query)
        for i, q in enumerate(self.queries):
            if q == serialized:
                self.query_used[i] = True
                answers = self.demo.queries[i].answers
                if not answers:
                    return None
                if hint is None:
                    answer_id = 0
                else:
                    for j, a in enumerate(answers):
                        if a.label == hint:
                            answer_id = j
                            break
                    else:
                        return None
                answer = answers[answer_id].answer
                tr_aid = self.tracer.fresh_or_cached_answer_id(answer, origin)
                self.answer_refs[tr_aid.id] = (i, answer_id)
                return answer
        return None

    def get_answer_refs(self) -> dict[fb.TraceAnswerId, fb.DemoAnswerId]:
        return self.answer_refs

    def set_reachability_diagnostics(self, feedback: fb.DemoFeedback):
        for i, used in enumerate(self.query_used):
            if not used:
                msg = "Unreachable query."
                feedback.query_diagnostics.append((i, ("warning", msg)))

    @dataclass
    class InvalidQuery(Exception):
        id: int
        exn: Exception

    @dataclass
    class InvalidAnswer(Exception):
        query_id: int
        answer_id: int
        parse_error: str


def demo_ref_resolver(hr: DemoHintResolver, tree: AnyNavTree) -> RefResolver:
    return RefResolver(
        hr.tracer, resolve_node_id=tree.goto, answer_from_hint=hr
    )


def _until_node(label: NodeLabel) -> Callable[[AnyNavTree], bool]:
    return lambda tree: tree.node.get_label() == label


def _unused_hints(diagnostics: list[fb.Diagnostic], rem: Sequence[refs.Hint]):
    if rem:
        msg = f"Unused hints: {dpy_pprint.hints(rem)}."
        diagnostics.append(("warning", msg))


def _strategy_exn(diagnostics: list[fb.Diagnostic], exn: StrategyException):
    msg = f"Exception raised in strategy:\n{exn}"
    diagnostics.append(("error", msg))


def _stuck_warning(diagnostics: list[fb.Diagnostic], exn: Stuck):
    msg = "Test is stuck."
    diagnostics.append(("warning", msg))


type SavedNodes = dict[str, AnyNavTree]


def _interpret_test_run_step(
    hint_resolver: DemoHintResolver,
    hint_rev: HintReverseMap,
    diagnostics: list[fb.Diagnostic],
    tree: AnyNavTree,
    step: demos.Run,
) -> tuple[AnyNavTree, Literal["stop", "continue"]]:
    try:
        resolver = demo_ref_resolver(hint_resolver, tree)
        resolver.info = NavigationInfo(hint_rev)
        if step.until is not None:
            resolver.interrupt = _until_node(step.until)
        try:
            tree, rem = resolver.follow_hints(tree, step.hints)
        except ReachedFailureNode as e:
            tree = e.tree
            rem = e.remaining_hints
        _unused_hints(diagnostics, rem)
        if step.until is not None:
            msg = f"Leaf node reached before '{step.until}'."
            diagnostics.append(("warning", msg))
        if step.until is None and not tree.node.leaf_node():
            msg = "The `run` command did not reach a leaf."
            diagnostics.append(("warning", msg))
        return tree, "continue"
    except Interrupted as intr:
        tree = intr.tree
        _unused_hints(diagnostics, intr.remaining_hints)
        return tree, "continue"
    except Stuck as stuck:
        tree = stuck.tree
        _stuck_warning(diagnostics, stuck)
        return tree, "stop"
    except StrategyException as e:
        _strategy_exn(diagnostics, e)
        return tree, "stop"


def _interpret_test_sub_step(
    hint_resolver: DemoHintResolver,
    hint_rev: HintReverseMap,
    diagnostics: list[fb.Diagnostic],
    tree: AnyNavTree,
    step: demos.SelectSub,
) -> tuple[AnyNavTree, Literal["stop", "continue"]]:
    resolver = demo_ref_resolver(hint_resolver, tree)
    nav_info = NavigationInfo(hint_rev)
    resolver.info = nav_info
    choice_ref = step.subchoice
    choice_ref_pretty = dpy_pprint.choice_ref(choice_ref)
    try:
        choice = resolver.resolve_subchoice_ref(tree, choice_ref)
        _unused_hints(diagnostics, nav_info.unused_hints)
        source = choice.source()
        if step.expects_query:
            if not isinstance(source, Query):
                msg = f"Not a query: {choice_ref_pretty}."
                diagnostics.append(("error", msg))
                return tree, "stop"
            query_origin = QueryOrigin(tree.node_id, choice.get_origin())
            resolver.tracer.declare_query(query_origin)
            answer = hint_resolver(source, query_origin, None)
            if answer is None:
                msg = f"Query not answered: {choice_ref_pretty}."
                diagnostics.append(("error", msg))
                return tree, "stop"
            return tree, "continue"
        else:
            if not isinstance(source, StrategyInstance):
                msg = f"Not a subtree: {choice_ref_pretty}."
                diagnostics.append(("error", msg))
                return tree, "stop"
            base_ref = choice.get_origin()
            tree = cast(AnyNavTree, tree.spawn(source.strategy, base_ref))
            return tree, "continue"
    except ReachedFailureNode as e:
        tree = e.tree
        msg = f"Failed to reach: {choice_ref_pretty}"
        diagnostics.append(("error", msg))
        return tree, "stop"
    except Stuck as stuck:
        tree = stuck.tree
        _stuck_warning(diagnostics, stuck)
        return tree, "stop"
    except StrategyException as e:
        _strategy_exn(diagnostics, e)
        return tree, "stop"
    except InvalidSubchoice as e:
        tree = e.tree
        msg = f"Invalid reference to subchoice '{e.choice_label}'."
        diagnostics.append(("error", msg))
        return tree, "stop"


def _interpret_test_step(
    hint_resolver: DemoHintResolver,
    hint_rev: HintReverseMap,
    diagnostics: list[fb.Diagnostic],
    saved: SavedNodes,
    tree: AnyNavTree,
    step: TestStep,
) -> tuple[AnyNavTree, Literal["stop", "continue"]]:
    match step:
        case demos.Run():
            return _interpret_test_run_step(
                hint_resolver, hint_rev, diagnostics, tree, step
            )
        case demos.SelectSub():
            return _interpret_test_sub_step(
                hint_resolver, hint_rev, diagnostics, tree, step
            )
        case demos.IsSuccess():
            if not isinstance(tree.node, Success):
                msg = "Success check failed."
                diagnostics.append(("error", msg))
                return tree, "stop"
            else:
                return tree, "continue"
        case demos.IsFailure():
            node = tree.node
            if not (node.leaf_node() and not isinstance(node, Success)):
                msg = "Failure check failed."
                diagnostics.append(("error", msg))
                return tree, "stop"
            else:
                return tree, "continue"
        case demos.Save():
            saved[step.name] = tree
            return tree, "continue"
        case demos.Load():
            if step.name not in saved:
                msg = f"No saved node named: '{step.name}'."
                diagnostics.append(("error", msg))
                return tree, "stop"
            return saved[step.name], "continue"


def _evaluate_test(
    root: AnyNavTree,
    hint_resolver: DemoHintResolver,
    hint_rev: HintReverseMap,
    saved: SavedNodes,
    test_str: TestCommandString,
) -> fb.TestFeedback:
    diagnostics: list[fb.Diagnostic] = []
    tree = root
    try:
        test = parse.test_command(test_str)
    except parse.ParseError:
        diagnostics = [("error", "Syntax error.")]
        return fb.TestFeedback(diagnostics, None)
    for step in test:
        tree, status = _interpret_test_step(
            hint_resolver, hint_rev, diagnostics, saved, tree, step
        )
        if status == "stop":
            break

    return fb.TestFeedback(diagnostics, tree.node_id.id)


def evaluate_demo_and_return_tree(
    demo: Demonstration, context: ExecutionContext
) -> tuple[fb.DemoFeedback, AnyNavTree | None]:
    feedback = fb.DemoFeedback(fb.Trace({}), {}, {}, [], [], [], [])
    try:
        strategy = context.find_and_instantiate_strategy(
            demo.strategy, demo.args
        )
    except Exception as e:
        msg = f"Failed to instantiate strategy:\n{e}"
        feedback.global_diagnostics.append(("error", msg))
        return feedback, None
    try:
        tree = NavigationTree.make(StrategyTree.new(strategy))
    except StrategyException as e:
        msg = f"Exception raised in strategy:\n{e}"
        feedback.global_diagnostics.append(("error", msg))
        return feedback, None
    try:
        hresolver = DemoHintResolver(context, tree.tracer, demo)
    except DemoHintResolver.InvalidQuery as e:
        msg = f"Failed to load query:\n{e.exn}"
        feedback.query_diagnostics.append((e.id, ("error", msg)))
        return feedback, tree
    except DemoHintResolver.InvalidAnswer as e:
        msg = f"Failed to parse answer:\n{e.parse_error}"
        diag = ("error", msg)
        feedback.answer_diagnostics.append(((e.query_id, e.answer_id), diag))
        return feedback, tree
    saved: SavedNodes = {}
    rm: HintReverseMap = HintReverseMap()
    for test_str in demo.tests:
        test_feedback = _evaluate_test(tree, hresolver, rm, saved, test_str)
        feedback.test_feedback.append(test_feedback)
    feedback.saved_nodes = {k: v.node_id.id for k, v in saved.items()}
    hresolver.set_reachability_diagnostics(feedback)
    simplifier = _RefSimplifier(tree, rm)
    feedback.trace = compute_browsable_trace(tree, simplifier)
    feedback.answer_refs = hresolver.get_answer_refs()
    return feedback, tree


def evaluate_demo(
    demo: Demonstration, context: ExecutionContext
) -> fb.DemoFeedback:
    feedback, _ = evaluate_demo_and_return_tree(demo, context)
    return feedback

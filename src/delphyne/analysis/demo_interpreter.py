"""
Demonstration Interpreter.
"""

import importlib
import json
import sys
import traceback
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import delphyne.core as dp
from delphyne.analysis import browsable_traces as br
from delphyne.analysis import feedback as fb
from delphyne.analysis import navigation as nv
from delphyne.core import demos as dm
from delphyne.core import refs
from delphyne.utils import typing as tp

#####
##### Execution Context and Object Loader
#####


@dataclass
class ModuleNotFound(Exception):
    """
    Raised by `ObjectLoader` when a module is not found.
    """

    module_name: str


@dataclass
class ObjectNotFound(Exception):
    """
    Raised by `ObjectLoader` when an object cannot be found.
    """

    object_name: str


@dataclass
class StrategyLoadingError(Exception):
    """
    Raised by `ObjectLoader` when a strategy instance cannot be loaded.
    """

    message: str


@dataclass(frozen=True)
class AmbiguousObjectIdentifier(Exception):
    """
    Raised when attempting to load an object with an ambiguous name.

    Attributes:
        identifier: the ambiguous identifier.
        modules: a list of modules where different objects with the same
            identifier were found
    """

    identifier: str
    modules: Sequence[str]


@dataclass(frozen=True)
class DemoExecutionContext:
    """
    Demonstration Execution Context.

    Attributes:
        strategy_dirs: A list of directories in which strategy modules
            can be found, to be added to `sys.path`.
        modules: A list of modules in which python object identifiers
            should be resolved. Modules can be part of packages and so
            their name may feature `.`.
    """

    strategy_dirs: Sequence[Path]
    modules: Sequence[str]

    def with_root(self, root: Path) -> "DemoExecutionContext":
        return DemoExecutionContext(
            strategy_dirs=[root / p for p in self.strategy_dirs],
            modules=self.modules,
        )


class ObjectLoader:
    """
    Utility class for loading Python objects.

    Demonstration and command files may refer to Python identifiers that
    need to be resolved. This is done relative to an execution context
    (`DemoExecutionContext`) that specifies a list of directories to be
    added to `sys.path`, along with a list of modules.

    An exception is raised if an object with the requested identifier
    can be found in several modules.
    """

    def __init__(
        self,
        ctx: DemoExecutionContext,
        extra_objects: dict[str, object] | None = None,
        reload: bool = True,
    ):
        """
        Attributes:
            ctx: The execution context in which to resolve Python
                identifiers.
            extra_objects: Additional objects that can be resolved by
                name (with higher precedence).
            reload: Whether to reload all modules specified in the
                execution context upon initialization. Setting this
                value to `True` makes `ObjectLoader` not thread-safe
                (also, multiple instances must not be used in an
                overlappaping way within a single thread).

        Raises:
            ModuleNotFound: a module could not be found.
        """
        self.ctx = ctx
        self.extra_objects = extra_objects if extra_objects is not None else {}
        self.modules: list[Any] = []
        with _append_path(self.ctx.strategy_dirs):
            for module_name in ctx.modules:
                try:
                    module = __import__(module_name)
                    if reload:
                        module = importlib.reload(module)
                    self.modules.append(module)
                except AttributeError:
                    raise ModuleNotFound(module_name)

    def find_object(self, name: str) -> Any:
        """
        Find an object with a given name.

        If the name is unqualified (it features no `.`), one attempts to
        find the object in every registered module in order. If the name
        is qualified, one looks at the specified registered module.

        Raises:
            ObjectNotFound: The object could not be found.
            AmbiguousObjectIdentifier: The object name is ambiguous,
                i.e. it is found in several modules.
        """
        if name in self.extra_objects:
            return self.extra_objects[name]
        comps = name.split(".")
        assert comps
        if len(comps) == 1:
            # unqualified name
            cands: list[object] = []
            modules_with_id: dict[int, list[str]] = defaultdict(list)
            for module in self.modules:
                if hasattr(module, name):
                    obj = getattr(module, name)
                    modules_with_id[id(obj)].append(module)
                    cands.append(obj)
            if len(modules_with_id) > 1:
                ambiguous = [ms[0] for ms in modules_with_id.values()]
                raise AmbiguousObjectIdentifier(name, ambiguous)
            if cands:
                return cands[0]
        else:
            # qualified name
            module = ".".join(comps[:-1])
            attr = comps[-1]
            if hasattr(module, attr):
                return getattr(module, attr)
        raise ObjectNotFound(name)

    def load_and_call_function(self, name: str, args: dict[str, Any]) -> Any:
        """
        Load and call a function by wrapping a call to `find_object`.
        """
        f = self.find_object(name)
        args = tp.parse_function_args(f, args)
        return f(**args)

    def load_strategy_instance(
        self, name: str, args: dict[str, Any]
    ) -> dp.StrategyComp[Any, Any, Any]:
        """
        Load and instantiate a strategy function with given arguments.

        Raises:
            ObjectNotFound: If the strategy function cannot be found.
            AmbiguousObjectIdentifier: If an ambiguous name is given.
            StrategyLoadingError: If the object is not a strategy function
                or if the arguments are invalid.
        """
        f = self.find_object(name)
        try:
            args = tp.parse_function_args(f, args)
            comp = f(**args)
            assert isinstance(comp, dp.StrategyComp), (
                f"Object {name} is not a strategy function."
                + " Did you forget to use the @strategy decorator?"
            )
            return cast(Any, comp)
        except Exception as e:
            raise StrategyLoadingError(str(e))

    def load_query(
        self, name: str, args: dict[str, Any]
    ) -> dp.AbstractQuery[Any]:
        """
        Load a query by name and instantiate it with given arguments.

        Raises:
            ObjectNotFound: if the query cannot be found.
            AmbiguousObjectIdentifier: if an ambiguous name is given.
            AssertionError: if the object is not a query.
        """
        obj = self.find_object(name)
        assert issubclass(obj, dp.AbstractQuery), (
            f"Object {name} is not a query type."
        )
        q = cast(type[dp.AbstractQuery[Any]], obj)
        return q.parse_instance(args)


@contextmanager
def _append_path(paths: Sequence[Path]):
    sys.path = [str(p) for p in paths] + sys.path
    yield
    sys.path = sys.path[len(paths) :]


#####
##### Demo Hint Resolver
#####


@dataclass(frozen=True)
class SerializedQuery:
    """
    A representation of a query used as an index for the example cache.
    """

    name: str
    args: str

    @staticmethod
    def make(query: dp.AbstractQuery[Any]) -> "SerializedQuery":
        args = json.dumps(query.serialize_args(), sort_keys=True)
        return SerializedQuery(query.query_name(), args)


class DemoHintResolver(nv.HintResolver):
    def __init__(self, loader: ObjectLoader, demo: dm.StrategyDemo):
        self.demo = demo
        self.queries: list[SerializedQuery] = []
        for i, q in enumerate(demo.queries):
            try:
                query = loader.load_query(q.query, q.args)
                self.queries.append(SerializedQuery.make(query))
            except Exception as e:
                raise DemoHintResolver.InvalidQuery(i, e)
            # We try to parse all answers in anticipation, to avoid
            # an error later.
            for j, a in enumerate(q.answers):
                parsed = query.parse_answer(dm.translate_answer(a))
                if isinstance(parsed, dp.ParseError):
                    raise DemoHintResolver.InvalidAnswer(i, j, parsed)
        # Used to populate `DemoFeedback.answer_refs`, which is needed
        # to implement the `Jump to Answer` action in the UI tree view.
        self.answer_refs: dict[nv.AnswerRef, fb.DemoAnswerId] = {}
        # To keep track of what queries are reachable
        self.query_used: list[bool] = [False] * len(self.queries)
        # Keeping track of implicit answers
        self.implicit: list[tuple[SerializedQuery, fb.ImplicitAnswer]] = []

    def __call__(
        self,
        query: dp.AttachedQuery[Any],
        hint: refs.HintValue | None,
        implicit_answer: Callable[[], str] | None,
    ) -> refs.Answer | None:
        serialized = SerializedQuery.make(query.query)
        # Look at the query demonstrations included in the demo
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
                demo_answer = answers[answer_id]
                answer = dm.translate_answer(demo_answer)
                self.answer_refs[(query.ref, answer)] = (i, answer_id)
                return answer
        # We only look at implicit answers if no hint is provided.
        if hint is not None:
            return None
        # Look at previous implicit answers
        for serialized_key, a in self.implicit:
            if serialized == serialized_key:
                return refs.Answer(None, a.answer)
        # Maybe we try adding an implicit answer
        if implicit_answer:
            try:
                ans_text = implicit_answer()
            except Exception as e:
                raise dp.StrategyException(e)
            implicit = fb.ImplicitAnswer(
                query.query.query_name(),
                query.query.serialize_args(),
                ans_text,
            )
            self.implicit.append((serialized, implicit))
            return refs.Answer(None, ans_text)
        return None

    def get_answer_refs(self) -> dict[nv.AnswerRef, fb.DemoAnswerId]:
        return self.answer_refs

    def get_implicit_answers(self) -> list[fb.ImplicitAnswer]:
        return [a for _, a in self.implicit]

    def set_reachability_diagnostics(self, feedback: fb.StrategyDemoFeedback):
        for i, used in enumerate(self.query_used):
            if not used:
                msg = "Unreachable query."
                feedback.query_diagnostics.append((i, ("warning", msg)))

    def navigator(self) -> nv.Navigator:
        return nv.Navigator(self)

    @dataclass
    class InvalidQuery(Exception):
        id: int
        exn: Exception

    @dataclass
    class InvalidAnswer(Exception):
        query_id: int
        answer_id: int
        parse_error: dp.ParseError


#####
##### Test Interpreter
#####


type SavedNodes = dict[str, dp.AnyTree]
"""
Nodes saved using the `save` test instruction.
"""


def _unused_hints(diagnostics: list[fb.Diagnostic], rem: Sequence[refs.Hint]):
    if rem:
        msg = f"Unused hints: {dp.pprint.hints(rem)}."
        diagnostics.append(("warning", msg))


def _strategy_exn(diagnostics: list[fb.Diagnostic], exn: dp.StrategyException):
    details = f"{repr(exn.exn)}\n\n{traceback.format_exc()}"
    msg = f"Exception raised in strategy:\n\n{details}"
    diagnostics.append(("error", msg))


def _handle_navigation_error_or_reraise(
    exn: Exception, diagnostics: list[fb.Diagnostic], test_step: dm.TestStep
) -> dp.AnyTree | None:
    """
    Handle navigation errors or warnings that require stopping
    navigation. Reraise if the exception is not known. Return the node
    at which the error arose if relevant.
    """
    if isinstance(exn, nv.Stuck):
        msg = "Test is stuck."
        diagnostics.append(("warning", msg))
        return exn.tree
    elif isinstance(exn, dp.StrategyException):
        _strategy_exn(diagnostics, exn)
        return None
    elif isinstance(exn, dp.NavigationError):
        details = f"{repr(exn.message)}\n\n{traceback.format_exc()}"
        msg = f"Navigation error:\n\n{details}"
        diagnostics.append(("error", msg))
        return None
    elif isinstance(exn, nv.ReachedFailureNode):
        step_str = dp.pprint.test_step(test_step)
        msg = f"Reached failure node while executing: {step_str}."
        diagnostics.append(("error", msg))
        return exn.tree
    elif isinstance(exn, nv.InvalidSpace):
        name = dp.pprint.space_name(exn.space_name)
        msg = f"Invalid reference to space: {name}."
        diagnostics.append(("error", msg))
        return exn.tree
    elif isinstance(exn, nv.NoPrimarySpace):
        msg = f"Node {exn.tree.node.effect_name()} has no primary space."
        diagnostics.append(("error", msg))
        return exn.tree
    raise exn


def _interpret_test_run_step(
    hint_resolver: DemoHintResolver,
    hint_rev: nv.HintReverseMap,
    diagnostics: list[fb.Diagnostic],
    tree: dp.AnyTree,
    tracer: dp.Tracer,
    step: dm.Run,
) -> tuple[dp.AnyTree, Literal["stop", "continue"]]:
    try:
        encountered = nv.EncounteredTags()
        navigator = hint_resolver.navigator()
        navigator.info = nv.NavigationInfo(hint_rev)
        navigator.tracer = tracer
        try:
            tree, rem = navigator.follow_hints(
                tree, step.hints, step.until, encountered
            )
        except nv.ReachedFailureNode as e:
            tree = e.tree
            rem = e.remaining_hints
        _unused_hints(diagnostics, rem)
        if step.until is not None:
            until_pp = dp.pprint.node_selector(step.until)
            msg = f"Leaf node reached before '{until_pp}'."
            diagnostics.append(("warning", msg))
        if step.until is None and not tree.node.leaf_node():
            msg = "The `run` command did not reach a leaf."
            diagnostics.append(("warning", msg))
        return tree, "continue"
    except nv.MatchedSelector as intr:
        tree = intr.tree
        _unused_hints(diagnostics, intr.remaining_hints)
        return tree, "continue"
    except Exception as e:
        error_tree = _handle_navigation_error_or_reraise(e, diagnostics, step)
        return error_tree or tree, "stop"


def _interpret_test_select_step(
    hint_resolver: DemoHintResolver,
    hint_rev: nv.HintReverseMap,
    diagnostics: list[fb.Diagnostic],
    tree: dp.AnyTree,
    tracer: dp.Tracer,
    step: dm.SelectSpace,
) -> tuple[dp.AnyTree, Literal["stop", "continue"]]:
    navigator = hint_resolver.navigator()
    nav_info = nv.NavigationInfo(hint_rev)
    navigator.info = nav_info
    navigator.tracer = tracer
    space_ref_pretty = dp.pprint.space_ref(step.space)
    try:
        space = navigator.resolve_space_ref(tree, step.space)
        source = space.source()
        _unused_hints(diagnostics, nav_info.unused_hints)
        if step.expects_query:
            if not isinstance(source, dp.AttachedQuery):
                msg = f"Not a query: {space_ref_pretty}."
                diagnostics.append(("error", msg))
                return tree, "stop"
            tracer.trace_query(source.ref)
            answer = hint_resolver(source, None, None)
            if answer is None:
                msg = f"Query not answered: {space_ref_pretty}."
                diagnostics.append(("error", msg))
                return tree, "stop"
            tracer.trace_answer(source.ref, answer)
            hint_rev.answers[(source.ref, answer)] = None
            return tree, "continue"
        else:
            if not isinstance(source, dp.NestedTree):
                msg = f"Not a nested tree: {space_ref_pretty}."
                diagnostics.append(("error", msg))
                return tree, "stop"
            tree = source.spawn_tree()
            return tree, "continue"
    except Exception as e:
        error_tree = _handle_navigation_error_or_reraise(e, diagnostics, step)
        return error_tree or tree, "stop"


def _interpret_test_goto_child_step(
    hint_resolver: DemoHintResolver,
    hint_rev: nv.HintReverseMap,
    diagnostics: list[fb.Diagnostic],
    tree: dp.AnyTree,
    tracer: dp.Tracer,
    step: dm.GoToChild,
) -> tuple[dp.AnyTree, Literal["stop", "continue"]]:
    navigator = hint_resolver.navigator()
    nav_info = nv.NavigationInfo(hint_rev)
    navigator.info = nav_info
    navigator.tracer = tracer
    try:
        action = navigator.resolve_value_ref(tree, step.action)
        _unused_hints(diagnostics, nav_info.unused_hints)
        tree = tree.child(action)
        return tree, "continue"
    except Exception as e:
        error_tree = _handle_navigation_error_or_reraise(e, diagnostics, step)
        return error_tree or tree, "stop"


def _interpret_test_step(
    hint_resolver: DemoHintResolver,
    hint_rev: nv.HintReverseMap,
    diagnostics: list[fb.Diagnostic],
    saved: SavedNodes,
    tree: dp.AnyTree,
    tracer: dp.Tracer,
    step: dm.TestStep,
) -> tuple[dp.AnyTree, Literal["stop", "continue"]]:
    match step:
        case dm.Run():
            return _interpret_test_run_step(
                hint_resolver, hint_rev, diagnostics, tree, tracer, step
            )
        case dm.SelectSpace():
            return _interpret_test_select_step(
                hint_resolver, hint_rev, diagnostics, tree, tracer, step
            )
        case dm.GoToChild():
            return _interpret_test_goto_child_step(
                hint_resolver, hint_rev, diagnostics, tree, tracer, step
            )
        case dm.IsSuccess():
            if not isinstance(tree.node, dp.Success):
                msg = "Success check failed."
                diagnostics.append(("error", msg))
                return tree, "stop"
            else:
                return tree, "continue"
        case dm.IsFailure():
            node = tree.node
            if not (node.leaf_node() and not isinstance(node, dp.Success)):
                msg = "Failure check failed."
                diagnostics.append(("error", msg))
                return tree, "stop"
            else:
                return tree, "continue"
        case dm.Save():
            saved[step.name] = tree
            return tree, "continue"
        case dm.Load():
            if step.name not in saved:
                msg = f"No saved node named: '{step.name}'."
                diagnostics.append(("error", msg))
                return tree, "stop"
            return saved[step.name], "continue"


def _evaluate_test(
    root: dp.AnyTree,
    tracer: dp.Tracer,
    hint_resolver: DemoHintResolver,
    hint_rev: nv.HintReverseMap,
    saved: SavedNodes,
    test_str: dm.TestCommandString,
) -> fb.TestFeedback:
    diagnostics: list[fb.Diagnostic] = []
    tree = root
    try:
        test = dp.parse.test_command(test_str)
    except dp.parse.ParseError:
        diagnostics = [("error", "Syntax error.")]
        return fb.TestFeedback(diagnostics, None)
    for step in test:
        tree, status = _interpret_test_step(
            hint_resolver, hint_rev, diagnostics, saved, tree, tracer, step
        )
        if status == "stop":
            break
    ref = tracer.trace.convert_global_node_path(tree.ref)
    return fb.TestFeedback(diagnostics, ref.id)


#####
##### Strategy Demo Evaluation
#####


def evaluate_strategy_demo_and_return_trace(
    demo: dm.StrategyDemo,
    context: DemoExecutionContext,
    extra_objects: dict[str, object],
) -> tuple[fb.StrategyDemoFeedback, dp.Trace | None]:
    feedback = fb.StrategyDemoFeedback(
        kind="strategy",
        trace=fb.Trace({}),
        answer_refs={},
        saved_nodes={},
        test_feedback=[],
        global_diagnostics=[],
        query_diagnostics=[],
        answer_diagnostics=[],
        implicit_answers=[],
    )
    try:
        loader = ObjectLoader(context, extra_objects)
        strategy = loader.load_strategy_instance(demo.strategy, demo.args)
    except Exception as e:
        msg = f"Failed to instantiate strategy:\n{e}"
        feedback.global_diagnostics.append(("error", msg))
        return feedback, None
    try:
        cache: dp.TreeCache = {}
        tracer = dp.Tracer()
        trace = tracer.trace
        monitor = dp.TreeMonitor(cache=cache, hooks=[dp.tracer_hook(tracer)])
        tree = dp.reify(strategy, monitor)
    except dp.StrategyException as e:
        _strategy_exn(feedback.global_diagnostics, e)
        return feedback, None
    try:
        hresolver = DemoHintResolver(loader, demo)
    except DemoHintResolver.InvalidQuery as e:
        msg = f"Failed to load query:\n{e.exn}"
        feedback.query_diagnostics.append((e.id, ("error", msg)))
        return feedback, trace
    except DemoHintResolver.InvalidAnswer as e:
        msg = f"Failed to parse answer:\n{e.parse_error}"
        diag = ("error", msg)
        feedback.answer_diagnostics.append(((e.query_id, e.answer_id), diag))
        return feedback, trace
    saved: SavedNodes = {}
    rm: nv.HintReverseMap = nv.HintReverseMap()
    for test_str in demo.tests:
        test_feedback = _evaluate_test(
            tree, tracer, hresolver, rm, saved, test_str
        )
        feedback.test_feedback.append(test_feedback)
    feedback.saved_nodes = {
        k: tracer.trace.convert_global_node_path(v.ref).id
        for k, v in saved.items()
    }
    trace.check_consistency()
    hresolver.set_reachability_diagnostics(feedback)
    simplifier = br.RefSimplifier(cache, rm)
    feedback.trace = br.compute_browsable_trace(trace, cache, simplifier)
    feedback.answer_refs = {
        trace.convert_answer_ref(k).id: v
        for k, v in hresolver.get_answer_refs().items()
    }
    feedback.implicit_answers = hresolver.get_implicit_answers()
    return feedback, trace


#####
##### Standalone Query Evaluation
#####


def evaluate_standalone_query_demo(
    demo: dm.QueryDemo,
    context: DemoExecutionContext,
    extra_objects: dict[str, object],
) -> fb.QueryDemoFeedback:
    feedback = fb.QueryDemoFeedback(
        kind="query", diagnostics=[], answer_diagnostics=[]
    )
    try:
        loader = ObjectLoader(context, extra_objects)
        query = loader.load_query(demo.query, demo.args)
    except Exception as e:
        msg = f"Failed to instantiate query:\n{e}"
        feedback.diagnostics.append(("error", msg))
        return feedback
    # We just check that all the answers parse
    for i, a in enumerate(demo.answers):
        try:
            elt = query.parse_answer(dm.translate_answer(a))
            if isinstance(elt, dp.ParseError):
                diag = ("error", f"Parse error: {str(elt)}")
                feedback.answer_diagnostics.append((i, diag))
        except Exception as e:
            diag = ("error", f"Internal parser error: {str(e)}")
            feedback.answer_diagnostics.append((i, diag))
    return feedback


#####
##### Main EntryPoint
#####


def evaluate_demo(
    demo: dm.Demo,
    context: DemoExecutionContext,
    extra_objects: dict[str, object],
) -> fb.DemoFeedback:
    """
    Evaluate a query or strategy demonstration.

    This is the main entrypoint of the demonstration interpreter.

    Attributes:
        demo: The demonstration to evaluate.
        context: The execution context in which to resolve Python
            identifiers.
        extra_objects: Additional objects that can be resolved by name
            (with higher precedence).

    Returns:
        A feedback object containing the results of the evaluation.

    !!! warning
        This function creates an `ObjectLoader` internally and is
        therefore not thread-safe.
    """
    if isinstance(demo, dm.StrategyDemo):
        feedback, _ = evaluate_strategy_demo_and_return_trace(
            demo, context, extra_objects
        )
        return feedback
    else:
        return evaluate_standalone_query_demo(demo, context, extra_objects)

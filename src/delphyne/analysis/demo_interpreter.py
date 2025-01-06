"""
Demonstration Interpreter.
"""

import importlib
import json
import sys
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import delphyne.core as dp
from delphyne.analysis import feedback as fb
from delphyne.analysis import navigation as nv
from delphyne.core import demos as dm
from delphyne.core import refs
from delphyne.utils import typing as tp

#####
##### Environment Execution Context
#####


@dataclass
class ModuleNotFound(Exception):
    module_name: str


@dataclass
class ObjectNotFound(Exception):
    object_name: str


@dataclass
class StrategyLoadingError(Exception):
    message: str


@dataclass(frozen=True)
class DemoExecutionContext:
    strategy_dirs: Sequence[Path]
    modules: Sequence[str]


class ObjectLoader:
    def __init__(self, ctx: DemoExecutionContext, reload: bool = True):
        """
        Raises `ModuleNotFound`.
        """
        self.ctx = ctx
        self.modules: list[Any] = []
        with _append_path(self.ctx.strategy_dirs):
            for module_name in self.modules:
                try:
                    module = __import__(module_name)
                    if reload:
                        module = importlib.reload(module)
                    self.modules.append(module)
                except AttributeError:
                    raise ModuleNotFound(module_name)

    def find_object(self, name: str) -> Any:
        for module in self.modules:
            if hasattr(module, name):
                return getattr(module, name)
        raise ObjectNotFound(name)

    def load_strategy_instance(
        self, name: str, args: dict[str, Any]
    ) -> dp.StrategyComp[Any, Any, Any]:
        f = self.find_object(name)
        try:
            args = tp.parse_function_args(f, args)
            return f(**args)
        except Exception as e:
            raise StrategyLoadingError(str(e))

    def load_query(
        self, name: str, args: dict[str, Any]
    ) -> dp.AbstractQuery[Any]:
        q = cast(type[dp.AbstractQuery[Any]], self.find_object(name))
        return q.parse(args)


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
        args = json.dumps(query.serialize_args())
        return SerializedQuery(query.name(), args)


class DemoHintResolver(nv.HintResolver):
    def __init__(self, loader: ObjectLoader, demo: dm.Demonstration):
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
                mode = query.modes().get(a.mode)
                if mode is None:
                    raise DemoHintResolver.InvalidAnswer(i, j, "Invalid mode.")
                parsed = mode.parse(query.answer_type(), a.answer)
                if isinstance(parsed, dp.ParseError):
                    raise DemoHintResolver.InvalidAnswer(i, j, parsed.error)
        # Used to populate `DemoFeedback.answer_refs`, which is needed
        # to implement the `Jump to Answer` action in the UI tree view.
        self.answer_refs: dict[nv.AnswerRef, fb.DemoAnswerId] = {}
        # To keep track of what queries are reachable
        self.query_used: list[bool] = [False] * len(self.queries)

    def __call__(
        self,
        query: dp.AttachedQuery[Any],
        hint: refs.HintValue | None,
    ) -> refs.Answer | None:
        serialized = SerializedQuery.make(query.query)
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
                answer = refs.Answer(demo_answer.mode, demo_answer.answer)
                self.answer_refs[(query.ref, answer)] = (i, answer_id)
                return answer
        return None

    def get_answer_refs(self) -> dict[nv.AnswerRef, fb.DemoAnswerId]:
        return self.answer_refs

    def set_reachability_diagnostics(self, feedback: fb.DemoFeedback):
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
        parse_error: str


#####
##### Interpreter
#####


def _until_node(label: dm.NodeTag) -> Callable[[nv.NavTree], bool]:
    return lambda tree: label in tree.node.get_tags()


def _unused_hints(diagnostics: list[fb.Diagnostic], rem: Sequence[refs.Hint]):
    if rem:
        msg = f"Unused hints: {dp.pprint.hints(rem)}."
        diagnostics.append(("warning", msg))


def _strategy_exn(diagnostics: list[fb.Diagnostic], exn: dp.StrategyException):
    msg = f"Exception raised in strategy:\n{exn}"
    diagnostics.append(("error", msg))


def _stuck_warning(diagnostics: list[fb.Diagnostic], exn: nv.Stuck):
    msg = "Test is stuck."
    diagnostics.append(("warning", msg))


type SavedNodes = dict[str, nv.NavTree]


def _interpret_test_run_step(
    hint_resolver: DemoHintResolver,
    hint_rev: nv.HintReverseMap,
    diagnostics: list[fb.Diagnostic],
    tree: nv.NavTree,
    step: dm.Run,
) -> tuple[nv.NavTree, Literal["stop", "continue"]]:
    try:
        navigator = hint_resolver.navigator()
        navigator.info = nv.NavigationInfo(hint_rev)
        if step.until is not None:
            navigator.interrupt = _until_node(step.until)
        try:
            tree, rem = navigator.follow_hints(tree, step.hints.hints)
        except nv.ReachedFailureNode as e:
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
    except nv.Interrupted as intr:
        tree = intr.tree
        _unused_hints(diagnostics, intr.remaining_hints)
        return tree, "continue"
    except nv.Stuck as stuck:
        tree = stuck.tree
        _stuck_warning(diagnostics, stuck)
        return tree, "stop"
    except dp.StrategyException as e:
        _strategy_exn(diagnostics, e)
        return tree, "stop"

"""
The `Compute` Effect
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Never, cast, override

import yaml

import delphyne.core.inspect as insp
import delphyne.core_and_base as dp
import delphyne.utils.typing as ty
from delphyne.core_and_base import PolicyEnv, spawn_node
from delphyne.stdlib.queries import create_prompt, json_object_digest
from delphyne.utils.yaml import dump_yaml


@dataclass
class __Computation__(dp.AbstractQuery[object]):
    """
    A special query that represents a computation.

    Returns a parsed JSON value.

    Attributes:
        fun: Name of the function to call.
        args: Arguments to pass to the function, as a dictionary.
    """

    fun: str
    args: dict[str, Any]

    @override
    def generate_prompt(
        self,
        *,
        kind: str,
        mode: dp.AnswerMode,
        params: dict[str, object],
        extra_args: dict[str, object] | None = None,
        env: dp.AbstractTemplatesManager | None = None,
    ) -> str:
        # The prompt is never sent to LLMs but it is important that it
        # uniquely identifies the query instance since it is used for
        # caching.
        return dump_yaml(Any, self.__dict__)

    @override
    def query_modes(self):
        return [None]

    @override
    def answer_type(self):
        return object

    @override
    def parse_answer(self, answer: dp.Answer) -> object:
        # We expect answers to feature a YAML serialization of the
        # computation result, as a string. Also, we do not return
        # `ParseError` when parsing fails since this is definitely a
        # user error and not an LLM error.
        # TODO: transition to using structured answers?
        assert isinstance(answer.content, str)
        return yaml.safe_load(answer.content)


@dataclass
class Compute(dp.Node):
    """
    The standard `Compute` effect.

    For efficiency and replicability reasons, strategies must not directly
    perform expensive and possibly non-replicable computations. For example,
    a strategy should not directly call an SMT solver since:

    - The call may be expensive, and stratgy computations are replayed from
    scratch every time a child is computed in the corresponding tree (see
    documentation for `reify`).
    - SMT solvers using wall-time timeouts may return different results when
    called repeatedly on the same input.

    The `Compute` effect allows performing an expensive and possibly
    non-deterministic computation by issuing a special `__Computation__`
    query that specifies the computation to be performed. Such a query is
    not answered by an LLM, but by _actually_ performing the described
    computation. Special support is available in the demonstration
    interpreter in the form of *implicit answers*, allowing
    `__Computation__` queries to be automatically answered when running
    tests. Generated answers can be hardcoded in demonstrations **after**
    the fact via proper editor support (i.e. using the `Add Implicit
    Answers` code action from Delphyne's VSCode extension).
    """

    query: dp.TransparentQuery[Any]
    override_args: dp.FromPolicy[dict[str, Any] | None] | None

    # Takes as an argument a dict of overriden args
    _comp: Callable[[dict[str, Any] | None], Any]
    _ret_type: ty.TypeAnnot[Any]

    def navigate(self) -> dp.Navigation:
        return (yield self.query)

    def run_computation(
        self, *, overriden_args: dict[str, Any] | None = None
    ) -> str:
        ret = self._comp(overriden_args)
        serialized = dump_yaml(self._ret_type, ret)
        return serialized

    def run_computation_with_cache(
        self,
        *,
        cache: dp.LLMCache | None,
        overriden_args: dict[str, Any] | None = None,
    ) -> str:
        """
        Run the computation using a fake oracle so that the LLM caching
        mechanism can be reused.
        """
        chat = create_prompt(
            self.query.attached.query,
            examples=[],
            params={},
            mode=None,
            env=None,
        )
        req = dp.LLMRequest(chat=chat, num_completions=1, options={})
        model = ComputationOracle(
            partial(self.run_computation, overriden_args=overriden_args)
        )
        resp = model.send_request(req, cache)
        assert len(resp.outputs) == 1
        answer = resp.outputs[0].content
        assert isinstance(answer, str)
        return answer


def compute[**A, T, P](
    f: Callable[A, T],
    *,
    override_args: Callable[[P], dict[str, Any] | None] | None = None,
    inner_policy_type: type[P] | None = None,
) -> Callable[A, dp.Strategy[Compute, P, T]]:
    """
    Triggering function for the `Compute` effect.

    Arguments:
        f: Function performing an expensive computation. It must feature
            type annotations and its arguments must be
            JSON-serializable. It does not need to be deterministic.
        override_args: Mapping from the ambient inner policy to a
            dictionary overriding some of the function arguments. These
            overrides are only visible on policy side and do not affect
            the underlying `__Compute__` query. This is particularly
            useful to override timeout parameters in policies.
        inner_policy_type: Ambient inner policy type. This information
            is not used at runtime but it can be provided to help type
            inference when necessary.
    """

    def wrapped(
        *args: A.args, **kwargs: A.kwargs
    ) -> dp.Strategy[Compute, object, T]:
        def comp(overriden_args: dict[str, Any] | None) -> Any:
            return f(*args, **(kwargs | (overriden_args or {})))  # type: ignore

        ret_type = insp.function_return_type(f)
        assert not isinstance(ret_type, ty.NoTypeInfo)
        fun_args = insp.function_args_dict(f, args, kwargs)
        fun = insp.function_name(f)
        assert fun is not None
        query = dp.TransparentQuery.build(__Computation__(fun, fun_args))
        unparsed = yield spawn_node(
            Compute,
            query=query,
            override_args=override_args,
            _comp=comp,
            _ret_type=ret_type,
        )
        ret = ty.pydantic_load(ret_type, unparsed)
        return cast(T, ret)

    return wrapped


@dataclass
class ComputationOracle(dp.LLM):
    """
    A fake LLM that performs computations.

    Using such an oracle allows reusing the LLM request caching
    infrasrtructure.
    """

    computation: Callable[[], str]

    @override
    def add_model_defaults(self, req: dp.LLMRequest) -> dp.LLMRequest:
        return req

    @override
    def _send_final_request(self, req: dp.LLMRequest) -> dp.LLMResponse:
        res = self.computation()
        return dp.LLMResponse(
            outputs=[
                dp.LLMOutput(content=res, tool_calls=[], finish_reason="stop")
            ],
            budget=dp.Budget({}),
            log_items=[],
            model_name=None,
            usage_info=None,
        )


@dp.contextual_tree_transformer
def elim_compute(
    env: PolicyEnv,
    policy: Any,
    *,
    force_bypass_cache: bool = False,
    log_computations: dp.LogLevel | None = None,
    log_long_computations: tuple[dp.LogLevel, float] | None = None,
    override_args: dict[str, Any] | None = None,
) -> dp.PureTreeTransformerFn[Compute, Never]:
    """
    Eliminate the `Compute` effect by performing the computation when
    computing tree children (making the `child` function possibly
    nondeterministic).

    Arguments:
        force_bypass_cache: If set to `True`, do not cache computation
            results, even if a cache is available in the global policy
            environment.
        log_computations: If set, log every performed computation at the
            given severity level.
        log_long_computations: If set, log every computation taking more
            than the given number of seconds at the given severity
            level. When set to `None`, this setting can be overriden by
            `PolicyEnv.log_long_computations`.
        override_args: Overriden argument values for all computations.
            This is particularly useful for setting global timeouts.
            Argument values specified this way have lower precedence
            than those specified with the `override_args` argument of
            `compute`.
    """

    if log_long_computations is None:
        log_long_computations = env.log_long_computations

    def transform[N: dp.Node, P, T](
        tree: dp.Tree[Compute | N, P, T],
    ) -> dp.Tree[N, P, T]:
        if isinstance(tree.node, Compute):
            cache = None
            if not force_bypass_cache:
                cache = env.cache
            query = tree.node.query.attached.query
            assert isinstance(query, __Computation__), str(type(query))
            digest = json_object_digest(ty.pydantic_dump(Any, query))
            if log_computations:
                env.log(
                    log_computations,
                    "computation_started",
                    {"hash": digest, "details": query},
                    loc=tree,
                )
            overriden: dict[str, Any] = override_args or {}
            if tree.node.override_args is not None:
                overriden_local = tree.node.override_args(policy)
                if overriden_local is not None:
                    overriden = overriden | overriden_local
            start = time.time()
            answer = tree.node.run_computation_with_cache(
                cache=cache, overriden_args=overriden
            )
            _elapsed = time.time() - start
            if log_computations:
                env.log(
                    log_computations,
                    "computation_finished",
                    {"hash": digest, "elapsed": _elapsed, "result": answer},
                    loc=tree,
                )
            if log_long_computations and _elapsed > log_long_computations[1]:
                env.log(
                    log_long_computations[0],
                    "long_computation",
                    {
                        "hash": digest,
                        "details": query,
                        "elapsed": _elapsed,
                        "result": answer,
                    },
                    loc=tree,
                )
            tracked = tree.node.query.attached.parse_answer(
                dp.Answer(None, answer)
            )
            assert not isinstance(tracked, dp.ParseError)
            return transform(tree.child(tracked))

        return tree.transform(tree.node, transform)

    return transform


def generate_implicit_answer(
    tree: dp.AnyTree, query: dp.AttachedQuery[Any]
) -> tuple[dp.ImplicitAnswerCategory, dp.Answer] | None:
    if isinstance(tree.node, Compute):
        try:
            answer = tree.node.run_computation()
        except Exception as e:
            raise dp.StrategyException(e)
        return ("computations", dp.Answer(None, answer))

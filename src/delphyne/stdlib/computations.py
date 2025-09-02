"""
The `Compute` Effect
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Never, cast, override

import yaml
import yaml.parser

import delphyne.core.inspect as insp
import delphyne.core_and_base as dp
import delphyne.utils.typing as ty
from delphyne.core_and_base import PolicyEnv, spawn_node
from delphyne.stdlib.queries import create_prompt
from delphyne.utils.yaml import dump_yaml


@dataclass
class __Computation__(dp.AbstractQuery[object]):
    """
    A special query that represents a computation.

    Returns a parsed JSON result.

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
    ):
        return dump_yaml(Any, self.__dict__)

    @override
    def query_modes(self):
        return [None]

    @override
    def answer_type(self):
        return object

    @override
    def parse_answer(self, answer: dp.Answer) -> object | dp.ParseError:
        try:
            assert isinstance(answer.content, str)
            return yaml.safe_load(answer.content)
        except yaml.parser.ParserError as e:
            return dp.ParseError(description=str(e))


@dataclass
class Compute(dp.ComputationNode):
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
    _comp: Callable[[], Any]
    _ret_type: ty.TypeAnnot[Any]

    def navigate(self) -> dp.Navigation:
        return (yield self.query)

    def run_computation(self) -> str:
        ret = self._comp()
        serialized = dump_yaml(self._ret_type, ret)
        return serialized

    def run_computation_with_cache(self, cache: dp.LLMCache | None) -> str:
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
        model = ComputationOracle(self.run_computation)
        resp = model.send_request(req, cache)
        assert len(resp.outputs) == 1
        answer = resp.outputs[0].content
        assert isinstance(answer, str)
        return answer


def compute[**P, T](
    f: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> dp.Strategy[Compute, object, T]:
    """
    Triggering function for the `Compute` effect.

    Arguments:
        f: Function performing an expensive computation. It must feature
            type annotations and its arguments must be
            JSON-serializable. It does not need to be deterministic.
        *args: Positional arguments to pass to `f`.
        **kwargs: Keyword arguments to pass to `f`.
    """
    comp = partial(f, *args, **kwargs)
    ret_type = insp.function_return_type(f)
    assert not isinstance(ret_type, ty.NoTypeInfo)
    fun_args = insp.function_args_dict(f, args, kwargs)
    fun = insp.function_name(f)
    assert fun is not None
    query = dp.TransparentQuery.build(__Computation__(fun, fun_args))
    unparsed = yield spawn_node(
        Compute, query=query, _comp=comp, _ret_type=ret_type
    )
    ret = ty.pydantic_load(ret_type, unparsed)
    return cast(T, ret)


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
    force_bypass_cache: bool = False,
) -> dp.PureTreeTransformerFn[Compute, Never]:
    """
    Eliminate the `Compute` effect by performing the computation when
    computing tree children (making the `child` function possibly
    nondeterministic).

    Arguments:
        force_bypass_cache: if set to `True`, do not cache computation
            results, even if a cache is available in the global policy
            environment.
    """

    def transform[N: dp.Node, P, T](
        tree: dp.Tree[Compute | N, P, T],
    ) -> dp.Tree[N, P, T]:
        if isinstance(tree.node, Compute):
            cache = None
            if not force_bypass_cache:
                cache = env.cache
            answer = tree.node.run_computation_with_cache(cache)
            tracked = tree.node.query.attached.parse_answer(
                dp.Answer(None, answer)
            )
            assert not isinstance(tracked, dp.ParseError)
            return transform(tree.child(tracked))

        return tree.transform(tree.node, transform)

    return transform

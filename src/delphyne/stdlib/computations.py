"""
Computation Nodes
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Never, cast

import yaml
import yaml.parser

import delphyne.core as dp
import delphyne.core.inspect as insp
import delphyne.stdlib.policies as pol
import delphyne.utils.typing as ty
from delphyne.stdlib.nodes import spawn_node
from delphyne.utils.yaml import dump_yaml


@dataclass
class __Computation__(dp.AbstractQuery[object]):
    """
    A special query that represents a cached computation.

    Returns a parsed JSON result.
    """

    fun: str
    args: dict[str, Any]

    def generate_prompt(
        self,
        kind: str,
        mode: dp.AnswerModeName,
        params: dict[str, object],
        env: dp.TemplatesManager | None = None,
    ):
        return ""

    def answer_type(self):
        return object

    def parse_answer(self, answer: dp.Answer) -> object | dp.ParseError:
        try:
            assert isinstance(answer.content, str)
            return yaml.safe_load(answer.content)
        except yaml.parser.ParserError as e:
            return dp.ParseError(description=str(e))


@dataclass
class Computation(dp.ComputationNode):
    query: dp.TransparentQuery[Any]
    _comp: Callable[[], Any]
    _ret_type: ty.TypeAnnot[Any]

    def navigate(self) -> dp.Navigation:
        return (yield self.query)

    def run_computation(self) -> str:
        ret = self._comp()
        serialized = dump_yaml(self._ret_type, ret)
        return serialized


def compute[**P, T](
    f: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> dp.Strategy[Computation, object, T]:
    comp = partial(f, *args, **kwargs)
    ret_type = insp.function_return_type(f)
    assert not isinstance(ret_type, ty.NoTypeInfo)
    fun_args = insp.function_args_dict(f, args, kwargs)
    fun = insp.function_name(f)
    assert fun is not None
    query = dp.TransparentQuery.build(__Computation__(fun, fun_args))
    unparsed = yield spawn_node(
        Computation, query=query, _comp=comp, _ret_type=ret_type
    )
    ret = ty.pydantic_load(ret_type, unparsed)
    return cast(T, ret)


@pol.contextual_tree_transformer
def elim_compute(
    env: dp.PolicyEnv,
    policy: Any,
    force_bypass_cache: bool = False,
) -> pol.PureTreeTransformerFn[Computation, Never]:
    def transform[N: dp.Node, P, T](
        tree: dp.Tree[Computation | N, P, T],
    ) -> dp.Tree[N, P, T]:
        if isinstance(tree.node, Computation):
            answer = tree.node.run_computation()
            tracked = tree.node.query.attached.parse_answer(
                dp.Answer(None, answer)
            )
            assert not isinstance(tracked, dp.ParseError)
            return transform(tree.child(tracked))

        return tree.transform(tree.node, transform)

    return transform

"""
Computation Nodes
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import yaml
import yaml.parser

import delphyne.core as dp
import delphyne.core.inspect as insp
import delphyne.utils.typing as ty
from delphyne.stdlib.nodes import spawn_node


@dataclass
class __Computation__(dp.AbstractQuery[object]):
    """
    A special query that represents a cached computation.

    Returns a parsed JSON result.
    """

    fun: str
    args: dict[str, Any]

    def system_prompt(
        self,
        mode: dp.AnswerModeName,
        params: dict[str, object],
        env: dp.TemplatesManager | None = None,
    ):
        return ""

    def instance_prompt(
        self,
        mode: dp.AnswerModeName,
        params: dict[str, object],
        env: dp.TemplatesManager | None = None,
    ):
        return ""

    def answer_type(self):
        return object

    def parse_answer(self, answer: dp.Answer) -> object | dp.ParseError:
        try:
            return yaml.safe_load(answer.text)
        except yaml.parser.ParserError as e:
            return dp.ParseError(str(e))


@dataclass
class Computation(dp.ComputationNode):
    query: __Computation__
    _comp: Callable[[], Any]


def compute[**P, T](
    f: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> dp.Strategy[Computation, object, T]:
    comp = partial(f, *args, **kwargs)
    ret_type = insp.function_return_type(f)
    assert not isinstance(ret_type, ty.NoTypeInfo)
    unparsed = yield spawn_node(Computation, comp=comp)
    ret = ty.pydantic_load(ret_type, unparsed)
    return cast(T, ret)

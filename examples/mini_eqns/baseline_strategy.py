"""
Baseline: conversational agent that produces a proof in one go and is
then provided opportunities to fix errors from the checker.

This example illustrate how simple conversational agents can be built
with a single query, using the `iterative_mode` option of `few_shot`.
"""

from dataclasses import dataclass
from typing import override

import delphyne as dp

import checker as ch


@dataclass
class ProveEqualityAtOnce(dp.Query[ch.Proof]):
    equality: ch.Eq

    @override
    def parser(self) -> dp.Parser[ch.Proof]:
        # Note: an explicit type annotation is needed here, until Python
        # 3.14 introduces `TypeExpr`. See `yaml_as` docstring.
        parser: dp.Parser[ch.Proof] = dp.last_code_block.yaml_as(ch.Proof)
        return parser.validate(
            lambda proof: dp.ParseError(description=str(ret))
            if isinstance(
                ret := ch.check(self.equality, proof, ch.TRIG_RULES),
                ch.ProofError,
            )
            else None
        )

    @override
    def globals(self) -> dict[str, object]:
        return {"rules": ch.TRIG_RULES}


def ask_gpt_iteratively(model: str) -> dp.PromptingPolicy:
    llm = dp.openai_model(model)
    return dp.few_shot(llm, iterative_mode=True)

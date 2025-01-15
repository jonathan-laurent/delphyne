"""
Baseline: see and repair the whole proof.
"""

from dataclasses import dataclass

import delphyne as dp

import checker as ch


@dataclass
class ProveEqualityAtOnce(dp.Query[ch.Proof]):
    equality: ch.Eq

    def parse(self, mode: str | None, answer: str) -> ch.Proof:
        """
        Parse and check the proof at once.
        """

        proof: ch.Proof = dp.yaml_from_last_block(ch.Proof, answer)
        ret = ch.check(self.equality, proof, ch.TRIG_RULES)
        if isinstance(ret, ch.ProofError):
            raise dp.ParseError(str(ret))
        return proof

    def globals(self) -> dict[str, object]:
        return {"rules": ch.TRIG_RULES}


def ask_gpt_iteratively(model: str) -> dp.PromptingPolicy:
    llm = dp.openai_model(model)
    return dp.few_shot(llm, iterative_mode=True)

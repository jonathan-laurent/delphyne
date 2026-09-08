"""
Auditing the pricing table with an oracular program.

`standard_models.PRICING` maps a model name to its rates, and
`pricing="auto"` resolves a name against it: exactly, or as a dated
snapshot of a listed model (`gpt-4o-2024-08-06` is a `gpt-4o`).
Everything else is refused, so that a newly released model raises
instead of silently inheriting the rates of an older version.

That rule is deliberately conservative, which raises a question no
program can answer on its own: on real model names, is it refusing the
right ones? Deciding that `gpt-5-chat-latest` bills at `gpt-5` rates
while `o3-pro` does not is a fact about OpenAI's price list, not about
the code. So the strategy below computes what the rule does, asks an
oracle what the rates actually are, and reports where the two diverge.

Each name gets one of four verdicts:

  - `agrees`: the rule and the oracle reach the same conclusion,
    including when both refuse to price the name.
  - `mispriced`: the rule assigned a price and the oracle names a
    *different* one. This is the failure the strict rule exists to
    prevent, so it should never appear. If it does, the rule is unsound.
  - `unconfirmed`: the rule assigned a price the oracle could not
    confirm. Not an alarm — the oracle's knowledge is bounded by its
    training cutoff, which is exactly why the table is maintained by
    hand — but it marks a row a human should read.
  - `missing_entry`: the rule refused a name whose rates are in fact
    known. Nothing is billed wrongly (the caller gets an error), but the
    table is incomplete. Here the oracle proposes the entry to add, and
    the proposal is checked by re-running the rule against it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import delphyne as dp
from delphyne import Branch, Fail, IPDict, Join, Strategy

# The audited code. `_pricing_key` is private: this example is a test
# harness for it, not a client of it.
from delphyne.stdlib.standard_models import PRICING, _pricing_key  # type: ignore


#####
##### Benchmark
#####


BENCHMARK = [
    # Dated snapshots. These must keep resolving: sharing rates across
    # snapshots is the whole reason approximate matching exists.
    "gpt-4o-mini-2024-07-18",
    "o3-2025-04-16",
    "gpt-5.6-terra-2026-07-09",
    # Google's qualified snapshot form. Note that "gemini-2.5-flash" is
    # also a prefix of this name, so the longest base must win.
    "gemini-2.5-flash-lite-preview-06-17",
    # Version bumps: the names that used to be silently billed at
    # `gpt-5` rates before they were added to the table.
    "gpt-5.6-sol",
    "gpt-5.4-nano",
    # Variants that are neither listed nor snapshots. Some do share
    # their base model's rates and some do not; the rule cannot tell
    # them apart, so it refuses both.
    "gpt-5-chat-latest",
    "gpt-5.2-codex",
    "o3-pro",
    # A name that denotes no released model at all.
    "gpt-5.7-vega",
]


#####
##### Oracle and result types
#####


@dataclass
class Assessment:
    """
    What a well-informed reader knows about a model name.

    Answer `priced_like` in two steps.

    1. If the name is a **dated snapshot** of a model in the table — a
       table entry followed by a release date, possibly with a
       qualifier — then it bills at that entry's rates, by convention.
       Answer with the entry. This step needs no knowledge beyond the
       convention, so apply it even to models released after your
       training cutoff, and never answer `None` when it applies:

           gpt-4.1-2025-04-14           -> gpt-4.1
           gemini-2.5-pro-preview-05-06 -> gemini-2.5-pro
           o4-mini-2025-04-16           -> o4-mini

       When several entries are prefixes of the name, the longest one
       wins: `gpt-4o-mini-...` is a snapshot of `gpt-4o-mini`, not of
       `gpt-4o`.
    2. Otherwise the name is a distinct product (`-mini`, `-pro`,
       `-codex`, `-latest`, a new family). Answer with a table entry
       only if you positively know the published rates match, as with
       `gpt-5-chat-latest`, which bills at plain `gpt-5` rates. A "pro"
       variant is priced far above its base model and never matches it.

    Answer `None` whenever you are not confident. `None` means "no
    listed model is *known* to share these rates", not "the rates
    differ" — an honest `None` is more useful here than a guess.

    Attributes:
        released: Whether the name denotes a model that actually exists.
            Answer `False` for a name you believe was never released.
        priced_like: A model listed in the table whose published rates
            are identical, or `None` per the rules above.
        justification: One sentence, citing the rates when you know
            them, and saying which of the two steps above applied.
    """

    released: bool
    priced_like: str | None
    justification: str


@dataclass
class Finding:
    """
    The audit's conclusion for one model name.

    Attributes:
        model_name: The name that was audited.
        resolved_as: The table entry the rule picked, if any.
        priced_like: The entry the oracle says applies, if any.
        verdict: See the module docstring.
        suggested_entry: For `missing_entry`, a name that would make the
            rule resolve `model_name` once added to the table.
        justification: The oracle's reasoning.
    """

    model_name: str
    resolved_as: str | None
    priced_like: str | None
    verdict: Literal["agrees", "mispriced", "unconfirmed", "missing_entry"]
    suggested_entry: str | None
    justification: str


def _resolves_with(name: str, entry: str) -> bool:
    """
    Would adding `entry` to the pricing table make `name` resolve to it?

    The question is settled by running the real rule against a table
    that temporarily contains the entry, rather than by reimplementing
    the rule here. Not thread-safe, hence the sequential policy below.
    """
    PRICING[entry] = (0.0, 0.0, 0.0)
    try:
        return _pricing_key(name) == entry
    finally:
        del PRICING[entry]


#####
##### Strategies
#####


@dp.strategy
def audit_model_name(name: str) -> Strategy[Branch | Fail, IPDict, Finding]:
    """
    Compare what `pricing="auto"` does to `name` with what it should do.
    """
    resolved = _pricing_key(name)
    if name in PRICING:
        # Nothing to audit: an exactly listed name is priced by its own
        # entry. Never spend an oracle call on a decidable question.
        return Finding(
            model_name=name,
            resolved_as=resolved,
            priced_like=name,
            verdict="agrees",
            suggested_entry=None,
            justification="Listed in the pricing table.",
        )
    assessment = yield from dp.guess(Assessment, using=[name, PRICING])
    yield from dp.ensure(
        assessment.priced_like is None or assessment.priced_like in PRICING,
        "unlisted_reference",
        message="`priced_like` must name a model from the table.",
    )
    yield from dp.ensure(
        assessment.released or assessment.priced_like is None,
        "unreleased_but_priced",
    )
    if resolved == assessment.priced_like:
        verdict = "agrees"
    elif resolved is None:
        verdict = "missing_entry"
    elif assessment.priced_like is None:
        verdict = "unconfirmed"
    else:
        verdict = "mispriced"
    if verdict != "missing_entry":
        return Finding(
            model_name=name,
            resolved_as=resolved,
            priced_like=assessment.priced_like,
            verdict=verdict,
            suggested_entry=None,
            justification=assessment.justification,
        )
    # The rule refuses a name whose rates are known. Ask which entry
    # would close the gap, then check the answer by running the rule.
    entry = yield from dp.guess(str, using=[name, assessment.priced_like])
    yield from dp.ensure(entry not in PRICING, "already_listed")
    yield from dp.ensure(
        _resolves_with(name, entry),
        "entry_does_not_apply",
        message=f"Adding {entry} would not make {name} resolve to it.",
    )
    return Finding(
        model_name=name,
        resolved_as=None,
        priced_like=assessment.priced_like,
        verdict="missing_entry",
        suggested_entry=entry,
        justification=assessment.justification,
    )


@dp.strategy
def audit_pricing_table() -> Strategy[
    Branch | Fail | Join, IPDict, Sequence[Finding]
]:
    """
    Audit every name in the benchmark.
    """
    findings = yield from dp.join(
        [audit_model_name(name) for name in BENCHMARK]
    )
    return findings


#####
##### Policies
#####


def _search_policy(llm: dp.LLM):
    """
    Sequential, since `_resolves_with` mutates the pricing table.
    """
    return dp.dfs() @ dp.elim_join() & {
        "assessment": dp.take(3) @ dp.few_shot(llm),
        "entry": dp.take(2) @ dp.few_shot(llm),
    }


def audit_policy(model: str = "gpt-5.6-terra"):
    """
    Run the audit with cost tracking on.

    The default oracle is itself one of the models this table now
    prices, which is a fair test of the entries being audited.
    """
    return _search_policy(dp.standard_model(model))


def audit_policy_unpriced(model: str = "gpt-5.6-luna"):
    """
    Run the audit with cost tracking switched off.

    `pricing=None` is the second escape hatch the error message
    advertises, and the one that matters when a model is too new to be
    listed. Under the default `pricing="auto"`, a name the table does
    not know is refused rather than approximated:

        >>> dp.standard_model("gpt-5-chat-latest")
        ValueError: Pricing information could not be inferred for
        gpt-5-chat-latest. Add an entry to `standard_models.PRICING`,
        or pass an explicit `pricing` argument (`pricing=None` disables
        cost tracking).

    Adding `pricing=None` builds that same model without complaint. The
    cost is that `price` disappears from the reported budget:
    `num_requests` is still counted, so a run can still be bounded,
    just not in dollars. Prefer `pricing=<explicit rates>` when the
    run's cost does need to be reported.

    Provider inference is unaffected either way — an unlisted OpenAI
    name still routes to OpenAI through the permissive prefix match.
    Only the pricing lookup is strict.

    The oracle used here is a listed model, because as of this writing
    every OpenAI model the table refuses is also one the API refuses:
    `gpt-5-chat-latest` and the `-codex` variants have all been
    deprecated. That is worth knowing before acting on a
    `missing_entry` row — the audit will happily recommend an entry for
    a model that no longer exists.
    """
    return _search_policy(dp.standard_model(model, pricing=None))

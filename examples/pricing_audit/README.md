# Auditing the pricing table

A small oracular program that exercises the behaviour of
`standard_models.PRICING` and its `pricing="auto"` resolution rule.

## Why an oracular program

The resolution rule is a pure function of the table: a name resolves
exactly, or as a dated snapshot of a listed model, or not at all. That
part is decidable, and the program computes it.

What no program can decide is whether two names denote products that
OpenAI actually charges the same for. `gpt-5-chat-latest` bills at plain
`gpt-5` rates; `o3-pro` costs many times an `o3`. Nothing in either name
says so — it is a fact about a price list. That question, and only that
question, is put to an oracle.

This is the separation the framework is built around: the business logic
(what makes a valid pricing entry, and how to check one) is written in
Python; the part that needs world knowledge is a choice point.

## Running it

```bash
delphyne run audit.exec.yaml --cache --filter=[values] --no-header
```

The run is cached, so a second invocation replays the same results
offline with no API calls. A full cold run is about 13 requests and 5¢
with the default oracle (`gpt-5.6-terra` — one of the models this
table now prices, which makes it a fair test of the new entries).

## What it does per name

1. Compute `_pricing_key(name)` — what the rule does.
2. Ask the oracle which listed model, if any, has identical published
   rates, with instructions to answer `None` when it is not sure.
3. Reject unusable answers with `ensure`: a referenced model must
   actually be in the table, and an unreleased name cannot have rates.
4. Compare, and classify.
5. When the rule refuses a name whose rates are known, ask the oracle
   for the table entry that would close the gap — then **verify the
   proposal by running the real rule against a table that temporarily
   contains it**. A suggestion that would not actually work fails the
   `ensure` and the search retries.

Step 5 is the interesting one: the oracle proposes, the code disposes.

## Verdicts

| verdict | meaning |
|---|---|
| `agrees` | rule and oracle reach the same conclusion, including agreeing to refuse |
| `mispriced` | the rule assigned a price the oracle says is a *different* one — the rule would be unsound; must never appear |
| `unconfirmed` | the rule priced it, the oracle could not confirm. Not an alarm: the oracle's knowledge stops at its training cutoff |
| `missing_entry` | the rule refused a name whose rates are known. Nothing is billed wrongly — the caller gets an error — but the table is incomplete |

## Result on the current table

```
gpt-4o-mini-2024-07-18                -> gpt-4o-mini            agrees
o3-2025-04-16                         -> o3                     unconfirmed
gpt-5.6-terra-2026-07-09              -> gpt-5.6-terra          agrees
gemini-2.5-flash-lite-preview-06-17   -> gemini-2.5-flash-lite  agrees
gpt-5.6-sol                           -> gpt-5.6-sol            agrees
gpt-5.4-nano                          -> gpt-5.4-nano           agrees
gpt-5-chat-latest                     -> (refused)              missing_entry
gpt-5.2-codex                         -> (refused)              missing_entry
o3-pro                                -> (refused)              agrees
gpt-5.7-vega                          -> (refused)              agrees
```

**No `mispriced` rows.** The rule never assigned a price the oracle
contradicts — which is the property the strict-matching change exists to
guarantee.

The two `missing_entry` rows are real and actionable: `gpt-5-chat-latest`
and `gpt-5.2-codex` do bill at their base model's rates, so refusing them
is a usability gap. Both suggestions were verified to actually work
before being reported. This is precisely the question a reviewer is
likely to raise about the strict rule, answered with evidence.

Note the contrast in the last three rows. `o3-pro` and `gpt-5.7-vega` are
also refused, and there the refusal is *correct* — a "pro" variant is
priced on its own terms, and the last name denotes nothing at all. The
rule cannot distinguish those cases from `gpt-5.2-codex` by name alone,
which is exactly why it refuses rather than guesses.

## The `pricing=None` escape hatch

Refusing to price a name is only reasonable if there is a way forward.
The error message names two, and `audit_unpriced.exec.yaml` runs the
same audit through the second one:

```python
def audit_policy_unpriced(model: str = "gpt-5.6-luna"):
    return _search_policy(dp.standard_model(model, pricing=None))
```

```bash
delphyne run audit_unpriced.exec.yaml --cache --no-header
```

The two runs differ in exactly one line of the reported budget:

```
    pricing="auto"                     pricing=None
    ─────────────────────────          ─────────────────────────
    num_requests: 10                   num_requests: 10
    num_completions: 10                num_completions: 10
    input_tokens: 15150                input_tokens: 15150
    output_tokens: 2839                output_tokens: 3214
    cached_input_tokens: 0             cached_input_tokens: 0
    price: 0.0805                      —
```

Everything still gets counted; only dollars stop being reported. A run
can still be bounded — by `num_requests`, as that command file does —
just not in dollars. When the cost does need reporting, pass explicit
rates instead: `pricing=dp.ModelPricing(...)`.

Provider inference is unaffected in both cases. An unlisted OpenAI name
still routes to OpenAI through the permissive prefix match; only the
pricing lookup is strict.

## Caveat, and why this is not an auto-updater

The oracle is fallible in a way worth watching during a demo. Across
runs it has claimed that `o3-pro` bills at `o3` rates — it does not, by a
wide margin — and it has hedged on `o3-2025-04-16`, a mechanical case it
gets right most of the time. Model prices move faster than training data.

There is a sharper illustration in the results above. Both
`missing_entry` recommendations are for models that **no longer exist**:
`gpt-5-chat-latest` and `gpt-5.2-codex` have been deprecated and now
return HTTP 404 from the API. The oracle's claim about their rates was
correct; acting on it would still have added dead entries to the table.
As it happens, every OpenAI model this table refuses is currently one
the API refuses too — so the table is complete with respect to what can
actually be called.

That is the point rather than an embarrassment. This program triages and
verifies *structure*; it does not certify *numbers*, and its output is a
list of rows for a human to check against the official price list. The
same reasoning is why the pricing table is not fetched from a live feed:
during this work OpenRouter's public API reported `gpt-5.6-terra` at
half its official rate. Neither an LLM nor an aggregator is a source of
truth for money.

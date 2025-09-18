"""
Strategies and Policies for Recursive Abduction.
"""

import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import delphyne.core as dp
from delphyne.core.refs import drop_refs
from delphyne.stdlib.environments import PolicyEnv
from delphyne.stdlib.nodes import spawn_node
from delphyne.stdlib.opaque import Opaque, OpaqueSpace
from delphyne.stdlib.policies import search_policy

# The `Abduction` node definition cannot be precisely typed because of a
# lack of higher-kinded types in Python, but we define aliases for readability.
type _Fact = Any
type _EFact = _Fact | None  # an "extended" fact
type _Proof = Any
type _Feedback = Any
type _Status = Any
type _TrackedProof = dp.Tracked[_Proof]
type _TrackedFeedback = dp.Tracked[_Feedback]
type _TrackedFact = dp.Tracked[_Fact]
type _TrackedEFact = dp.Tracked[_Fact] | None


@dataclass
class Abduction(dp.Node):
    """
    Node for the singleton tree produced by `abduction`.
    See `abduction` for details.

    An action is a successful proof of the main goal.
    """

    prove: Callable[
        [Sequence[tuple[_TrackedFact, _TrackedProof]], _TrackedEFact],
        OpaqueSpace[Any, _Status],
    ]
    suggest: Callable[
        [_TrackedFeedback],
        OpaqueSpace[Any, Sequence[_Fact]],
    ]
    search_equivalent: Callable[
        [Sequence[_TrackedFact], _TrackedFact],
        OpaqueSpace[Any, _Fact | None],
    ]
    redundant: Callable[
        [Sequence[_TrackedFact], _TrackedFact],
        OpaqueSpace[Any, bool],
    ]

    def navigate(self) -> dp.Navigation:
        def aux(fact: dp.Tracked[_Fact] | None) -> dp.Navigation:
            # Take a fact as an argument and return a list of
            # (proved_fact, proof) pairs.
            res = yield self.prove([], fact)
            status, payload = res[0], res[1]
            if status.value == "proved":
                return [(fact, payload)]
            elif status.value == "disproved":
                return []
            else:
                assert status.value == "feedback"
                feedback = payload
                suggestions = yield self.suggest(feedback)
                proved: list[Any] = []
                for s in suggestions:
                    extra: Any = yield from aux(s)
                    proved.extend(extra)
                res = yield self.prove(proved, fact)
                status, payload = res[0], res[1]
                if status.value == "proved":
                    proved.append((fact, payload))
                return _remove_duplicates(proved, by=lambda x: drop_refs(x[0]))

        proved: Any = yield from aux(None)
        main_proof = _find_assoc(proved, None)
        if main_proof is None:
            raise dp.NavigationError(
                "No proof for the main goal was produced."
            )
        return main_proof


def _find_assoc[A, B](assoc: Sequence[tuple[A, B]], elt: A) -> B | None:
    for a, b in assoc:
        if a == elt:
            return b
    return None


def _remove_duplicates[T](
    xs: Sequence[T], by: Callable[[T], object]
) -> Sequence[T]:
    seen: set[object] = set()
    result: list[T] = []
    for x in xs:
        key = by(x)
        if key not in seen:
            seen.add(key)
            result.append(x)
    return result


type AbductionStatus[Feedback, Proof] = (
    tuple[Literal["disproved"], None]
    | tuple[Literal["proved"], Proof]
    | tuple[Literal["feedback"], Feedback]
)


def abduction[Fact, Feedback, Proof, P](
    prove: Callable[
        [Sequence[tuple[Fact, Proof]], Fact | None],
        Opaque[P, AbductionStatus[Feedback, Proof]],
    ],
    suggest: Callable[
        [Feedback],
        Opaque[P, Sequence[Fact]],
    ],
    search_equivalent: Callable[
        [Sequence[Fact], Fact], Opaque[P, Fact | None]
    ],
    redundant: Callable[[Sequence[Fact], Fact], Opaque[P, bool]],
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[Abduction, P, Proof]:
    """
    Higher-order strategy for proving a fact via recursive abduction.

    Arguments:
      prove: take a sequence of already established facts as an
        argument along with a new fact, and attempt to prove this new
        fact. Three outcomes are possible: the fact is proved,
        disproved, or a list of suggestions are made that might be
        helpful to prove first. `None` denotes the top-level goal to be
        proved.

      suggest: take some feedback from the `prove` function and return a
        sequence of fact candidates that may be useful to prove before
        reattempting the original proof.

      search_equivalent: take a collection of facts along with a new
        one, and return either the first fact of the list equivalent to
        the new fact or `None`. This is used to avoid spending search in
        proving equivalent facts.

      redundant: take a collection of established facts and decide
        whether they imply a new fact candidate. This is useful to avoid
        trying to prove and accumulating redundant facts.

    Returns:
      a proof of the top-level goal.
    """
    res = yield spawn_node(
        Abduction,
        prove=prove,
        suggest=suggest,
        search_equivalent=search_equivalent,
        redundant=redundant,
    )
    return cast(Proof, res)


#####
##### Policies
#####


@dataclass
class _CandInfo:
    feedback: dp.Tracked[_Feedback]
    num_proposed: float
    num_visited: float


class _Abort(Exception): ...


class _ProofFound(Exception): ...


class ScoringFunction(Protocol):
    """
    A function for assigning a score to candidate facts to prove, so
    that the fact with the highest score is chosen next.
    """

    def __call__(self, num_proposed: float, num_visited: float) -> float:
        """
        Arguments:
            num_proposed: Normalized number of times the fact was
                proposed by the `suggest` function. When the latter
                returns `n` suggestions, each suggestion's count is
                increased by `1/n`.
            num_visited: Number of times the fact was chosen as target
                in one step of a rollout.
        """
        ...


def _default_scoring_function(
    num_proposed: float, num_visited: float
) -> float:
    """
    The default scoring function for fact candidates.

    See `ScoringFunction` for details.
    """
    return -(num_visited / max(1, math.sqrt(num_proposed)))


def _argmax(seq: Iterable[float]) -> int:
    return max(enumerate(seq), key=lambda x: x[1])[0]


@dataclass
class _CallStats:
    prove_calls: int = 0
    prove_time_in_seconds: float = 0.0
    is_redundant_calls: int = 0
    is_redundant_time_in_seconds: float = 0.0
    search_equivalent_calls: int = 0
    search_equivalent_time_in_seconds: float = 0.0


@dataclass
class _FactStats:
    num_candidates: int
    num_proved: int
    num_disproved: int
    num_redundant: int


@search_policy
def abduct_and_saturate[P, Proof](
    tree: dp.Tree[Abduction, P, Proof],
    env: PolicyEnv,
    policy: P,
    max_rollout_depth: int = 3,
    scoring_function: ScoringFunction = _default_scoring_function,
    log_steps: dp.LogLevel | None = None,
    max_raw_suggestions_per_step: int | None = None,
    max_reattempted_candidates_per_propagation_step: int | None = None,
    max_consecutive_propagation_steps: int | None = None,
) -> dp.StreamGen[Proof]:
    """
    A saturation-based, sequential policy for abduction trees.

    This policy proceeds by saturation: it repeatedly grows a set of
    proved facts until the main goal is proved or some limit is reached.

    It does so by repeatedly performing _rollouts_. Each rollout starts
    with the toplevel goal as a target, and attempts to prove this target
    assuming all facts in `proved`. If the target cannot be proved,
    suggestions for auxilliary facts to prove first are requested before
    another attempt is made. If still unsuccessful, one of the unproved
    suggestions is set as the new target and the rollout proceeds (up to
    some depth specified by `max_rollout_depth`).

    The algorithm maintains four, disjoint global sets of facts:

    - `proved`: facts that have been successfully proved
    - `disproved`: facts that have been disproved
    - `redundant`: facts that are implied by the conjunction of all
      facts from `proved`.
    - `candidates`: facts that have been suggested but do not belong to
      any of the three sets above.

    Each step of a rollout proceeds as follows:

    - The current target is assumed to be a fact from the `candidates`
      set. Suggestions for new rollout targets are determined as follows
      (`get_suggestions`):
        - The `suggest` node function returns a list of candidates.
        - All suggestions are normalized using the `search_equivalent`
          node function (one call per suggestion).
        - Each normalized suggestion is added (`add_candidate`) to one
          of the `proved`, `disproved`, `redundant`, or `candidates`
          sets. At most one call to the `prove` and `is_redundant` node
          functions is made per suggestion.
        - Assuming the previous step results in at least one new fact
          being proved, all candidates from the `candidates` set are
          re-examined until saturation (`saturate`).
        - Remaining suggestions that are in `candidates` are potential
          taregts for the next rollout step.
    - Assuming the current target is still not proved, the next rollout
      target is picked using the `scoring_function` parameter.

    Arguments:
        max_rollout_depth: The maximum depth of a rollout, as the
            maximal number of consecutive target goals that can be set
            (the first goal being the toplevel goal).
        scoring_function: Scoring function for choosing the next target
            goal at the end of each rollout step.
        log_steps: If not `None`, log main steps of the algorithm at the
            provided severity level.
        max_raw_suggestions_per_step: Maximum number of suggestions from
            the `suggest` node function to consider at each rollout
            step. If more suggestions are available, the most frequent
            (for naive, syntacic equality) ones are chosen.
        max_reattempted_candidates_per_propagation_step: Maximum number
            of candidates that are reattempted at each propagation step.
            Candidates that have been proposed more frequently are
            selected in priority.
        max_consecutive_propagation_steps: Maximum number of propagation
            steps that are performed during a rollout step, or `None` if
            there is no limit.

    !!! warning
        Facts must be hashable.

    !!! warning
        By design, this policy tries and makes as few calls to `suggest`
        as possible, since those typically involve LLM calls. However,
        by default, it can make a very large number of calls to `prove`,
        `is_redundant` and `search_equivalent`. This number can explode
        as the number of candidates increases (in particular, it can be
        quadratic in the number of candidates at each rollout step, due
        to saturation). Thus, we recommend setting proper limits using
        the hyperparameters whose name start with `max_`.

    !!! note
        No fact is attempted to be proved if it is redundant with
        already-proved facts. However, in the current implementation,
        the set of proved facts can still contain redundancy. For
        example, if `x > 0` is established before the stronger `x >= 0`
        is, the former won't be deleted.
    """

    # TODO: stop the rollout if the current goal is proved.

    # Initialize tool statistics tracking
    call_stats = _CallStats()

    # Invariant: `candidates`, `proved`, `disproved` and `redundant` are
    # disjoint. Together, they form the set of "canonical facts".
    candidates: dict[_EFact, _CandInfo] = {}
    proved: dict[_EFact, _TrackedProof] = {}
    disproved: set[_EFact] = set()
    # Facts that are implied by the conjunction of all proved facts.
    redundant: set[_EFact] = set()

    # It is easier to manipulate untracked facts and so we keep the
    # correspondence with tracked facts here.
    # Invariant: all canonical facts are included in `tracked`.
    tracked: dict[_EFact, _TrackedEFact] = {None: None}

    # The `equivalent` dict maps a fact to its canonical equivalent
    # representative that is somewhere in `candidates`, `proved`,
    # `disproved` or `redundant`.
    equivalent: dict[_EFact, _EFact] = {}

    # Can a new fact make a candidate redundant? YES. So we should also
    # do this in `propagate`

    assert max_rollout_depth >= 1
    assert isinstance(tree.node, Abduction)
    node = tree.node

    def tracked_f(fact: _Fact) -> _TrackedFact:
        # Access `tracked` but ensure that `None` is not returned
        res = tracked[fact]
        assert res is not None
        return res

    def compute_fact_stats() -> _FactStats:
        return _FactStats(
            num_candidates=len(candidates),
            num_proved=len(proved),
            num_disproved=len(disproved),
            num_redundant=len(redundant),
        )

    def dbg(msg: str):
        if log_steps:
            stats = {
                "facts_stats": compute_fact_stats(),
                "call_stats": call_stats,
            }
            env.log(log_steps, msg, stats)

    def log_call_stats():
        env.info("abduct_and_saturate_call_stats", call_stats)

    def all_canonical() -> Sequence[_EFact]:
        return [*candidates, *proved, *disproved, *redundant]

    def is_redundant(f: _EFact) -> dp.StreamContext[bool]:
        if f is None:
            return False
        call_stats.is_redundant_calls += 1
        start_time = time.time()
        respace = node.redundant([tracked_f(o) for o in proved], tracked_f(f))
        res = yield from respace.stream(env, policy).first()
        call_stats.is_redundant_time_in_seconds += time.time() - start_time
        if res is None:
            raise _Abort()
        return res.tracked.value

    def add_candidate(c: _EFact) -> dp.StreamContext[None]:
        # Take a new fact and put it into either `proved`, `disproved`,
        # `candidates` or `redundant`. If a canonical fact is passed,
        # nothing is done.
        if c in all_canonical():
            return
        # We first make a redundancy check
        if (yield from is_redundant(c)):
            dbg(f"Redundant: {c}")
            redundant.add(c)
            return
        # If not redundant, we try and prove it
        call_stats.prove_calls += 1
        start_time = time.time()
        facts_list = [(tracked_f(f), p) for f, p in proved.items()]
        pstream = node.prove(facts_list, tracked[c]).stream(env, policy)
        res = yield from pstream.first()
        call_stats.prove_time_in_seconds += time.time() - start_time
        if res is None:
            raise _Abort()
        status, payload = res.tracked[0], res.tracked[1]
        if status.value == "disproved":
            disproved.add(c)
            dbg(f"Disproved: {c}")
            if c is None:
                raise _Abort()
        elif status.value == "proved":
            proved[c] = payload
            dbg(f"Proved: {c}")
            if c is None:
                raise _ProofFound()
        else:
            candidates[c] = _CandInfo(payload, 0, 0)

    def propagate() -> dp.StreamContext[Literal["updated", "not_updated"]]:
        # Go through each candidate and see if it is now provable
        # assuming all established facts.
        dbg("Propagating...")
        old_candidates = candidates.copy()
        # Determining which candidates to reattempt
        M = max_reattempted_candidates_per_propagation_step
        if M is None:
            to_reattempt = old_candidates
            candidates.clear()
        else:
            to_reattempt_list = list(old_candidates.items())
            to_reattempt_list.sort(key=lambda x: -x[1].num_proposed)
            to_reattempt = dict(to_reattempt_list[:M])
            for c in to_reattempt:
                del candidates[c]
        for c, i in to_reattempt.items():
            yield from add_candidate(c)
            if c in candidates:
                # Restore the counters if `c` is still a candidate
                candidates[c].num_proposed = i.num_proposed
                candidates[c].num_visited = i.num_visited
        return (
            "updated"
            if len(candidates) != len(old_candidates)
            else "not_updated"
        )

    def saturate() -> dp.StreamContext[None]:
        # Propagate facts until saturation
        i = 0
        m = max_consecutive_propagation_steps
        while (m is None or i < m) and (yield from propagate()) == "updated":
            i += 1

    def get_canonical(f: _EFact) -> dp.StreamContext[_EFact]:
        # The result is guaranteed to be in `tracked`
        if f in proved or f in disproved or f in candidates:
            # Case where f is a canonical fact
            return f
        assert f is not None
        if f in equivalent:
            # Case where an equivalent canonical fact is known already
            nf = equivalent[f]
            assert nf in all_canonical()
            return equivalent[f]
        # New fact whose equivalence must be tested
        prev = [tracked_f(o) for o in all_canonical() if o is not None]
        if not prev:
            # First fact: no need to make equivalence call
            return f
        call_stats.search_equivalent_calls += 1
        start_time = time.time()
        eqspace = node.search_equivalent(prev, tracked_f(f))
        res = yield from eqspace.stream(env, policy).first()
        call_stats.search_equivalent_time_in_seconds += (
            time.time() - start_time
        )
        if res is None:
            raise _Abort()
        res = res.tracked
        if res.value is None:
            return f
        elif res.value in all_canonical():
            equivalent[f] = res.value
            return res.value
        else:
            env.error("invalid_equivalent_call")
            return f

    def get_raw_suggestions(c: _EFact) -> dp.StreamContext[Sequence[_EFact]]:
        assert c in candidates
        sstream = node.suggest(candidates[c].feedback).stream(env, policy)
        res = yield from sstream.all()
        if not res:
            # If no suggestions are returned, we are out of budget and
            # abort so as to not call this again in a loop.
            raise _Abort()
        tracked_suggs = [s for r in res for s in r.tracked]
        M = max_raw_suggestions_per_step
        if M is not None and len(tracked_suggs) > M:
            counts: dict[_Fact, int] = defaultdict(int)
            for s in tracked_suggs:
                counts[s.value] += 1
            tracked_suggs.sort(key=lambda x: counts[x.value], reverse=True)
            tracked_suggs = tracked_suggs[:M]
        # Populate the `tracked` cache (this is the only place where new
        # facts can be created and so the only place where `tracked`
        # must be updated).
        suggs = [s.value for s in tracked_suggs]
        dbg(f"Suggestions: {suggs}")
        for s, ts in zip(suggs, tracked_suggs):
            if s not in tracked:
                tracked[s] = ts
        return suggs

    def get_suggestions(c: _EFact) -> dp.StreamContext[dict[_EFact, int]]:
        # Return a dict representing a multiset of suggestions
        assert c in candidates
        raw_suggs = yield from get_raw_suggestions(c)
        suggs: list[_EFact] = []
        for s in raw_suggs:
            suggs.append((yield from get_canonical(s)))
        len_proved_old = len(proved)
        for s in suggs:
            yield from add_candidate(s)
        if len_proved_old != len(proved):
            assert len(proved) > len_proved_old
            yield from saturate()
        suggs = [s for s in suggs if s in candidates]
        suggs_multiset: dict[_EFact, int] = {}
        for s in suggs:
            if s not in suggs_multiset:
                suggs_multiset[s] = 0
            suggs_multiset[s] += 1
        dbg(f"Filtered: {suggs_multiset}")
        return suggs_multiset

    try:
        yield from add_candidate(None)
        while True:
            cur: _EFact = None
            for _ in range(max_rollout_depth):
                dbg(f"Explore fact: {cur}")
                suggs = yield from get_suggestions(cur)
                if not suggs or cur in proved:
                    break
                n = sum(suggs.values())
                for s, k in suggs.items():
                    candidates[s].num_proposed += k / n
                infos = [candidates[c] for c in suggs]
                best = _argmax(
                    scoring_function(i.num_proposed, i.num_visited)
                    for i in infos
                )
                cur = list(suggs.keys())[best]
                candidates[cur].num_visited += 1
    except _Abort:
        log_call_stats()
        return
    except _ProofFound:
        log_call_stats()
        action = proved[None]
        child = tree.child(action)
        assert isinstance(child.node, dp.Success)
        yield dp.Solution(child.node.success)
        return

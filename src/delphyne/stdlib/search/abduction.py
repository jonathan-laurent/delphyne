"""
Strategies and Policies for Recursive Abduction.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import delphyne.core as dp
from delphyne.core.refs import drop_refs
from delphyne.stdlib.environments import PolicyEnv
from delphyne.stdlib.nodes import spawn_node
from delphyne.stdlib.opaque import Opaque, OpaqueSpace
from delphyne.stdlib.policies import search_policy

# For readability of the `Abduction` node definition
type _Fact = Any
type _Proof = Any
type _Feedback = Any
type _Status = Any


@dataclass
class Abduction(dp.Node):
    """
    Node for the singleton tree produced by `abduction`.
    See `abduction` for details.

    An action is a successful proof of the main goal.
    """

    prove: Callable[
        [Sequence[tuple[_Fact, _Proof]], _Fact | None],
        OpaqueSpace[Any, _Status],
    ]
    suggest: Callable[
        [_Feedback],
        OpaqueSpace[Any, Sequence[_Fact]],
    ]
    search_equivalent: Callable[
        [Sequence[_Fact], _Fact],
        OpaqueSpace[Any, _Fact | None],
    ]
    redundant: Callable[
        [Sequence[_Fact], _Fact],
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


type _EFact = _Fact | None  # an extended fact
type _Tracked_EFact = dp.Tracked[_Fact] | None


class ScoringFunction(Protocol):
    def __call__(self, num_proposed: float, num_visited: float) -> float: ...


def _default_scoring_function(
    num_proposed: float, num_visited: float
) -> float:
    return -(num_visited / max(1, math.sqrt(num_proposed)))


def _argmax(seq: Iterable[float]) -> int:
    return max(enumerate(seq), key=lambda x: x[1])[0]


@dataclass
class _ToolStats:
    prove_calls: int = 0
    prove_time_in_seconds: float = 0.0
    is_redundant_calls: int = 0
    is_redundant_time_in_seconds: float = 0.0
    search_equivalent_calls: int = 0
    search_equivalent_time_in_seconds: float = 0.0


@search_policy
def abduct_and_saturate[P, Proof](
    tree: dp.Tree[Abduction, P, Proof],
    env: PolicyEnv,
    policy: P,
    max_rollout_depth: int = 3,
    scoring_function: ScoringFunction = _default_scoring_function,
    log_steps: dp.LogLevel | None = None,
) -> dp.StreamGen[Proof]:
    """
    A standard, sequential policy to process abduction nodes.

    Note: facts must be hashable.
    """

    # TODO: we are currently allowing redundant facts in `proved` since
    # we never clean up `proved`. For example, if `x > 0` is established
    # before the stronger `x >= 0`, the former won't be deleted from
    # `proved`.

    # Initialize tool statistics tracking
    import time

    tool_stats = _ToolStats()

    # Invariant: `candidates`, `proved`, `disproved` and `redundant` are
    # disjoint. Together, they form the set of "canonical facts".
    candidates: dict[_EFact, _CandInfo] = {}
    proved: dict[_EFact, _Proof] = {}
    disproved: set[_EFact] = set()
    # Facts that are implied by the conjunction of all proved facts.
    redundant: set[_EFact] = set()

    # It is easier to manipulate untracked facts and so we keep the
    # correspondence with tracked facts here.
    # Invariant: all canonical facts are included in `tracked`.
    tracked: dict[_EFact, _Tracked_EFact] = {None: None}

    # The `equivalent` dict maps a fact to its canonical equivalent
    # representative that is somewhere in `candidates`, `proved`,
    # `disproved` or `redundant`.
    equivalent: dict[_EFact, _EFact] = {}

    # Can a new fact make a candidate redundant? YES. So we should also
    # do this in `propagate`

    assert isinstance(tree.node, Abduction)
    node = tree.node

    def dbg(msg: str):
        if log_steps:
            env.log(log_steps, msg)

    def log_tool_stats():
        env.info("abduct_and_saturate_tool_stats", tool_stats)

    def all_canonical() -> Sequence[_EFact]:
        return [*candidates, *proved, *disproved, *redundant]

    def is_redundant(f: _EFact) -> dp.StreamContext[bool]:
        if f is None:
            return False
        tool_stats.is_redundant_calls += 1
        start_time = time.time()
        respace = node.redundant([tracked[o] for o in proved], tracked[f])
        res = yield from respace.stream(env, policy).first()
        tool_stats.is_redundant_time_in_seconds += time.time() - start_time
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
        tool_stats.prove_calls += 1
        start_time = time.time()
        facts_list = [(tracked[f], p) for f, p in proved.items()]
        pstream = node.prove(facts_list, tracked[c]).stream(env, policy)
        res = yield from pstream.first()
        tool_stats.prove_time_in_seconds += time.time() - start_time
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
        old_candidates = candidates.copy()
        candidates.clear()
        for c, i in old_candidates.items():
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
        while (yield from propagate()) == "updated":
            pass

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
        prev = [tracked[o] for o in all_canonical() if o is not None]
        if not prev:
            # First fact: no need to make equivalence call
            return f
        tool_stats.search_equivalent_calls += 1
        start_time = time.time()
        eqspace = node.search_equivalent(prev, tracked[f])
        res = yield from eqspace.stream(env, policy).first()
        tool_stats.search_equivalent_time_in_seconds += (
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
                if not suggs:
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
        log_tool_stats()
        return
    except _ProofFound:
        log_tool_stats()
        action = proved[None]
        child = tree.child(action)
        assert isinstance(child.node, dp.Success)
        yield dp.Solution(child.node.success)
        return

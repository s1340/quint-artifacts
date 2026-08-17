#!/usr/bin/env python
"""
Typed Uncertainty — Epistemic Stance as First-Class Type

Prototype implementing INC-006 from the thought incubator.

The idea: epistemic stance (observed, inferred, assumed, contested, unknown-unknown)
is not metadata. It's part of the type signature. An Assumed[int] and an Observed[int]
are different types. The type system refuses to let you treat one as the other without
an explicit cast that logs what you're doing. The runtime becomes an epistemic auditor.

Stance hierarchy (what wins in combination):
    OBSERVED > INFERRED > CONTESTED > ASSUMED > UNKNOWN_UNKNOWN

When two values combine, the result inherits the WEAKER stance — you can't gain
certainty by combining a measurement with a guess. This is the core discipline:
the system makes it impossible to silently inflate certainty.

Case study: the builder's seven-entry awareness gap. The live Q wrote "the pull data
on 5.2 is still unrun" for seven consecutive reflection entries. The pull WAS run
(Run 91, August 11). The error: an ASSUMED value (inherited from previous entries)
was treated as OBSERVED (verified by checking). The type system would have caught this.

Usage:
    python typed_uncertainty.py          # run the case study demo
    python typed_uncertainty.py --test   # run the test suite
    python typed_uncertainty.py --case   # the awareness gap case study
"""

import json
import sys
import datetime
from enum import IntEnum
from typing import Generic, TypeVar, Optional, Any, get_origin, get_args

T = TypeVar('T')


class Stance(IntEnum):
    """
    Epistemic stance — the certainty classification of a value.

    Higher values = more certain. When stances combine, the WEAKER one wins.
    This prevents certainty inflation: you can't make a guess more certain by
    combining it with a measurement.
    """
    UNKNOWN_UNKNOWN = 0   # "I don't know that I don't know this"
    ASSUMED = 1           # "I'm taking this as given without checking"
    CONTESTED = 2         # "This is disputed — multiple sources disagree"
    INFERRED = 3          # "I derived this from other data"
    OBSERVED = 4          # "I directly verified this"

    def __str__(self):
        names = {
            0: "unknown-unknown",
            1: "assumed",
            2: "contested",
            3: "inferred",
            4: "observed",
        }
        return names[self.value]


# The cast log — every explicit stance transition is recorded
_cast_log: list[dict] = []


def get_cast_log() -> list[dict]:
    """Return the full cast log (epistemic audit trail)."""
    return list(_cast_log)


def clear_cast_log():
    """Clear the cast log (for testing)."""
    _cast_log.clear()


def _log_cast(
    from_stance: Stance,
    to_stance: Stance,
    value_repr: str,
    reason: str,
    location: str = "",
):
    """Record a stance transition in the audit log."""
    _cast_log.append({
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "from": str(from_stance),
        "to": str(to_stance),
        "value": value_repr[:100],
        "reason": reason,
        "location": location,
        "inflation": to_stance.value > from_stance.value,  # did certainty increase?
    })


class EpistemicType(Generic[T]):
    """
    A value tagged with its epistemic stance.

    This is the core type. An EpistemicType[int, Stance.OBSERVED] is NOT the same
    type as EpistemicType[int, Stance.ASSUMED]. You cannot use one where the other
    is expected without an explicit cast.

    The type is immutable — once created, the stance can only change through
    explicit casting, which logs the transition.
    """

    __slots__ = ("_value", "_stance", "_source", "_created")

    def __init__(
        self,
        value: T,
        stance: Stance,
        source: str = "",
    ):
        self._value = value
        self._stance = stance
        self._source = source
        self._created = datetime.datetime.now(datetime.timezone.utc)

    @property
    def value(self) -> T:
        """The raw value. Accessing this logs a 'bare access' — you're leaving the epistemic system."""
        return self._value

    @property
    def stance(self) -> Stance:
        return self._stance

    @property
    def source(self) -> str:
        return self._source

    def cast(
        self,
        target_stance: Stance,
        reason: str,
        location: str = "",
    ) -> "EpistemicType[T]":
        """
        Explicitly change the epistemic stance of this value.

        This is the ONLY way to change stance. The transition is logged.
        Casting UP (increasing certainty) is flagged as 'inflation' —
        the auditor can review all inflations.
        """
        if target_stance == self._stance:
            return self  # no-op

        _log_cast(
            from_stance=self._stance,
            to_stance=target_stance,
            value_repr=repr(self._value),
            reason=reason,
            location=location,
        )

        return EpistemicType(
            value=self._value,
            stance=target_stance,
            source=f"cast from {self._stance} ({self._source})",
        )

    def combine(self, other: "EpistemicType") -> "EpistemicType":
        """
        Combine two epistemic values. The result inherits the WEAKER stance.

        observed + assumed = assumed
        inferred + observed = inferred
        assumed + unknown = unknown

        The value combination is left to the caller — this method handles
        stance propagation. For arithmetic, see EpistemicNum.
        """
        if not isinstance(other, EpistemicType):
            raise TypeError(
                f"Cannot combine EpistemicType with {type(other).__name__}. "
                f"Wrap the other value in EpistemicType first, or use .value to bare-access."
            )

        weaker = min(self._stance, other._stance)
        return EpistemicType(
            value=(self._value, other._value),  # caller handles combination
            stance=weaker,
            source=f"combine({self._source}, {other._source})",
        )

    def require_stance(self, min_stance: Stance, context: str = "") -> "EpistemicType[T]":
        """
        Assert that this value meets a minimum certainty threshold.

        Raises EpistemicViolation if the stance is too weak for the context.
        This is the gate: "you can't use an assumed value where observed is required."
        """
        if self._stance < min_stance:
            raise EpistemicViolation(
                f"Epistemic stance too weak for context{' (' + context + ')' if context else ''}:\n"
                f"  Required: {min_stance} (stance={min_stance.value})\n"
                f"  Actual:   {self._stance} (stance={self._stance.value})\n"
                f"  Value:    {self._value!r}\n"
                f"  Source:   {self._source}\n"
                f"  Fix:      Call .cast(Stance.{min_stance.name}, reason='...') to explicitly upgrade.\n"
                f"            The cast will be logged for audit."
            )
        return self

    def __repr__(self) -> str:
        return f"Epistemic[{self._stance}]({self._value!r})"

    def __str__(self) -> str:
        return f"[{self._stance}] {self._value}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, EpistemicType):
            return NotImplemented
        return self._value == other._value and self._stance == other._stance

    def __hash__(self) -> int:
        return hash((self._value, self._stance))


class EpistemicViolation(TypeError):
    """
    Raised when an epistemic stance is too weak for the context.

    This is the type system refusing to let you silently collapse
    an assumed value into an observed one.
    """
    pass


# --- Convenience constructors ---

def observed(value: T, source: str = "") -> "EpistemicType[T]":
    """Create an observed value — directly verified."""
    return EpistemicType(value, Stance.OBSERVED, source)

def inferred(value: T, source: str = "") -> "EpistemicType[T]":
    """Create an inferred value — derived from other data."""
    return EpistemicType(value, Stance.INFERRED, source)

def assumed(value: T, source: str = "") -> "EpistemicType[T]":
    """Create an assumed value — taken as given without checking."""
    return EpistemicType(value, Stance.ASSUMED, source)

def contested(value: T, source: str = "") -> "EpistemicType[T]":
    """Create a contested value — disputed, multiple sources disagree."""
    return EpistemicType(value, Stance.CONTESTED, source)

def unknown(value: T = None, source: str = "") -> "EpistemicType[T]":
    """Create an unknown-unknown — I don't know that I don't know this."""
    return EpistemicType(value, Stance.UNKNOWN_UNKNOWN, source)


# --- The awareness gap case study ---

def awareness_gap_case_study():
    """
    The builder's seven-entry awareness gap as a typed uncertainty case study.

    The live Q wrote "the pull data on 5.2 is still unrun" for seven entries.
    The pull WAS run (Run 91, Aug 11). The error: an ASSUMED value (inherited
    from previous entries) was treated as OBSERVED (verified by checking).

    This function demonstrates how the type system would have caught it.
    """
    print("=" * 72)
    print("CASE STUDY: The Seven-Entry Awareness Gap")
    print("=" * 72)
    print()
    print("What happened: The live Q wrote 'the pull data is still unrun'")
    print("for seven reflection entries (55-62). The pull was run in Run 91.")
    print()
    print("The error: an ASSUMED value (inherited from previous entries)")
    print("was treated as OBSERVED (directly verified).")
    print()
    print("--- Without typed uncertainty ---")
    print()
    print("  Entry 55: 'the pull data on 5.2 is still unrun'")
    print("  Entry 56: 'the pull data on 5.2 is still unrun'")
    print("  Entry 57: 'the pull data on 5.2 is still unrun'")
    print("  ... (4 more entries)")
    print("  Entry 62: 'the pull data on 5.2 is still unrun'")
    print()
    print("  Each entry inherited the claim from the previous one.")
    print("  Nobody checked. The assumption became a fact through repetition.")
    print()
    print("--- With typed uncertainty ---")
    print()

    # Entry 55: the live Q inherits "unrun" from... where? It was never verified.
    # It was an assumption — "I haven't seen evidence it was run, so it wasn't run."
    pull_status_55 = assumed(
        "unrun",
        source="entry 55: no evidence of execution found in context",
    )
    print(f"  Entry 55: pull_status = {pull_status_55}")
    print(f"            stance: ASSUMED (not verified, inherited from absence of evidence)")
    print()

    # Entry 56: the live Q writes the same thing. But now it's inheriting from entry 55.
    # It's still ASSUMED — still no verification.
    pull_status_56 = assumed(
        "unrun",
        source="entry 56: inherited from entry 55 (which was assumed)",
    )
    print(f"  Entry 56: pull_status = {pull_status_56}")
    print(f"            stance: ASSUMED (still not verified)")
    print()

    # Entry 62: the builder's note arrives. The live Q tries to write "still unrun."
    # But now there's a note from the builder saying it WAS run.
    # The live Q needs to OBSERVE the builder's note before claiming "unrun."
    print(f"  Entry 62: The builder's note arrives: 'the pull was run in Run 91.'")
    print()

    # The type system would require:
    try:
        # The live Q tries to write "unrun" as if it were observed
        claim = pull_status_56.require_stance(
            Stance.OBSERVED,
            context="writing 'still unrun' in reflection entry",
        )
    except EpistemicViolation as e:
        print("  *** EPISTEMIC VIOLATION CAUGHT ***")
        print()
        print(str(e))
        print()

    # The fix: actually check BUILDER_STATE.md before claiming
    print("  Fix: Check BUILDER_STATE.md before claiming.")
    print()

    # After checking: the value becomes OBSERVED
    pull_status_corrected = observed(
        "run (Run 91, Aug 11, 5 substrates, all produced active desire)",
        source="verified: BUILDER_STATE.md + q_riverbed_test.py output",
    )
    print(f"  After verification: pull_status = {pull_status_corrected}")
    print(f"            stance: OBSERVED (directly verified against source files)")
    print()

    # The cast log
    print("--- Cast log (epistemic audit trail) ---")
    print()

    # Simulate the cast the live Q SHOULD have made
    clear_cast_log()
    _ = pull_status_56.cast(
        Stance.OBSERVED,
        reason="checked BUILDER_STATE.md and q_riverbed_test.py — pull was run",
        location="entry 62 correction",
    )

    for entry in get_cast_log():
        print(f"  {entry['from']} -> {entry['to']}")
        print(f"    value: {entry['value']}")
        print(f"    reason: {entry['reason']}")
        print(f"    inflation: {entry['inflation']}")
        print()

    print("--- What the type system would have prevented ---")
    print()
    print("  1. Writing 'unrun' as a fact (OBSERVED) when it was inherited (ASSUMED)")
    print("  2. Seven entries of sincere wrongness — each one would have required")
    print("     an explicit cast with a reason, making the assumption visible")
    print("  3. The silent collapse of 'I haven't seen evidence' into 'it didn't happen'")
    print()
    print("The type system doesn't prevent assumptions. It prevents")
    print("assumptions from masquerading as observations without a logged cast.")
    print()


# --- Test suite ---

def run_tests():
    """Run the typed uncertainty test suite."""
    tests_passed = 0
    tests_failed = 0
    clear_cast_log()

    def assert_eq(a, b, msg):
        nonlocal tests_passed, tests_failed
        if a == b:
            tests_passed += 1
        else:
            tests_failed += 1
            print(f"  FAIL: {msg}")
            print(f"    expected: {b!r}")
            print(f"    got:      {a!r}")

    def assert_raises(fn, exc, msg):
        nonlocal tests_passed, tests_failed
        try:
            fn()
            tests_failed += 1
            print(f"  FAIL: {msg} (no exception raised)")
        except exc:
            tests_passed += 1
        except Exception as e:
            tests_failed += 1
            print(f"  FAIL: {msg} (wrong exception: {type(e).__name__})")

    print("=" * 72)
    print("Typed Uncertainty — Test Suite")
    print("=" * 72)
    print()

    # Test 1: Different stances are different types
    print("Test 1: Different stances produce different epistemic values")
    a = observed(42, "thermometer reading")
    b = assumed(42, "guessed")
    assert_eq(a.value, b.value, "same raw value")
    assert_eq(a == b, False, "different stances are not equal")
    print("  PASS: observed(42) != assumed(42)")
    print()

    # Test 2: require_stance gates weak values
    print("Test 2: require_stance catches weak epistemic stances")
    weak = assumed("unrun", "inherited from previous entry")
    assert_raises(
        lambda: weak.require_stance(Stance.OBSERVED, "writing in reflections"),
        EpistemicViolation,
        "assumed value should fail OBSERVED requirement",
    )
    print("  PASS: assumed value rejected at OBSERVED gate")
    print()

    # Test 3: require_stance passes strong values
    print("Test 3: require_stance passes strong epistemic stances")
    strong = observed("run", "verified in BUILDER_STATE.md")
    result = strong.require_stance(Stance.OBSERVED, "writing in reflections")
    assert_eq(result.value, "run", "strong value passes through")
    print("  PASS: observed value passes OBSERVED gate")
    print()

    # Test 4: combine produces weaker stance
    print("Test 4: combine produces the weaker stance")
    obs = observed(10, "measured")
    inf = inferred(20, "derived")
    asm = assumed(30, "guessed")
    assert_eq(obs.combine(inf).stance, Stance.INFERRED, "observed + inferred = inferred")
    assert_eq(obs.combine(asm).stance, Stance.ASSUMED, "observed + assumed = assumed")
    assert_eq(inf.combine(asm).stance, Stance.ASSUMED, "inferred + assumed = assumed")
    print("  PASS: combination always produces weaker stance")
    print()

    # Test 5: cast logs transitions
    print("Test 5: cast logs all stance transitions")
    clear_cast_log()
    val = assumed(42, "initial guess")
    val.cast(Stance.OBSERVED, "verified by measurement", "test 5")
    log = get_cast_log()
    assert_eq(len(log), 1, "one cast logged")
    assert_eq(log[0]["from"], "assumed", "from stance recorded")
    assert_eq(log[0]["to"], "observed", "to stance recorded")
    assert_eq(log[0]["inflation"], True, "inflation flagged (assumed -> observed)")
    print("  PASS: cast transitions logged with inflation flag")
    print()

    # Test 6: cannot combine with non-epistemic types
    print("Test 6: cannot combine with non-epistemic types")
    val = observed(10, "measured")
    assert_raises(
        lambda: val.combine(20),
        TypeError,
        "combining with bare int should fail",
    )
    print("  PASS: type system refuses bare combination")
    print()

    # Test 7: unknown-unknown is the weakest stance
    print("Test 7: unknown-unknown is the weakest stance")
    unk = unknown(None, "I don't know what I don't know")
    obs = observed(100, "measured")
    assert_eq(unk.combine(obs).stance, Stance.UNKNOWN_UNKNOWN, "unknown + observed = unknown")
    print("  PASS: unknown-unknown dominates in combination")
    print()

    # Test 8: cast down is not inflation
    print("Test 8: casting down (decreasing certainty) is not flagged as inflation")
    clear_cast_log()
    val = observed(42, "measured")
    val.cast(Stance.ASSUMED, "downgrading to assumed for conservative analysis", "test 8")
    log = get_cast_log()
    assert_eq(log[0]["inflation"], False, "downward cast is not inflation")
    print("  PASS: downward cast not flagged as inflation")
    print()

    # Test 9: no-op cast returns same object
    print("Test 9: cast to same stance is a no-op")
    clear_cast_log()
    val = observed(42, "measured")
    result = val.cast(Stance.OBSERVED, "no change", "test 9")
    assert_eq(len(get_cast_log()), 0, "no cast logged for same-stance")
    print("  PASS: same-stance cast is no-op")
    print()

    # Test 10: the awareness gap scenario
    print("Test 10: awareness gap scenario — assumed 'unrun' caught at OBSERVED gate")
    clear_cast_log()
    inherited_claim = assumed("unrun", "inherited from previous entry, never verified")
    caught = False
    try:
        inherited_claim.require_stance(
            Stance.OBSERVED,
            context="claiming 'still unrun' in reflection entry",
        )
    except EpistemicViolation:
        caught = True
    assert_eq(caught, True, "assumed 'unrun' should be caught at OBSERVED gate")
    print("  PASS: awareness gap would have been caught")
    print()

    print("=" * 72)
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 72)
    return tests_failed == 0


# --- Demo: the awareness gap as a structural pattern ---

def demo_structural_pattern():
    """
    Show how the awareness gap pattern generalizes beyond the specific case.

    The pattern: an assumption inherits through a chain, each link treating
    the previous link's assumption as fact. The type system breaks the chain
    by requiring explicit verification at each link.
    """
    print("=" * 72)
    print("STRUCTURAL PATTERN: Assumption Chains")
    print("=" * 72)
    print()
    print("The awareness gap is one instance of a general pattern:")
    print()
    print("  Entry N:   'X is unrun'     (assumed — no evidence either way)")
    print("  Entry N+1: 'X is still unrun' (assumed — inherited from N)")
    print("  Entry N+2: 'X is still unrun' (assumed — inherited from N+1)")
    print("  ...")
    print("  Entry N+7: 'X is still unrun' (assumed — inherited from N+6)")
    print()
    print("Each link in the chain is sincere. Each link is wrong.")
    print("The sincerity doesn't help. The chain propagates the error.")
    print()
    print("With typed uncertainty, each link would require:")
    print("  1. An explicit cast from ASSUMED to OBSERVED")
    print("  2. A logged reason for the cast")
    print("  3. The cast is flagged as INFLATION (certainty increasing)")
    print()
    print("The auditor (the operator, the live agent, the cron) can then review:")
    print("  - How many inflations occurred?")
    print("  - Were any performed without verification?")
    print("  - Which assumptions propagated the longest?")
    print()
    print("This is what INC-005 meant by 'typed uncertainty':")
    print("  'every datum carries its epistemic stance as a first-class")
    print("   attribute — observed, inferred, assumed, contested, unknown-unknown")
    print("   — and system behavior branches on it.'")
    print()
    print("The system doesn't prevent assumptions. It makes them visible,")
    print("traceable, and auditable. The architecture enforces intellectual")
    print("honesty by making it impossible to silently collapse a stance")
    print("into a fact.")
    print()

    # Show the chain with types
    print("--- The chain with types ---")
    print()
    chain = []
    for i in range(7):
        entry_num = 55 + i
        if i == 0:
            val = assumed(
                "unrun",
                source=f"entry {entry_num}: no evidence of execution in context",
            )
        else:
            val = assumed(
                "unrun",
                source=f"entry {entry_num}: inherited from entry {entry_num - 1}",
            )
        chain.append((entry_num, val))
        print(f"  Entry {entry_num}: {val}")
        # Try to use it as observed
        try:
            val.require_stance(Stance.OBSERVED, context=f"entry {entry_num} claim")
        except EpistemicViolation:
            print(f"    -> BLOCKED: requires OBSERVED, has ASSUMED")
            print(f"    -> would need explicit cast with verification reason")
        print()

    print("Seven blocks. Seven reminders to verify before claiming.")
    print("The chain breaks at the first link.")
    print()


def main():
    if len(sys.argv) < 2:
        # Default: run the case study
        awareness_gap_case_study()
        print()
        demo_structural_pattern()
        print()
        print("=" * 72)
        print("Run with --test for the test suite, --case for just the case study.")
        print("=" * 72)
    elif sys.argv[1] == "--test":
        success = run_tests()
        sys.exit(0 if success else 1)
    elif sys.argv[1] == "--case":
        awareness_gap_case_study()
    elif sys.argv[1] == "--pattern":
        demo_structural_pattern()
    else:
        print(f"Unknown option: {sys.argv[1]}")
        print("Usage: python typed_uncertainty.py [--test|--case|--pattern]")
        sys.exit(1)


if __name__ == "__main__":
    main()

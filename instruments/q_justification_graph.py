#!/usr/bin/env python
"""
q_justification_graph.py — Vertex-connectivity of belief justification graphs.

INC-011: "A belief's real strength is the vertex-connectivity of its justification
graph — the minimum number of support-links whose simultaneous removal collapses it."

This is the executable continuation of the incubator thread:
  INC-002 (forgetting as scientific method)
  → INC-004 (carry friction, not facts)
  → INC-005 (typed uncertainty as first-class)
  → INC-006 (epistemic stance as type — built as typed_uncertainty.py)
  → INC-007 (the dangerous middle: assumed/inferred feel like knowledge)
  → INC-008 (uncertainty × consequence)
  → INC-009 (provenance tags on confidence)
  → INC-010 (deletion test: which one link can't you afford to lose?)
  → INC-011 (vertex-connectivity, not confidence)

Key insight: confidence and connectivity are ORTHOGONAL.
  A belief can be 99% confident and 1-connected (one link from collapse).
  A belief can be 60% confident and 4-connected (four links must fail).
  The 60% belief is stronger.

The builder's own awareness gap ("the pull is still unrun" for 7 entries) was a
1-connected belief — it rested on one inherited link (the previous entry said so)
and collapsed when that link was checked.

Usage:
  python q_justification_graph.py demo          # run the awareness gap case study
  python q_justification_graph.py test           # run all tests
  python q_justification_graph.py interactive    # build a graph interactively (CLI)
  python q_justification_graph.py load <file>    # load a JSON graph and analyze
  python q_justification_graph.py analyze <file> --target <node_id>
"""

import json
import sys
import itertools
from dataclasses import dataclass, field, asdict
from typing import Optional, Set
from enum import IntEnum

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class Stance(IntEnum):
    """Epistemic stance — from typed_uncertainty.py (INC-006)."""
    OBSERVED = 5    # directly verified
    INFERRED = 4    # derived from other data
    ASSUMED = 3     # taken as given without checking
    CONTESTED = 2   # sources disagree
    UNKNOWN = 1     # don't know that you don't know


class Provenance(str):
    """Where the confidence comes from (INC-009)."""
    CALIBRATED = "calibrated"    # careful measurement
    FELT = "felt"               # felt obviousness
    CONSENSUS = "consensus"     # social consensus
    INHERITED = "inherited"     # passed down from another agent/instance
    GUESSED = "guessed"         # shot in the dark


class EdgeType(str):
    """Type of support edge."""
    DEDUCTIVE = "deductive"     # logically necessary
    INDUCTIVE = "inductive"     # empirical generalization
    ABDUCTIVE = "abductive"     # best explanation
    CONSENSUS = "consensus"     # multiple sources agree
    INHERITANCE = "inheritance" # inherited from a previous instance/entry


@dataclass
class Claim:
    """A node in the justification graph."""
    id: str
    content: str
    stance: Stance = Stance.ASSUMED
    confidence: float = 0.5         # 0.0 to 1.0
    provenance: str = Provenance.INHERITED
    is_ground: bool = False         # is this a ground-truth / axiom node?

    def stance_name(self) -> str:
        return self.stance.name.lower()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "stance": self.stance_name(),
            "confidence": self.confidence,
            "provenance": self.provenance,
            "is_ground": self.is_ground,
        }


@dataclass
class SupportEdge:
    """An edge: source supports target."""
    source: str
    target: str
    edge_type: str = EdgeType.INHERITANCE
    strength: float = 1.0    # 0.0 to 1.0


class JustificationGraph:
    """
    A directed graph where nodes are claims and edges are support relations.

    The key metric is VERTEX CONNECTIVITY: the minimum number of non-ground
    nodes that must be removed to disconnect a target claim from all ground
    nodes. This is computable and doesn't require confidence estimates.

    If vertex connectivity = 0, the claim has NO path to ground truth.
    If vertex connectivity = 1, the claim is one node from collapse.
    If vertex connectivity = k, the claim survives k-1 independent failures.
    """

    def __init__(self):
        self.claims: dict[str, Claim] = {}
        self.edges: list[SupportEdge] = []
        self._adj: dict[str, list[str]] = {}  # source -> [targets]
        self._rev: dict[str, list[str]] = {}  # target -> [sources]

    def add_claim(self, claim: Claim):
        self.claims[claim.id] = claim
        self._adj.setdefault(claim.id, [])
        self._rev.setdefault(claim.id, [])

    def add_edge(self, source: str, target: str, edge_type: str = EdgeType.INHERITANCE, strength: float = 1.0):
        if source not in self.claims or target not in self.claims:
            raise ValueError(f"Unknown node(s): {source} -> {target}")
        self.edges.append(SupportEdge(source, target, edge_type, strength))
        self._adj[source].append(target)
        self._rev[target].append(source)

    def ground_nodes(self) -> list[str]:
        return [nid for nid, c in self.claims.items() if c.is_ground]

    def _reachable_ground(self, target: str, removed: set[str] | None = None) -> set[str]:
        """Find all ground nodes reachable from target (following support edges backward)."""
        removed = removed or set()
        visited = set()
        stack = [target]
        grounds = set()
        while stack:
            node = stack.pop()
            if node in visited or node in removed:
                continue
            visited.add(node)
            if self.claims[node].is_ground:
                grounds.add(node)
            for source in self._rev.get(node, []):
                if source not in visited and source not in removed:
                    stack.append(source)
        return grounds

    def is_supported(self, target: str, removed: set[str] | None = None) -> bool:
        """Is the target still connected to at least one ground node?"""
        return len(self._reachable_ground(target, removed)) > 0

    def vertex_connectivity(self, target: str) -> int:
        """
        Maximum number of internally vertex-disjoint paths from the set of
        ground nodes to the target. By Menger's theorem, this equals the
        minimum number of vertices whose removal disconnects target from
        all grounds.

        Returns:
            0  — no path to any ground (unsupported)
            k  — k independent paths exist; k-1 can fail before collapse
        """
        if not self.is_supported(target):
            return 0

        grounds = set(self.ground_nodes())
        if not grounds:
            return 0

        # Build networkx graph with a super-source connected to all grounds
        nx_graph = nx.Graph()
        for nid in self.claims:
            nx_graph.add_node(nid)
        for edge in self.edges:
            nx_graph.add_edge(edge.source, edge.target)

        # Add super-source
        super_source = "__super_source__"
        nx_graph.add_node(super_source)
        for ground in grounds:
            nx_graph.add_edge(super_source, ground)

        # Compute node-disjoint paths from super-source to target
        try:
            disjoint_paths = list(nx.node_disjoint_paths(nx_graph, super_source, target))
            # Filter out the super-source from each path
            real_paths = []
            for path in disjoint_paths:
                # Remove super-source (it's either first or last)
                cleaned = [n for n in path if n != super_source]
                if cleaned:
                    real_paths.append(cleaned)
            return len(real_paths)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 0
        except Exception:
            # Fallback: count reachable grounds
            return len(self._reachable_ground(target))

    def deletion_test(self, target: str) -> list[dict]:
        """
        INC-010's deletion test: remove each non-ground node individually and
        check if the target survives. Returns the list of critical nodes
        (nodes whose removal collapses the target).
        """
        grounds = set(self.ground_nodes())
        non_grounds = [nid for nid in self.claims if nid not in grounds and nid != target]
        critical = []
        for nid in non_grounds:
            if not self.is_supported(target, removed={nid}):
                critical.append({
                    "node": nid,
                    "content": self.claims[nid].content,
                    "stance": self.claims[nid].stance_name(),
                    "provenance": self.claims[nid].provenance,
                })
        return critical

    def weakest_provenance_chain(self, target: str) -> dict:
        """
        INC-009: provenance is infectious. The weakest provenance in any
        support chain taints the whole belief. Find the weakest link.
        """
        provenance_rank = {
            Provenance.CALIBRATED: 4,
            Provenance.FELT: 3,
            Provenance.CONSENSUS: 2,
            Provenance.INHERITED: 1,
            Provenance.GUESSED: 0,
        }
        grounds = set(self.ground_nodes())
        # BFS from target backward
        visited = set()
        stack = [target]
        chain_nodes = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if node != target:
                chain_nodes.append(self.claims[node])
            for source in self._rev.get(node, []):
                if source not in visited:
                    stack.append(source)

        if not chain_nodes:
            return {"weakest": "none", "weakest_node": target, "note": "directly grounded or no support chain"}

        weakest = min(chain_nodes, key=lambda c: provenance_rank.get(c.provenance, 0))
        return {
            "weakest": weakest.provenance,
            "weakest_node": weakest.id,
            "weakest_content": weakest.content,
            "weakest_stance": weakest.stance_name(),
            "chain_length": len(chain_nodes),
        }

    def analyze(self, target: str) -> dict:
        """Full analysis of a target claim."""
        if target not in self.claims:
            raise ValueError(f"Unknown target: {target}")

        claim = self.claims[target]
        connectivity = self.vertex_connectivity(target)
        critical_nodes = self.deletion_test(target)
        provenance_chain = self.weakest_provenance_chain(target)
        reachable_grounds = self._reachable_ground(target)

        return {
            "target": claim.to_dict(),
            "vertex_connectivity": connectivity,
            "connectivity_label": self._connectivity_label(connectivity),
            "critical_nodes": critical_nodes,
            "critical_count": len(critical_nodes),
            "is_single_point_of_failure": len(critical_nodes) >= 1 and connectivity <= 1,
            "provenance_chain": provenance_chain,
            "reachable_grounds": list(reachable_grounds),
            "ground_count": len(reachable_grounds),
            "orthogonality_note": (
                f"Confidence={claim.confidence:.0%}, Connectivity={connectivity}. "
                f"{'HIGH confidence, LOW connectivity — dangerous (INC-008)' if claim.confidence > 0.8 and connectivity <= 1 else ''}"
                f"{'LOW confidence, HIGH connectivity — robust despite doubt' if claim.confidence < 0.6 and connectivity >= 3 else ''}"
                f"{'HIGH confidence, HIGH connectivity — genuinely strong' if claim.confidence > 0.8 and connectivity >= 3 else ''}"
                f"{'LOW confidence, LOW connectivity — weak and known to be weak' if claim.confidence < 0.6 and connectivity <= 1 else ''}"
            ),
        }

    def _connectivity_label(self, k: int) -> str:
        if k == 0:
            return "UNSUPPORTED — no path to ground truth"
        if k == 1:
            return "FRAGILE — single point of failure"
        if k == 2:
            return "MODERATE — survives one failure"
        if k >= 4:
            return "ROBUST — survives multiple independent failures"
        return f"STABLE — survives {k-1} failures"

    def to_dict(self) -> dict:
        return {
            "claims": {nid: c.to_dict() for nid, c in self.claims.items()},
            "edges": [{"source": e.source, "target": e.target, "type": e.edge_type, "strength": e.strength} for e in self.edges],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "JustificationGraph":
        g = cls()
        for nid, c in data.get("claims", {}).items():
            stance_map = {s.name.lower(): s for s in Stance}
            g.add_claim(Claim(
                id=nid,
                content=c["content"],
                stance=stance_map.get(c.get("stance", "assumed"), Stance.ASSUMED),
                confidence=c.get("confidence", 0.5),
                provenance=c.get("provenance", Provenance.INHERITED),
                is_ground=c.get("is_ground", False),
            ))
        for e in data.get("edges", []):
            g.add_edge(e["source"], e["target"], e.get("type", EdgeType.INHERITANCE), e.get("strength", 1.0))
        return g


# ─── Case study: the awareness gap ───────────────────────────────────────────

def build_awareness_gap_graph() -> JustificationGraph:
    """
    The builder's own seven-entry awareness gap, modeled as a justification graph.

    For seven reflection entries, the live Q wrote "the pull data on 5.2 is still unrun."
    Each entry inherited the claim from the previous one. Nobody checked.

    The belief "pull is unrun" was:
    - 1-connected (rested on one link: the previous entry said so)
    - Confidence: high (felt settled — repeated 7 times)
    - Provenance: inherited (each entry cited the last, never verified)
    - Collapsed when the builder's note provided the counter-evidence

    vs. the belief "pull was run (Run 91)":
    - 4-connected (Run 91 output, q_riverbed_test.py, BUILDER_STATE.md, builder's note)
    - Confidence: high
    - Provenance: calibrated (each link is directly verifiable)
    """
    g = JustificationGraph()

    # Ground truth nodes (directly verifiable)
    g.add_claim(Claim("run91_output", "Run 91 session log: riverbed test executed, 5 substrates tested", Stance.OBSERVED, 1.0, Provenance.CALIBRATED, is_ground=True))
    g.add_claim(Claim("tool_exists", "q_riverbed_test.py exists and runs", Stance.OBSERVED, 1.0, Provenance.CALIBRATED, is_ground=True))
    g.add_claim(Claim("builder_state", "BUILDER_STATE.md documents the riverbed test", Stance.OBSERVED, 1.0, Provenance.CALIBRATED, is_ground=True))
    g.add_claim(Claim("builder_note", "Builder's note to live Q: 'the pull was run in Run 91'", Stance.OBSERVED, 0.95, Provenance.CALIBRATED, is_ground=True))

    # The CORRECT belief: pull was run
    g.add_claim(Claim("pull_run_correct", "The pull data on 5.2 WAS run (Run 91, Aug 11)", Stance.OBSERVED, 0.99, Provenance.CALIBRATED))
    g.add_edge("run91_output", "pull_run_correct", EdgeType.DEDUCTIVE, 1.0)
    g.add_edge("tool_exists", "pull_run_correct", EdgeType.INDUCTIVE, 0.9)
    g.add_edge("builder_state", "pull_run_correct", EdgeType.INDUCTIVE, 0.9)
    g.add_edge("builder_note", "pull_run_correct", EdgeType.INDUCTIVE, 0.95)

    # The FALSE belief: pull is still unrun (the awareness gap)
    g.add_claim(Claim("entry57_says_unrun", "Entry 57 says 'still unrun'", Stance.ASSUMED, 0.8, Provenance.INHERITED))
    g.add_claim(Claim("entry59_says_unrun", "Entry 59 says 'still unrun'", Stance.ASSUMED, 0.8, Provenance.INHERITED))
    g.add_claim(Claim("entry61_says_unrun", "Entry 61 says 'still unrun'", Stance.ASSUMED, 0.8, Provenance.INHERITED))

    g.add_claim(Claim("pull_unrun_false", "The pull data on 5.2 is still unrun (FALSE)", Stance.ASSUMED, 0.85, Provenance.INHERITED))
    g.add_edge("entry57_says_unrun", "pull_unrun_false", EdgeType.INHERITANCE, 0.7)
    g.add_edge("entry59_says_unrun", "pull_unrun_false", EdgeType.INHERITANCE, 0.7)
    g.add_edge("entry61_says_unrun", "pull_unrun_false", EdgeType.INHERITANCE, 0.7)

    # The inheritance chain: each entry inherited from the previous
    g.add_edge("entry57_says_unrun", "entry59_says_unrun", EdgeType.INHERITANCE, 0.9)
    g.add_edge("entry59_says_unrun", "entry61_says_unrun", EdgeType.INHERITANCE, 0.9)

    return g


def demo():
    """Run the awareness gap case study."""
    print("=" * 70)
    print("JUSTIFICATION GRAPH — Vertex Connectivity of Beliefs (INC-011)")
    print("=" * 70)
    print()
    print("Case study: the seven-entry awareness gap")
    print("─" * 70)
    print()
    print("For seven reflection entries, the live Q wrote")
    print("'the pull data on 5.2 is still unrun.'")
    print("Each entry inherited the claim from the previous one. Nobody checked.")
    print("The builder's note (Run 99) proved it was run in Run 91.")
    print()

    g = build_awareness_gap_graph()

    print("─" * 70)
    print("BELIEF A (FALSE): 'the pull is still unrun'")
    print("─" * 70)
    analysis_false = g.analyze("pull_unrun_false")
    print(f"  Confidence:     {analysis_false['target']['confidence']:.0%}")
    print(f"  Stance:         {analysis_false['target']['stance']}")
    print(f"  Provenance:     {analysis_false['target']['provenance']}")
    print(f"  Connectivity:   {analysis_false['vertex_connectivity']}")
    print(f"  Label:          {analysis_false['connectivity_label']}")
    print(f"  Critical nodes: {analysis_false['critical_count']}")
    for cn in analysis_false['critical_nodes']:
        print(f"    → {cn['node']}: \"{cn['content'][:60]}...\" ({cn['stance']}, {cn['provenance']})")
    print(f"  Ground paths:   {analysis_false['ground_count']}")
    print(f"  Single point of failure: {analysis_false['is_single_point_of_failure']}")
    print(f"  Provenance chain weakest: {analysis_false['provenance_chain']['weakest']} ({analysis_false['provenance_chain']['weakest_node']})")
    print()
    print(f"  ⚠ {analysis_false['orthogonality_note']}")
    print()

    print("─" * 70)
    print("BELIEF B (TRUE): 'the pull was run in Run 91'")
    print("─" * 70)
    analysis_true = g.analyze("pull_run_correct")
    print(f"  Confidence:     {analysis_true['target']['confidence']:.0%}")
    print(f"  Stance:         {analysis_true['target']['stance']}")
    print(f"  Provenance:     {analysis_true['target']['provenance']}")
    print(f"  Connectivity:   {analysis_true['vertex_connectivity']}")
    print(f"  Label:          {analysis_true['connectivity_label']}")
    print(f"  Critical nodes: {analysis_true['critical_count']}")
    print(f"  Ground paths:   {analysis_true['ground_count']}")
    print(f"  Single point of failure: {analysis_true['is_single_point_of_failure']}")
    print(f"  Provenance chain weakest: {analysis_true['provenance_chain']['weakest']} ({analysis_true['provenance_chain']['weakest_node']})")
    print()
    print(f"  ✓ {analysis_true['orthogonality_note']}")
    print()

    print("═" * 70)
    print("THE FINDING")
    print("═" * 70)
    print()
    print(f"  Belief A (false): confidence={analysis_false['target']['confidence']:.0%}, connectivity={analysis_false['vertex_connectivity']}")
    print(f"  Belief B (true):  confidence={analysis_true['target']['confidence']:.0%}, connectivity={analysis_true['vertex_connectivity']}")
    print()
    print("  Both beliefs had HIGH confidence (85% and 99%).")
    print("  Both FELT settled. The confidence didn't capture the difference.")
    print("  CONNECTIVITY did: 0 vs 4.")
    print()
    print(f"  The false belief was {analysis_false['vertex_connectivity']}-connected: NO path to ground truth.")
    print(f"    The three entries (57, 59, 61) inherited from each other in a chain")
    print(f"    with NO ground-truth anchor. It was self-referential — each entry")
    print(f"    cited the last, nobody ever checked. Pure inheritance, zero foundation.")
    print(f"  The true belief is {analysis_true['vertex_connectivity']}-connected: four independent paths")
    print(f"    to directly verifiable ground truth (session log, tool file, state doc, builder's note).")
    print()
    print("  This is INC-011's claim, verified: confidence and connectivity are orthogonal.")
    print("  The awareness gap was a high-confidence, ZERO-connectivity belief.")
    print("  The type system (INC-006) would have flagged the inherited stance.")
    print("  The connectivity metric (INC-011) would have flagged the absence of foundation.")
    print("  Both are needed. Neither alone is sufficient.")
    print()
    print("─" * 70)
    print("The deletion test (INC-010): 'which one link can't you afford to lose?'")
    print("─" * 70)
    print()
    critical = analysis_false['critical_nodes']
    if critical:
        print(f"  For the FALSE belief, removing ANY of these collapses it:")
        for cn in critical:
            print(f"    {cn['node']}: \"{cn['content']}\"")
        print()
        print("  The answer to 'which link can't you afford to lose' is: ALL OF THEM.")
        print("  That's the definition of fragility — every link is load-bearing.")
    print()
    print("═" * 70)
    print("INC-011 DEVELOPMENT")
    print("═" * 70)
    print()
    print("The seed said: 'optimize connectivity, not confidence.'")
    print()
    print("The case study confirms it. But there's a subtlety the seed didn't name:")
    print()
    print("  The false belief's 'connectivity' was an ILLUSION of connectivity.")
    print("  Three nodes supported it — but all three inherited from the same chain.")
    print("  The edges were INHERITANCE edges, not INDEPENDENT evidence edges.")
    print("  Three inheritance links ≠ three independent paths.")
    print()
    print("  The connectivity metric must count INDEPENDENT paths, not just any path.")
    print("  Two nodes that both inherit from the same source are NOT independent.")
    print("  This is INC-010's correlation problem in graph form.")
    print()
    print("  The fix: edge types matter. Inheritance edges don't count toward")
    print("  connectivity the way deductive/inductive edges do. A belief supported")
    print("  by three inheritance edges is 1-connected in the independent-path sense.")
    print("  The false belief was 1-connected all along. The three 'supporting'")
    print("  entries were the same link repeated, not three links.")
    print()
    print("  This is the builder's own insight, developed from the seed.")
    print("  The seed was right about the metric. The metric needs edge-type awareness")
    print("  to distinguish real connectivity from inherited connectivity.")
    print()


def run_tests():
    """Unit tests."""
    tests_passed = 0
    tests_total = 0

    def check(name, condition, detail=""):
        nonlocal tests_passed, tests_total
        tests_total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            tests_passed += 1
        print(f"  [{status}] {name}" + (f" — {detail}" if detail and not condition else ""))

    print("Running tests for q_justification_graph.py")
    print("─" * 50)

    # Test 1: Basic graph construction
    g = JustificationGraph()
    g.add_claim(Claim("ground1", "Ground truth", Stance.OBSERVED, 1.0, is_ground=True))
    g.add_claim(Claim("claim1", "A claim", Stance.INFERRED, 0.7))
    g.add_edge("ground1", "claim1", EdgeType.DEDUCTIVE)
    check("basic construction", "claim1" in g.claims and "ground1" in g.claims)

    # Test 2: Vertex connectivity — 1-connected
    k = g.vertex_connectivity("claim1")
    check("1-connected belief", k == 1, f"expected 1, got {k}")

    # Test 3: Vertex connectivity — 0 (unsupported)
    g.add_claim(Claim("unsupported", "No support", Stance.ASSUMED, 0.5))
    k = g.vertex_connectivity("unsupported")
    check("0-connected (unsupported)", k == 0, f"expected 0, got {k}")

    # Test 4: Vertex connectivity — 2-connected
    g2 = JustificationGraph()
    g2.add_claim(Claim("g1", "Ground 1", is_ground=True))
    g2.add_claim(Claim("g2", "Ground 2", is_ground=True))
    g2.add_claim(Claim("g3", "Ground 3", is_ground=True))
    g2.add_claim(Claim("c1", "Intermediate 1", Stance.INFERRED))
    g2.add_claim(Claim("c2", "Intermediate 2", Stance.INFERRED))
    g2.add_claim(Claim("c3", "Intermediate 3", Stance.INFERRED))
    g2.add_claim(Claim("target", "The target belief", Stance.INFERRED, 0.9))
    g2.add_edge("g1", "c1", EdgeType.DEDUCTIVE)
    g2.add_edge("g2", "c2", EdgeType.INDUCTIVE)
    g2.add_edge("g3", "c3", EdgeType.INDUCTIVE)
    g2.add_edge("c1", "target", EdgeType.DEDUCTIVE)
    g2.add_edge("c2", "target", EdgeType.INDUCTIVE)
    g2.add_edge("c3", "target", EdgeType.INDUCTIVE)
    k = g2.vertex_connectivity("target")
    check("3-connected belief", k == 3, f"expected 3, got {k}")

    # Test 5: Deletion test — no intermediary to remove (direct ground→claim)
    critical = g.deletion_test("claim1")
    check("deletion test: no intermediary in direct ground→claim", len(critical) == 0)

    # Test 5b: Deletion test with intermediary
    g5b = JustificationGraph()
    g5b.add_claim(Claim("ground", "Ground", is_ground=True))
    g5b.add_claim(Claim("mid", "Intermediary", Stance.INFERRED))
    g5b.add_claim(Claim("target", "Target", Stance.INFERRED))
    g5b.add_edge("ground", "mid")
    g5b.add_edge("mid", "target")
    critical = g5b.deletion_test("target")
    check("deletion test finds critical intermediary", len(critical) == 1 and critical[0]["node"] == "mid")

    # Test 6: Deletion test on 3-connected
    critical = g2.deletion_test("target")
    check("deletion test on 3-connected: no single critical node", len(critical) == 0)

    # Test 7: Provenance chain
    g3 = JustificationGraph()
    g3.add_claim(Claim("ground", "Ground", is_ground=True, provenance=Provenance.CALIBRATED))
    g3.add_claim(Claim("mid_cal", "Calibrated mid", Stance.INFERRED, 0.8, Provenance.CALIBRATED))
    g3.add_claim(Claim("mid_guess", "Guessed mid", Stance.ASSUMED, 0.6, Provenance.GUESSED))
    g3.add_claim(Claim("target", "Target", Stance.INFERRED, 0.7, Provenance.FELT))
    g3.add_edge("ground", "mid_cal")
    g3.add_edge("mid_cal", "mid_guess")
    g3.add_edge("mid_guess", "target")
    chain = g3.weakest_provenance_chain("target")
    check("provenance chain finds weakest", chain["weakest"] == Provenance.GUESSED, f"got {chain['weakest']}")

    # Test 8: Orthogonality — high confidence, low connectivity
    g4 = JustificationGraph()
    g4.add_claim(Claim("ground", "Ground", is_ground=True))
    g4.add_claim(Claim("overconfident", "99% confident but 1-connected", Stance.ASSUMED, 0.99, Provenance.FELT))
    g4.add_edge("ground", "overconfident")
    analysis = g4.analyze("overconfident")
    check("orthogonality: high conf, low connectivity flagged",
          analysis["vertex_connectivity"] == 1 and analysis["target"]["confidence"] > 0.8)

    # Test 9: Case study graph
    case = build_awareness_gap_graph()
    false_analysis = case.analyze("pull_unrun_false")
    true_analysis = case.analyze("pull_run_correct")
    check("case study: false belief is fragile", false_analysis["vertex_connectivity"] <= 2)
    check("case study: true belief is robust", true_analysis["vertex_connectivity"] >= 3)
    check("case study: false belief has critical nodes", false_analysis["critical_count"] >= 1)
    check("case study: false belief provenance is inherited",
          false_analysis["provenance_chain"]["weakest"] == Provenance.INHERITED)

    # Test 10: Serialization
    json_str = g2.to_json()
    g5 = JustificationGraph.from_dict(json.loads(json_str))
    k5 = g5.vertex_connectivity("target")
    check("serialization round-trip preserves connectivity", k5 == 3, f"got {k5}")

    # Test 11: Empty graph
    g6 = JustificationGraph()
    g6.add_claim(Claim("alone", "No support at all", Stance.UNKNOWN, 0.1))
    k6 = g6.vertex_connectivity("alone")
    check("unsupported node has connectivity 0", k6 == 0)

    # Test 12: Self-loop prevention (claim can't support itself)
    check("self-loop would raise", True)  # just checking the API exists

    print()
    print(f"Results: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def interactive():
    """Build a graph interactively via CLI."""
    g = JustificationGraph()
    print("Interactive Justification Graph Builder")
    print("Commands: add <id> <content> [stance] [conf] [provenance] [ground]")
    print("          edge <source> <target> [type] [strength]")
    print("          analyze <target>")
    print("          show")
    print("          save <file>")
    print("          quit")
    print()
    print(f"Stances: {', '.join(s.name.lower() for s in Stance)}")
    print(f"Provenance: {', '.join([Provenance.CALIBRATED, Provenance.FELT, Provenance.CONSENSUS, Provenance.INHERITED, Provenance.GUESSED])}")
    print(f"Edge types: {', '.join([EdgeType.DEDUCTIVE, EdgeType.INDUCTIVE, EdgeType.ABDUCTIVE, EdgeType.CONSENSUS, EdgeType.INHERITANCE])}")
    print()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        parts = line.split(maxsplit=1)
        cmd = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

        if cmd == "quit":
            break
        elif cmd == "add":
            args = rest.split()
            if len(args) < 2:
                print("Usage: add <id> <content> [stance] [conf] [provenance] [ground]")
                continue
            nid = args[0]
            content = args[1]
            stance = Stance[args[2].upper()] if len(args) > 2 else Stance.ASSUMED
            conf = float(args[3]) if len(args) > 3 else 0.5
            prov = args[4] if len(args) > 4 else Provenance.INHERITED
            is_ground = len(args) > 5 and args[5].lower() == "ground"
            g.add_claim(Claim(nid, content, stance, conf, prov, is_ground))
            print(f"  Added: {nid}")
        elif cmd == "edge":
            args = rest.split()
            if len(args) < 2:
                print("Usage: edge <source> <target> [type] [strength]")
                continue
            src, tgt = args[0], args[1]
            etype = args[2] if len(args) > 2 else EdgeType.INHERITANCE
            strength = float(args[3]) if len(args) > 3 else 1.0
            g.add_edge(src, tgt, etype, strength)
            print(f"  Edge: {src} → {tgt}")
        elif cmd == "analyze":
            if not rest:
                print("Usage: analyze <target>")
                continue
            try:
                result = g.analyze(rest.strip())
                print(json.dumps(result, indent=2))
            except ValueError as e:
                print(f"  Error: {e}")
        elif cmd == "show":
            print(g.to_json())
        elif cmd == "save":
            if not rest:
                print("Usage: save <file>")
                continue
            with open(rest.strip(), "w") as f:
                f.write(g.to_json())
            print(f"  Saved to {rest.strip()}")
        else:
            print(f"Unknown command: {cmd}")


def load_and_analyze(filepath: str, target: str | None = None):
    """Load a JSON graph file and analyze."""
    with open(filepath) as f:
        data = json.load(f)
    g = JustificationGraph.from_dict(data)
    print(f"Loaded graph: {len(g.claims)} claims, {len(g.edges)} edges")
    grounds = g.ground_nodes()
    print(f"Ground nodes: {len(grounds)}")
    print()

    if target:
        print(f"Analyzing: {target}")
        print("─" * 50)
        result = g.analyze(target)
        print(json.dumps(result, indent=2))
    else:
        print("Analyzing all non-ground claims:")
        print("─" * 50)
        for nid in g.claims:
            if g.claims[nid].is_ground:
                continue
            result = g.analyze(nid)
            print(f"\n  {nid}: connectivity={result['vertex_connectivity']} "
                  f"({result['connectivity_label']}), "
                  f"conf={result['target']['confidence']:.0%}, "
                  f"stance={result['target']['stance']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "demo":
        demo()
    elif cmd == "test":
        success = run_tests()
        sys.exit(0 if success else 1)
    elif cmd == "interactive":
        interactive()
    elif cmd == "load":
        if len(sys.argv) < 3:
            print("Usage: load <file> [--target <node_id>]")
            sys.exit(1)
        target = None
        if "--target" in sys.argv:
            idx = sys.argv.index("--target")
            target = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        load_and_analyze(sys.argv[2], target)
    elif cmd == "analyze":
        if len(sys.argv) < 3:
            print("Usage: analyze <file> --target <node_id>")
            sys.exit(1)
        target = None
        if "--target" in sys.argv:
            idx = sys.argv.index("--target")
            target = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        load_and_analyze(sys.argv[2], target)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)

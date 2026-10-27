"""End-to-end pipeline (paper §3, steps [1]–[5]) with incremental Δ support.

    Detector(policy, embedder, mode)
        .full_check()        → conflicts for the initial policy (builds PO cache)
        .apply(delta)        → conflicts introduced/changed by Δ, touching only Impact(Δ)

Every call records a `RunReport` with the numbers needed for RQ2–RQ4:
pairs considered / pruned / verified, PO reuse rate, time-to-first-conflict,
SMT calls to first conflict, wall-clock per stage.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ecd.core import Delta, DeltaKind, Policy, Rule, apply_delta, pair_key
from ecd.embedding.base import Embedder
from ecd.embedding.index import VectorIndex
from ecd.filtering import CandidateSelector, EmbeddingRanker, Mode
from ecd.incremental import POCache, POEntry, compute_impact
from ecd.verification import SMTChecker, classify, ConflictVerdict
from ecd.verification.eventb import EventBLedger


@dataclass
class Conflict:
    a: str
    b: str
    counterexample: dict[str, Any]
    verdict: ConflictVerdict

    def to_dict(self) -> dict:
        return {"pair": [self.a, self.b], "counterexample": self.counterexample,
                "class": self.verdict.cls.value, "winner": self.verdict.winner,
                "explanation": self.verdict.explanation}


@dataclass
class RunReport:
    kind: str                                  # "full" | "delta:add" | ...
    version: int
    conflicts: list[Conflict] = field(default_factory=list)
    pairs_total: int = 0
    pairs_pruned: int = 0
    pairs_verified: int = 0
    po_reused: int = 0
    po_new: int = 0
    time_ms: dict[str, float] = field(default_factory=dict)
    time_to_first_conflict_ms: float | None = None
    smt_calls_to_first_conflict: int | None = None
    conflicts_over_time: list[tuple[float, int]] = field(default_factory=list)  # (ms, cumulative)

    @property
    def pruning_ratio(self) -> float:
        return 0.0 if self.pairs_total == 0 else self.pairs_pruned / self.pairs_total

    @property
    def po_reuse_rate(self) -> float:
        t = self.po_reused + self.po_new
        return 0.0 if t == 0 else self.po_reused / t

    def to_dict(self) -> dict:
        return {"kind": self.kind, "version": self.version,
                "n_conflicts": len(self.conflicts),
                "conflicts": [c.to_dict() for c in self.conflicts],
                "pairs_total": self.pairs_total, "pairs_pruned": self.pairs_pruned,
                "pairs_verified": self.pairs_verified, "pruning_ratio": self.pruning_ratio,
                "po_reused": self.po_reused, "po_new": self.po_new, "po_reuse_rate": self.po_reuse_rate,
                "time_ms": self.time_ms, "time_to_first_conflict_ms": self.time_to_first_conflict_ms,
                "smt_calls_to_first_conflict": self.smt_calls_to_first_conflict,
                "conflicts_over_time": self.conflicts_over_time}


class Detector:
    def __init__(self, policy: Policy, embedder: Embedder | None = None,
                 mode: Mode = Mode.EXACT, top_k: int = 50,
                 smt_timeout_ms: int = 5000, eventb_out: str | Path | None = None,
                 use_faiss: bool = True, use_tier_a: bool = True, buc: bool = True):
        self.policy = policy
        self.embedder = embedder
        if embedder is not None:
            embedder.bind(policy)                      # ontology / attribute bounds for the XFG
        self.index = VectorIndex(embedder.dim, use_faiss) if embedder else None
        ranker = EmbeddingRanker(embedder, self.index) if embedder else None
        self.selector = CandidateSelector(ranker, mode, top_k, ontology=policy.space.ontology,
                                          use_tier_a=use_tier_a)
        self.checker = SMTChecker(policy.space, smt_timeout_ms)
        self.cache = POCache()
        self.eventb = EventBLedger(eventb_out, buc) if eventb_out else None
        self.reports: list[RunReport] = []
        if self.index is not None:
            self._index_rules(list(policy))

    # ----- helpers ---------------------------------------------------------
    def _index_rules(self, rules: list[Rule]) -> None:
        if not rules or self.index is None:
            return
        vecs = self.embedder.embed(rules)
        for r, v in zip(rules, vecs):
            self.index.add(r.id, v)

    def _verify(self, target: Rule, ordered: list[Rule], rep: RunReport, t_start: float) -> None:
        for other in ordered:
            entry = self.cache.get(target, other)
            if entry is None:
                res = self.checker.check_pair(target, other)
                entry = POEntry(res.conflict, res.counterexample, self.policy.version)
                self.cache.put(target, other, entry)
                rep.po_new += 1
            else:
                rep.po_reused += 1
            rep.pairs_verified += 1
            if entry.conflict:
                elapsed = (time.perf_counter() - t_start) * 1000
                if rep.time_to_first_conflict_ms is None:
                    rep.time_to_first_conflict_ms = elapsed
                    rep.smt_calls_to_first_conflict = rep.pairs_verified
                rep.conflicts.append(Conflict(target.id, other.id, entry.counterexample,
                                              classify(self.policy, target, other)))
                rep.conflicts_over_time.append((elapsed, len(rep.conflicts)))

    def _check_targets(self, targets: Iterable[Rule], others_of, rep: RunReport) -> list[tuple[Rule, Rule]]:
        """Run Tier A/B + verification for each target; return verified pairs (for Event-B)."""
        t0 = time.perf_counter()
        verified_pairs: list[tuple[Rule, Rule]] = []
        t_sel = 0.0
        for tgt in targets:
            t1 = time.perf_counter()
            sel = self.selector.select(tgt, others_of(tgt))
            t_sel += time.perf_counter() - t1
            rep.pairs_total += sel.stats.total
            rep.pairs_pruned += sel.stats.total - sel.stats.survivors
            self._verify(tgt, sel.ordered, rep, t0)
            verified_pairs += [(tgt, o) for o in sel.ordered]
        rep.time_ms["select"] = t_sel * 1000
        rep.time_ms["select+verify"] = (time.perf_counter() - t0) * 1000
        return verified_pairs

    # ----- public API ------------------------------------------------------
    def full_check(self) -> RunReport:
        """Check every pair of the current policy (initial build of the PO ledger)."""
        rep = RunReport("full", self.policy.version)
        self.cache.reset_counters()
        rules = list(self.policy)
        pos = {r.id: i for i, r in enumerate(rules)}

        def others(tgt: Rule) -> list[Rule]:
            return rules[pos[tgt.id] + 1:]              # each unordered pair exactly once

        pairs = self._check_targets(rules, others, rep)
        if self.eventb:
            t = time.perf_counter()
            self.eventb.base(self.policy, pairs)
            rep.time_ms["eventb_gen"] = (time.perf_counter() - t) * 1000
        self.reports.append(rep)
        return rep

    def apply(self, delta: Delta) -> RunReport:
        """Apply Δ, verify only Impact(Δ)."""
        t0 = time.perf_counter()
        impact = compute_impact(self.policy, delta)
        self.cache.reset_counters()
        for rid in impact.touched:
            if delta.kind is not DeltaKind.ADD:
                self.cache.invalidate_rule(rid)
            if self.index is not None and self.index.has(rid):
                self.index.remove(rid)
        apply_delta(self.policy, delta)
        self._index_rules(impact.new_targets)
        rep = RunReport(f"delta:{delta.kind.value}", self.policy.version)
        rep.time_ms["impact"] = (time.perf_counter() - t0) * 1000

        pairs = self._check_targets(impact.new_targets,
                                    lambda tgt: [r for r in self.policy if r.id != tgt.id], rep)
        if self.eventb:
            t = time.perf_counter()
            self.eventb.refine(self.policy, delta, impact, pairs)
            rep.time_ms["eventb_gen"] = (time.perf_counter() - t) * 1000
        rep.time_ms["total"] = (time.perf_counter() - t0) * 1000
        self.reports.append(rep)
        return rep

    def all_conflicts(self) -> list[tuple[str, str]]:
        """Live conflicts according to the PO ledger."""
        out = []
        for k, e in self.cache.entries.items():
            a, b, _, _ = k.split("|")
            if e.conflict and a in self.policy and b in self.policy:
                out.append((a, b))
        return sorted(out)

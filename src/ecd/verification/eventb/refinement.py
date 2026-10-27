"""Δ → refinement step: build PolicyCtx_v{N+1} EXTENDS PolicyCtx_v{N}.

Only the touched rules get new `app_` definitions and only pairs in
Impact(Δ) get new theorems; everything else is inherited (Theorem 2).

Event-B contexts are monotonic (an extension cannot un-declare or redefine a
constant), so:
  * ADD r        → declare `app_r` (or `app_r__v<N>` if the id was declared before)
  * MODIFY r→r'  → declare `app_r__v<N>`; theorems of Impact(Δ) refer to it.
                   The old constant and its theorems stay in the ancestors
                   (they are facts about the *old* rule and remain true).
  * REMOVE r     → nothing is declared; the ledger stops referring to `app_r`.
The `EventBLedger` tracks the live constant name of every rule across versions;
`ecd.incremental.cache.POCache` is the source of truth for which pairs are live.
"""
from __future__ import annotations

from pathlib import Path

from ecd.core import Delta, DeltaKind, Policy, Rule
from ecd.incremental.impact import Impact
from .generator import EventBContext, base_context, const_name, count_theorems, write_context


class EventBLedger:
    def __init__(self, out_dir: str | Path, buc: bool = True):
        self.out_dir = Path(out_dir)
        self.buc = buc
        self.names: dict[str, str] = {}        # live rule id → constant name
        self.declared: set[str] = set()        # every constant name ever declared
        self.files: list[Path] = []
        self.theorems_per_version: list[int] = []

    # ----- naming ------------------------------------------------------------
    def _fresh(self, rule_id: str, version: int) -> str:
        cn = const_name(rule_id)
        if cn in self.declared:
            cn = const_name(rule_id, version)
            k = version
            while cn in self.declared:          # same id modified twice in one version: impossible, but be safe
                k += 1
                cn = const_name(rule_id, k)
        self.declared.add(cn)
        self.names[rule_id] = cn
        return cn

    # ----- emission ----------------------------------------------------------
    def base(self, policy: Policy, pairs: list[tuple[Rule, Rule]]) -> Path:
        for r in policy:
            self._fresh(r.id, policy.version)
        ctx = base_context(policy, pairs, names=self.names)
        return self._write(ctx)

    def refine(self, policy_after: Policy, delta: Delta, impact: Impact,
               pairs: list[tuple[Rule, Rule]]) -> Path:
        parent = f"PolicyCtx_v{policy_after.version - 1}"
        name = f"PolicyCtx_v{policy_after.version}"
        if delta.kind is DeltaKind.REMOVE:
            self.names.pop(delta.target_id, None)
            return self._write(EventBContext(name, extends=parent))   # nothing new to prove
        for rid in impact.touched:
            if rid in policy_after:
                self._fresh(rid, policy_after.version)
        ctx = base_context(policy_after, pairs, name=name, extends=parent,
                           only_rules=set(impact.touched), names=self.names)
        return self._write(ctx)

    def _write(self, ctx: EventBContext) -> Path:
        p = write_context(ctx, self.out_dir, self.buc)
        self.files.append(p)
        self.theorems_per_version.append(len(ctx.theorems))
        return p

    # ----- RQ3 accounting ----------------------------------------------------
    def po_stats(self) -> list[dict]:
        """Per version: new THM POs generated at that step vs inherited from ancestors."""
        out, cum = [], 0
        for v, n in enumerate(self.theorems_per_version):
            out.append({"version": v, "po_new": n, "po_inherited": cum,
                        "po_reuse_rate": (cum / (cum + n)) if (cum + n) else 1.0})
            cum += n
        return out


def refinement_context(policy_after: Policy, delta: Delta, impact: Impact,
                       pairs: list[tuple[Rule, Rule]], parent: str,
                       names: dict[str, str] | None = None) -> EventBContext:
    """Stateless variant (no versioning across calls); prefer EventBLedger."""
    name = f"PolicyCtx_v{policy_after.version}"
    if delta.kind is DeltaKind.REMOVE:
        return EventBContext(name, extends=parent)
    return base_context(policy_after, pairs, name=name, extends=parent,
                        only_rules=set(impact.touched), names=names)


def po_stats_from_dir(out_dir: str | Path) -> list[dict]:
    """Recompute po_stats from the .ctx files on disk (e.g. after a run)."""
    files = sorted(Path(out_dir).glob("PolicyCtx_v*.ctx"), key=lambda p: int(p.stem.split("_v")[1]))
    out, cum = [], 0
    for p in files:
        n = count_theorems(p)
        v = int(p.stem.split("_v")[1])
        out.append({"version": v, "po_new": n, "po_inherited": cum,
                    "po_reuse_rate": (cum / (cum + n)) if (cum + n) else 1.0})
        cum += n
    return out

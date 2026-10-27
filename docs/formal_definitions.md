# Formal definitions (paper §4) ↔ code

| Paper | Code |
|---|---|
| attribute space 𝒜 | `core.AttributeSpace` |
| ontology O = (aliases, hierarchy) | `core.Ontology` (`AttributeSpace.ontology`) |
| rule rᵢ = (Tᵢ, Cᵢ, effᵢ) | `core.Rule(target, condition, effect)` |
| Tᵢ ∧ Cᵢ | `Rule.applicability` |
| canonical applicability ⟦rᵢ⟧ = canon_O(Tᵢ ∧ Cᵢ) | `AttributeSpace.applicability(r)` |
| conflict(rᵢ,rⱼ) ⇔ ∃q. ⟦rᵢ⟧(q) ∧ ⟦rⱼ⟧(q) ∧ effᵢ≠effⱼ | `SMTChecker.check_pair(...).conflict` |
| Δ ∈ {add, remove, modify} | `core.Delta`, `core.apply_delta` |
| Impact(Δ) | `incremental.compute_impact` |
| Theorem 1 (Tier A sound) | `filtering/sound_pruning.py` docstring; `tests/test_sound_pruning.py`, `tests/test_ontology.py` |
| Theorem 2 (PO preservation) | `incremental/impact.py` docstring; `verification/eventb/refinement.py`; `tests/test_eventb_gen.py` |
| Event-B theorem per pair: app_a ∩ app_b = ∅ | `verification.eventb.pair_theorem` |

## Canonicalisation (ontology)

For alias attribute a with canonical attribute c and value map μ: an atom `a op v` becomes
`c op μ(v)` (`in`/`notin` map every value). For a value hierarchy ⊑ on c: `c = v` becomes
`c ∈ desc(v)` where desc(v) = {v} ∪ {w | w ⊑⁺ v}, and `c ≠ v` becomes `c ∉ desc(v)`.
Numeric atoms are unchanged. Canonicalisation is applied *before* Tier A, before the Z3
encoding and before Event-B generation, so all three agree on the semantics (and the
literal exact-index baseline, which does not use it, is what misses cross-vocabulary conflicts).

## Theorem 1 — proof (expanded)

Let α_k(x) be the abstraction of rule r_k on attribute x computed by `core.intervals.abstract_attr`
from the atoms of ⟦r_k⟧ that mention x. Each abstraction step (interval intersection,
allow/deny set intersection) is an over-approximation of the concrete satisfying set
S_k(x) = { q(x) | q ⊨ ⟦r_k⟧ }, because:
1. the concrete set is the intersection over *all* conjuncts, including raw_expr and atoms
   on other attributes, hence S_k(x) ⊆ ⋂_{atoms on x} sat(atom) = α_k(x);
2. `Interval.intersect` / `ValueSet.intersect` compute exactly the intersection of the
   atom sets they are given.
If α_i(x) ∩ α_j(x) = ∅ then S_i(x) ∩ S_j(x) = ∅, so no q satisfies both rules. ∎

## Theorem 2 — proof (expanded)

PO(r_i, r_j) is the theorem app_i ∩ app_j = ∅ in a context whose only hypotheses about
r_i, r_j are their definitional axioms def_app_i, def_app_j. Context extension in Event-B
is monotonic: PolicyCtx_v{N+1} extends PolicyCtx_v{N} keeps every axiom and theorem.
Δ changes only the definitions of rules in touched(Δ); for r_i, r_j ∉ touched(Δ) the
hypotheses of PO(r_i, r_j) are unchanged, hence the proof remains valid. ∎
A modified rule r → r' is introduced as a *new* constant app_r__v{N+1} (the ledger in
`eventb/refinement.py`), so def_app_r is never redefined and the old POs stay true facts
about the old rule; the live pair set is tracked by `incremental.cache.POCache`.

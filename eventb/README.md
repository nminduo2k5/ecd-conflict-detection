# Event-B artefacts

`ecd` emits every policy version as a context in two formats:

* `PolicyCtx_v<N>.ctx` — textual Event-B (Camille syntax, shown in the paper)
* `PolicyCtx_v<N>.buc` — Rodin's native XML, importable as-is

```
context PolicyCtx_v1
extends PolicyCtx_v0
constants app_r1000 app_r5__v1
axioms
  @def_app_r1000  app_r1000 = {q · q ∈ REQ ∧ role(q) = role_doctor ∧ hour(q) ∈ 8 ‥ 18 | q}
  @def_app_r5__v1 app_r5__v1 = {q · q ∈ REQ ∧ role(q) ∈ {role_doctor, role_senior_doctor} | q}
  theorem @thm_r1000_r12 app_r1000 ∩ app_r12 = ∅
  theorem @thm_r5__v1_r7 app_r5__v1 ∩ app_r7 = ∅
end
```

* Rules are written over the **canonical** vocabulary (aliases renamed, hierarchies expanded),
  exactly what Tier A and Z3 reason about.
* Each Δ is a context extension; only pairs in Impact(Δ) get theorems (Theorem 2). A modified
  rule gets a **versioned constant** (`app_r5__v1`) because contexts are monotonic — the old
  constant and its theorems stay valid facts about the old rule.
* Raw SMT-LIB conditions are translated to Event-B arithmetic (`(> (+ hour age) 120)` →
  `(hour(q) + age(q)) > 120`); untranslatable ones are over-approximated by `TRUE = TRUE`
  and flagged in a comment (conservative: can only make a disjointness theorem harder).

## Discharging with Rodin (headless)

```bash
ecd rodin results/eventb/rq3 --project ECD            # lay out <workspace>/ECD with .buc, .project, build.xml
ecd rodin results/eventb/rq3 --project ECD --build --rodin-path /opt/rodin/rodin
```

The build runs `rodin -application org.eclipse.ant.core.antRunner -data <ws> -buildfile build.xml`
(refresh + full build = static checker, PO generator, auto-prover). Install the *SMT Solvers*
plugin and add it to the auto-tactic profile so every `thm_*` PO goes to Z3 / CVC4 / veriT.
`RodinRunner.status()` parses `*.bps` (confidence 1000 = discharged) and `*.bpo` (PO counts).
An undischarged theorem = conflict; the SMT model = counterexample.

Eclipse does not auto-import a project folder dropped into the workspace: import it once in the
Rodin UI (File ▸ Import ▸ Existing Projects), after that every new `PolicyCtx_v<N>.buc` is picked up
by the headless build. Rodin was not available on the machine the paper runs were made on; the
PO counts in RQ3 are taken from the generated contexts (one THM PO per pair theorem) and the
verdicts from Z3 on the same theorems (`app_a ∩ app_b = ∅` ⇔ UNSAT(⟦a⟧ ∧ ⟦b⟧)).

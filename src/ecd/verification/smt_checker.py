"""Exact conflict check for a pair of rules using Z3.

conflict(r_i, r_j) ⇔ SAT( T_i ∧ C_i ∧ T_j ∧ C_j ) ∧ eff_i ≠ eff_j

The satisfying model is returned as a concrete request q (the counterexample
shown to the user). This module is also used to generate *training labels*
for the overlap-aware embedding (RQ1) — there the effect is ignored and we
only ask whether the two applicability regions intersect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import z3

from ecd.core import Atom, AttributeSpace, AttrType, Predicate, Rule


@dataclass
class CheckResult:
    overlap: bool                       # do applicability regions intersect?
    conflict: bool                      # overlap ∧ effects differ
    counterexample: dict[str, Any] = field(default_factory=dict)
    status: str = "sat"                 # sat | unsat | unknown
    solver_ms: float = 0.0


class SMTChecker:
    """Stateless (per call) encoder of rules into Z3 constraints."""

    def __init__(self, space: AttributeSpace | None = None, timeout_ms: int = 5000):
        self.space = space or AttributeSpace()
        self.timeout_ms = timeout_ms
        self._cat_codes: dict[str, dict[Any, int]] = {}
        # One shared variable environment + one solver reused with push/pop:
        # variables are global per attribute name, so encodings can be cached per rule.
        self._env: dict[str, z3.ExprRef] = {}
        self._rule_enc: dict[str, z3.ExprRef] = {}      # content_hash -> encoded T∧C
        self._domain_added: set[str] = set()
        self._solver = z3.Solver()
        self._solver.set("timeout", self.timeout_ms)

    # ----- public API ------------------------------------------------------
    def encode_rule(self, r: Rule) -> z3.ExprRef:
        """Encode the *canonical* applicability ⟦r⟧ (aliases renamed, hierarchies
        expanded — see core.ontology), so the oracle agrees with Tier A and Event-B."""
        h = r.content_hash()
        if h not in self._rule_enc:
            self._rule_enc[h] = self._encode(self.space.applicability(r), self._env)
            self._add_domain_constraints(self._solver, self._env)
        return self._rule_enc[h]

    def check_pair(self, a: Rule, b: Rule) -> CheckResult:
        import time
        t0 = time.perf_counter()
        ea, eb = self.encode_rule(a), self.encode_rule(b)
        s = self._solver
        s.push()
        s.add(ea, eb)
        res = s.check()
        ms = (time.perf_counter() - t0) * 1000
        if res == z3.sat:
            ont = self.space.ontology
            used = {ont.canonical_attr(n) for n in (a.attrs() | b.attrs())}
            env = {n: v for n, v in self._env.items() if n in used} or self._env
            cex = self._model_to_request(s.model(), env)
            s.pop()
            return CheckResult(True, a.effect != b.effect, cex, "sat", ms)
        s.pop()
        if res == z3.unsat:
            return CheckResult(False, False, {}, "unsat", ms)
        return CheckResult(True, a.effect != b.effect, {}, "unknown", ms)  # conservative

    def overlaps(self, a: Rule, b: Rule) -> bool:
        return self.check_pair(a, b).overlap

    # ----- encoding ----------------------------------------------------
    def _var(self, name: str, env: dict, hint: Atom | None = None) -> z3.ExprRef:
        if name in env:
            return env[name]
        attr = self.space.get(name)
        t = attr.type if attr else self._infer_type(hint)
        if t is AttrType.NUMERIC:
            v = z3.Real(name)
        elif t is AttrType.BOOLEAN:
            v = z3.Bool(name)
        else:
            v = z3.Int(name)       # categorical values are encoded as ints
        env[name] = v
        return v

    @staticmethod
    def _infer_type(hint: Atom | None) -> AttrType:
        if hint is None:
            return AttrType.CATEGORICAL
        val = hint.value
        if hint.op in {"lt", "le", "gt", "ge", "range"}:
            return AttrType.NUMERIC
        if isinstance(val, bool):
            return AttrType.BOOLEAN
        if isinstance(val, (int, float)):
            return AttrType.NUMERIC
        return AttrType.CATEGORICAL

    def _code(self, attr: str, value: Any) -> int:
        table = self._cat_codes.setdefault(attr, {})
        if value not in table:
            table[value] = len(table)
        return table[value]

    def _lit(self, var: z3.ExprRef, attr: str, value: Any):
        if z3.is_int(var) and not isinstance(value, (int, float)):
            return z3.IntVal(self._code(attr, value))
        if z3.is_int(var) and isinstance(value, (int, float)) and self.space.get(attr) is None:
            # untyped numeric literal on a categorical-inferred var
            return z3.IntVal(self._code(attr, value))
        if z3.is_bool(var):
            return z3.BoolVal(bool(value))
        return z3.RealVal(value) if z3.is_real(var) else z3.IntVal(int(value))

    def _encode_atom(self, a: Atom, env: dict):
        v = self._var(a.attr, env, a)
        if a.op == "any":
            return z3.BoolVal(True)
        if a.op == "eq":
            return v == self._lit(v, a.attr, a.value)
        if a.op == "neq":
            return v != self._lit(v, a.attr, a.value)
        if a.op == "in":
            return z3.Or([v == self._lit(v, a.attr, x) for x in a.value]) if a.value else z3.BoolVal(False)
        if a.op == "notin":
            return z3.And([v != self._lit(v, a.attr, x) for x in a.value])
        if a.op == "lt":
            return v < self._lit(v, a.attr, a.value)
        if a.op == "le":
            return v <= self._lit(v, a.attr, a.value)
        if a.op == "gt":
            return v > self._lit(v, a.attr, a.value)
        if a.op == "ge":
            return v >= self._lit(v, a.attr, a.value)
        if a.op == "range":
            lo, hi = a.value
            return z3.And(v >= self._lit(v, a.attr, lo), v <= self._lit(v, a.attr, hi))
        raise ValueError(a.op)

    def _encode(self, p: Predicate, env: dict):
        parts = [self._encode_atom(a, env) for a in p.atoms]
        if p.raw_expr:
            parts.append(self._encode_raw(p.raw_expr, env))
        return z3.And(parts) if parts else z3.BoolVal(True)

    def _encode_raw(self, raw: str, env: dict):
        """Parse an SMT-LIB2 expression. Variables must be declared in `self.space`
        (so their sorts are known) — undeclared names default to Int."""
        import re
        names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.:-]*", raw)) - {
            "and", "or", "not", "ite", "true", "false", "div", "mod", "abs", "distinct", "to_real", "to_int"}
        ont = self.space.ontology
        decls = []
        canon: dict[str, z3.ExprRef] = {}
        for n in sorted(names):
            v = self._var(ont.canonical_attr(n), env)          # alias names share the canonical variable
            canon[n] = v
            sort = "Real" if z3.is_real(v) else "Bool" if z3.is_bool(v) else "Int"
            decls.append(f"(declare-const {n} {sort})")
        script = "\n".join(decls) + f"\n(assert {raw})"
        asts = z3.parse_smt2_string(script)
        # rebind parsed consts to our env vars by name
        subst = []
        for n in names:
            for sort_ctor in (z3.Int, z3.Real, z3.Bool):
                try:
                    cand = sort_ctor(n)
                    if cand.sort() == canon[n].sort():
                        subst.append((cand, canon[n]))
                        break
                except Exception:
                    pass
        return z3.substitute(z3.And(list(asts)), *subst)

    def _add_domain_constraints(self, s: z3.Solver, env: dict) -> None:
        """Add domain axioms (once per variable) at the solver's base level."""
        for name, v in env.items():
            if name in self._domain_added:
                continue
            self._domain_added.add(name)
            attr = self.space.get(name)
            if attr is None:
                continue
            if attr.type is AttrType.NUMERIC:
                if attr.lo is not None:
                    s.add(v >= attr.lo)
                if attr.hi is not None:
                    s.add(v <= attr.hi)
            elif attr.type is AttrType.CATEGORICAL and attr.domain:
                codes = [self._code(name, x) for x in attr.domain]
                s.add(z3.Or([v == c for c in codes]))

    def _model_to_request(self, m: z3.ModelRef, env: dict) -> dict[str, Any]:
        req: dict[str, Any] = {}
        for name, v in env.items():
            val = m.eval(v, model_completion=True)
            if z3.is_bool(val):
                req[name] = z3.is_true(val)
            elif z3.is_int_value(val):
                i = val.as_long()
                inv = {c: x for x, c in self._cat_codes.get(name, {}).items()}
                req[name] = inv.get(i, i)
            else:
                try:
                    req[name] = float(val.as_decimal(6).rstrip("?"))
                except Exception:
                    req[name] = str(val)
        return req

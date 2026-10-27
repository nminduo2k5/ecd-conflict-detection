"""Step [4]: generate Event-B artefacts from a policy.

Two renderings of the same context:
  * `.ctx`  textual Event-B (Camille syntax, human-readable, what the paper shows)
  * `.buc`  Rodin's native XML, importable into a Rodin project as-is (see
            rodin_runner.py for the headless build).

Layout:

    PolicyCtx_v<N>       context   — attribute carrier sets/constants, one
                                     predicate constant `app_<rid>` per rule
                                     (its *canonical* applicability ⟦r⟧ as a set of REQ),
                                     and one THEOREM per candidate pair:

        thm_<a>_<b>:  app_<a> ∩ app_<b> = ∅        (no conflict)

    A *failed* theorem (Rodin cannot discharge; SMT returns a model) is a
    conflict, and the model is the counterexample.

Incremental scheme (C3): each Δ becomes `PolicyCtx_v<N+1> EXTENDS PolicyCtx_v<N>`.
Only theorems for pairs in Impact(Δ) are added; theorems in the parent
context are inherited and their POs are not regenerated (Theorem 2).
Contexts are monotonic, so a MODIFIED rule gets a *versioned* constant
(`app_r5__v3`, see refinement.EventBLedger) instead of redefining `app_r5`.

Complex conditions (`raw_expr`, SMT-LIB arithmetic) are translated to Event-B
arithmetic when they use + - * div mod and comparisons; anything else falls
back to `TRUE = TRUE` (an over-approximation of app_r, hence conservative:
it can only make a disjointness theorem harder, never hide a conflict) and
is flagged in a comment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import quoteattr

from ecd.core import Atom, AttrType, Policy, Predicate, Rule


def _ident(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in s)


def _val(attr: str, v) -> str:
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return str(int(v)) if float(v).is_integer() else str(v)
    sv = _ident(str(v))
    return sv if sv.startswith(_ident(attr) + "_") else f"{_ident(attr)}_{sv}"


def atom_to_eventb(a: Atom) -> str:
    x = _ident(a.attr)
    if a.op == "any":
        return "TRUE = TRUE"
    if a.op == "eq":
        return f"{x}(q) = {_val(a.attr, a.value)}"
    if a.op == "neq":
        return f"{x}(q) ≠ {_val(a.attr, a.value)}"
    if a.op == "in":
        return f"{x}(q) ∈ {{{', '.join(_val(a.attr, v) for v in a.value)}}}"
    if a.op == "notin":
        return f"{x}(q) ∉ {{{', '.join(_val(a.attr, v) for v in a.value)}}}"
    if a.op == "lt":
        return f"{x}(q) < {_val(a.attr, a.value)}"
    if a.op == "le":
        return f"{x}(q) ≤ {_val(a.attr, a.value)}"
    if a.op == "gt":
        return f"{x}(q) > {_val(a.attr, a.value)}"
    if a.op == "ge":
        return f"{x}(q) ≥ {_val(a.attr, a.value)}"
    if a.op == "range":
        lo, hi = a.value
        return f"{x}(q) ∈ {_val(a.attr, lo)} ‥ {_val(a.attr, hi)}"
    raise ValueError(a.op)


# ----- SMT-LIB arithmetic → Event-B ------------------------------------------
def _parse_sexpr(s: str):
    toks = s.replace("(", " ( ").replace(")", " ) ").split()
    pos = 0

    def read():
        nonlocal pos
        t = toks[pos]
        pos += 1
        if t == "(":
            lst = []
            while toks[pos] != ")":
                lst.append(read())
            pos += 1
            return lst
        return t

    node = read()
    if pos != len(toks):
        raise ValueError("trailing tokens")
    return node


_ARITH = {"+": "+", "-": "−", "*": "∗", "div": "÷", "mod": "mod"}
_CMP = {">": ">", "<": "<", ">=": "≥", "<=": "≤", "=": "="}


def _sexpr_to_eventb(n) -> str:
    if isinstance(n, str):
        if n in ("true", "false"):
            return "TRUE = TRUE" if n == "true" else "TRUE = FALSE"
        try:
            v = float(n)
            if not v.is_integer():
                raise ValueError("Event-B integers only")
            return str(int(v)) if v >= 0 else f"(−{abs(int(v))})"
        except ValueError:
            if n.replace(".", "").replace("-", "").replace(":", "").replace("_", "").isalnum():
                return f"{_ident(n)}(q)"
            raise
    op, args = n[0], n[1:]
    if op in _ARITH:
        if op == "-" and len(args) == 1:
            return f"(−{_sexpr_to_eventb(args[0])})"
        return "(" + f" {_ARITH[op]} ".join(_sexpr_to_eventb(a) for a in args) + ")"
    if op in _CMP:
        parts = [_sexpr_to_eventb(a) for a in args]
        chain = [f"{parts[i]} {_CMP[op]} {parts[i + 1]}" for i in range(len(parts) - 1)]
        return " ∧ ".join(chain) if len(chain) == 1 else "(" + " ∧ ".join(chain) + ")"
    if op == "distinct" and len(args) == 2:
        return f"{_sexpr_to_eventb(args[0])} ≠ {_sexpr_to_eventb(args[1])}"
    if op == "and":
        return "(" + " ∧ ".join(_sexpr_to_eventb(a) for a in args) + ")"
    if op == "or":
        return "(" + " ∨ ".join(_sexpr_to_eventb(a) for a in args) + ")"
    if op == "not" and len(args) == 1:
        return f"¬({_sexpr_to_eventb(args[0])})"
    raise ValueError(f"unsupported SMT-LIB operator {op}")


def raw_to_eventb(raw: str) -> tuple[str, bool]:
    """Translate an SMT-LIB2 conjunct; (text, ok). Not ok ⇒ over-approximated by TRUE."""
    try:
        return _sexpr_to_eventb(_parse_sexpr(raw)), True
    except Exception:
        return f"/* RAW (untranslated, over-approximated): {raw} */ TRUE = TRUE", False


def predicate_to_eventb(p: Predicate) -> str:
    parts = [atom_to_eventb(a) for a in p.atoms]
    if p.raw_expr:
        parts.append(raw_to_eventb(p.raw_expr)[0])
    return " ∧ ".join(parts) if parts else "TRUE = TRUE"


# ----- context model ---------------------------------------------------------
@dataclass
class EventBContext:
    name: str
    extends: str | None = None
    sets: list[str] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)
    axioms: list[tuple[str, str]] = field(default_factory=list)
    theorems: list[tuple[str, str]] = field(default_factory=list)

    def render(self) -> str:
        """Camille text."""
        out = [f"context {self.name}"]
        if self.extends:
            out.append(f"extends {self.extends}")
        if self.sets:
            out.append("sets " + " ".join(self.sets))
        if self.constants:
            out.append("constants " + " ".join(self.constants))
        if self.axioms or self.theorems:
            out.append("axioms")
            for lbl, body in self.axioms:
                out.append(f"  @{lbl} {body}")
            for lbl, body in self.theorems:
                out.append(f"  theorem @{lbl} {body}")
        out.append("end")
        return "\n".join(out) + "\n"

    def render_buc(self) -> str:
        """Rodin XML (org.eventb.core.contextFile, version 3)."""
        L = ['<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
             '<org.eventb.core.contextFile org.eventb.core.configuration="org.eventb.core.fwd" version="3">']
        n = 0
        if self.extends:
            L.append(f'<org.eventb.core.extendsContext name="_e{n}" org.eventb.core.target={quoteattr(self.extends)}/>')
            n += 1
        for s in self.sets:
            L.append(f'<org.eventb.core.carrierSet name="_s{n}" org.eventb.core.identifier={quoteattr(s)}/>')
            n += 1
        for c in self.constants:
            L.append(f'<org.eventb.core.constant name="_c{n}" org.eventb.core.identifier={quoteattr(c)}/>')
            n += 1
        for lbl, body in self.axioms:
            L.append(f'<org.eventb.core.axiom name="_a{n}" org.eventb.core.label={quoteattr(lbl)} '
                     f'org.eventb.core.predicate={quoteattr(_strip_comments(body))} org.eventb.core.theorem="false"/>')
            n += 1
        for lbl, body in self.theorems:
            L.append(f'<org.eventb.core.axiom name="_t{n}" org.eventb.core.label={quoteattr(lbl)} '
                     f'org.eventb.core.predicate={quoteattr(_strip_comments(body))} org.eventb.core.theorem="true"/>')
            n += 1
        L.append("</org.eventb.core.contextFile>")
        return "\n".join(L) + "\n"


def _strip_comments(body: str) -> str:
    import re
    return re.sub(r"/\*.*?\*/\s*", "", body)


def const_name(rule_id: str, version: int | None = None) -> str:
    base = f"app_{_ident(rule_id)}"
    return base if version is None else f"{base}__v{version}"


def pair_theorem(a: Rule, b: Rule, names: dict[str, str] | None = None) -> tuple[str, str]:
    """THEOREM: app_a ∩ app_b = ∅  (negation of the conflict formula)."""
    ca = names.get(a.id, const_name(a.id)) if names else const_name(a.id)
    cb = names.get(b.id, const_name(b.id)) if names else const_name(b.id)
    return (f"thm_{ca[4:]}_{cb[4:]}", f"{ca} ∩ {cb} = ∅")


def base_context(policy: Policy, pairs: list[tuple[Rule, Rule]], name: str | None = None,
                 extends: str | None = None, only_rules: set[str] | None = None,
                 names: dict[str, str] | None = None) -> EventBContext:
    """Build (or extend) a context.

    only_rules: when extending, declare only these rules' `app_` constants
                (the rest are inherited from the parent).
    names:      rule id → constant name (versioned for modified rules).
    """
    ctx = EventBContext(name or f"PolicyCtx_v{policy.version}", extends)
    ont = policy.space.ontology
    if extends is None:
        ctx.sets = ["REQ"]
        for attr in policy.space.attributes.values():
            if ont.canonical_attr(attr.name) != attr.name:
                continue                                   # alias attrs never appear in ⟦r⟧
            x = _ident(attr.name)
            if attr.type is AttrType.CATEGORICAL:
                ctx.sets.append(f"{x}_T")
                ctx.constants.append(x)
                ctx.axioms.append((f"ax_{x}", f"{x} ∈ REQ → {x}_T"))
                if attr.domain:
                    vals = [_val(attr.name, v) for v in attr.domain]
                    ctx.constants += vals
                    ctx.axioms.append((f"ax_{x}_part", f"partition({x}_T, " + ", ".join(f"{{{v}}}" for v in vals) + ")"))
            elif attr.type is AttrType.BOOLEAN:
                ctx.constants.append(x)
                ctx.axioms.append((f"ax_{x}", f"{x} ∈ REQ → BOOL"))
            else:
                ctx.constants.append(x)
                ctx.axioms.append((f"ax_{x}", f"{x} ∈ REQ → ℤ"))
    for r in policy:
        if only_rules is not None and r.id not in only_rules:
            continue
        cn = names.get(r.id, const_name(r.id)) if names else const_name(r.id)
        ctx.constants.append(cn)
        ctx.axioms.append((f"def_{cn}",
                           f"{cn} = {{q · q ∈ REQ ∧ {predicate_to_eventb(policy.space.applicability(r))} | q}}"))
    for a, b in pairs:
        ctx.theorems.append(pair_theorem(a, b, names))
    return ctx


def write_context(ctx: EventBContext, out_dir: str | Path, buc: bool = True) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{ctx.name}.ctx"
    p.write_text(ctx.render(), encoding="utf-8")
    if buc:
        (out_dir / f"{ctx.name}.buc").write_text(ctx.render_buc(), encoding="utf-8")
    return p


def count_theorems(path: str | Path) -> int:
    """Number of pair theorems (= THM proof obligations) declared in a .ctx file."""
    return sum(1 for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip().startswith("theorem @"))

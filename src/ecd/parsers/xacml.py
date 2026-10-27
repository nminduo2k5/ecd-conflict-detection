"""XACML 3.0 (and most 2.0) → Policy.

Normalisation (step [1]) keeps every rule a *conjunction* so Tier A stays sound:

* PolicySet / Policy nesting: a rule's effective target is the AND of its own
  Target and every enclosing Policy/PolicySet Target.
* Target = AND over AnyOf, AnyOf = OR over AllOf, AllOf = AND over Match
  ⇒ the target is a DNF; each disjunct becomes its own sub-rule `rid#k`
  (same effect, same original rule — sub-rules of one rule never conflict
  with each other because they share the effect).
* Condition: the <Apply> tree is translated into an expression AST over
  atomic predicates and put into DNF too (multiplied with the target DNF).
  Supported: and/or/not, <type>-equal / less-than / greater-than(-or-equal)
  on `one-and-only(attr)` or a bare designator, string-is-in, any-of,
  any-of-any / at-least-one-member-of with a literal bag, boolean-one-and-only,
  integer/double arithmetic (add, subtract, multiply, divide) → `raw_expr`
  in SMT-LIB over the attribute variables.
* Anything that cannot be translated (regexp match, dates, cross-attribute
  comparisons, XPath functions, variable references) is treated as TRUE —
  an *over-approximation* of the rule's applicability.  This never hides a
  conflict (no false negatives); the pair is flagged `approx` in the rule's
  meta so the report can say "potential conflict, condition not fully
  translated".
"""
from __future__ import annotations

import itertools
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ecd.core import Atom, Attribute, AttrType, CombiningAlgorithm, Effect, Policy, Predicate, Rule
from .base import PolicyParser

ALG_MAP = {
    "ordered-deny-overrides": CombiningAlgorithm.DENY_OVERRIDES,
    "ordered-permit-overrides": CombiningAlgorithm.PERMIT_OVERRIDES,
    "deny-overrides": CombiningAlgorithm.DENY_OVERRIDES,
    "permit-overrides": CombiningAlgorithm.PERMIT_OVERRIDES,
    "first-applicable": CombiningAlgorithm.FIRST_APPLICABLE,
    "only-one-applicable": CombiningAlgorithm.ONLY_ONE_APPLICABLE,
    "deny-unless-permit": CombiningAlgorithm.PERMIT_OVERRIDES,   # a Permit always wins
    "permit-unless-deny": CombiningAlgorithm.DENY_OVERRIDES,     # a Deny always wins
}
CMP_OPS = {"equal": "eq", "less-than": "lt", "less-than-or-equal": "le",
           "greater-than": "gt", "greater-than-or-equal": "ge"}
FLIP = {"lt": "gt", "gt": "lt", "le": "ge", "ge": "le", "eq": "eq", "neq": "neq"}
NEG = {"eq": "neq", "neq": "eq", "in": "notin", "notin": "in", "lt": "ge", "ge": "lt", "le": "gt", "gt": "le"}
ARITH = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "mod": "mod", "abs": "abs"}
MAX_ALTERNATIVES = 128


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _cmp_op(func: str) -> str | None:
    """`integer-greater-than-or-equal` → ge (longest suffix first, so `-equal` does not win)."""
    for k in sorted(CMP_OPS, key=len, reverse=True):
        if func.endswith("-" + k):
            return CMP_OPS[k]
    return None


def _fn(uri: str) -> str:
    return uri.rsplit(":", 1)[-1]


def _attr_name(el: ET.Element) -> str:
    if _local(el.tag) == "AttributeSelector":
        path = el.get("Path", "selector")
        name = re.sub(r"/text\(\)$", "", path).rstrip("/").rsplit("/", 1)[-1]
        name = name.split(":")[-1] or "selector"
        cat = el.get("Category", "").rsplit(":", 1)[-1]
        return f"{cat}.{name}" if cat else name
    cat = el.get("Category", "")
    cat_short = cat.rsplit(":", 1)[-1] if cat else ""
    aid = el.get("AttributeId", "").rsplit(":", 1)[-1]
    return f"{cat_short}.{aid}" if cat_short else aid


def _cast(v: str, dtype: str = ""):
    dt = dtype.rsplit("#", 1)[-1]
    if dt == "integer":
        try:
            return int(v)
        except ValueError:
            return v
    if dt == "double":
        try:
            return float(v)
        except ValueError:
            return v
    if dt == "boolean":
        return v.strip().lower() == "true"
    try:
        return int(v)
    except ValueError:
        if v.strip().lower() in ("true", "false"):
            return v.strip().lower() == "true"
        return v


# ----- expression AST ---------------------------------------------------------
@dataclass(frozen=True)
class Leaf:          # atomic predicate
    atom: Atom | None = None      # Atom(attr, op, value)
    raw: str | None = None        # SMT-LIB conjunct
    unknown: bool = False         # TRUE (over-approximation)


@dataclass(frozen=True)
class Node:
    op: str                      # and | or | not
    kids: tuple


class _Ref:          # attribute reference (bag of one)
    def __init__(self, name: str):
        self.name = name


class _Lit:          # literal (possibly a bag of literals)
    def __init__(self, values: tuple):
        self.values = values


class _Arith:        # arithmetic expression over refs/literals, as SMT-LIB text
    def __init__(self, smt: str, names: set[str]):
        self.smt, self.names = smt, names


TRUE = Leaf(unknown=True)    # untranslatable → over-approximated by TRUE (flagged `approx`)
TAUT = Leaf()                # literally true (no condition / literal true): exact


def _negate(e):
    if isinstance(e, Leaf):
        if e.unknown:
            return TRUE
        if e.raw is not None:
            return Leaf(raw=f"(not {e.raw})")
        a = e.atom
        if a.op == "range":
            lo, hi = a.value
            return Node("or", (Leaf(Atom(a.attr, "lt", lo)), Leaf(Atom(a.attr, "gt", hi))))
        if a.op in NEG:
            return Leaf(Atom(a.attr, NEG[a.op], a.value))
        return TRUE
    if e.op == "not":
        return e.kids[0]
    return Node("or" if e.op == "and" else "and", tuple(_negate(k) for k in e.kids))


def _dnf(e) -> list[list[Leaf]]:
    if isinstance(e, Leaf):
        return [[e]]
    if e.op == "not":
        return _dnf(_negate(e.kids[0]))
    if e.op == "or":
        out = []
        for k in e.kids:
            out += _dnf(k)
            if len(out) > MAX_ALTERNATIVES:
                return [[TRUE]]
        return out
    # and
    alts = [[]]
    for k in e.kids:
        kd = _dnf(k)
        alts = [a + b for a in alts for b in kd]
        if len(alts) > MAX_ALTERNATIVES:
            return [[TRUE]]
    return alts


# ----- parser -----------------------------------------------------------------
class XACMLParser(PolicyParser):
    def __init__(self):
        self.stats = {"rules": 0, "sub_rules": 0, "approx_rules": 0, "untranslated": []}

    def parse(self, path: str | Path) -> Policy:
        root = ET.parse(path).getroot()
        self.stats = {"rules": 0, "sub_rules": 0, "approx_rules": 0, "untranslated": []}
        self._attr_types: dict[str, set] = {}
        alg = self._alg(root)
        pol = Policy(name=root.get("PolicyId") or root.get("PolicySetId") or Path(path).stem, combining=alg)
        rules: list[Rule] = []
        self._walk(root, [], alg, rules)
        # attribute typing from observed literals / operators
        for name, kinds in self._attr_types.items():
            if kinds == {"bool"}:
                pol.space.add(Attribute(name, AttrType.BOOLEAN))
            elif "num" in kinds and "str" not in kinds:
                pol.space.add(Attribute(name, AttrType.NUMERIC))
        seen: set[str] = set()
        for r in rules:
            rid = r.id
            if rid in seen:                          # same RuleId in two policies of a PolicySet
                k = 2
                while f"{rid}~{k}" in seen:
                    k += 1
                rid = f"{rid}~{k}"
                r = Rule(rid, r.target, r.condition, r.effect, r.meta)
            seen.add(rid)
            pol.add(r)
        return pol

    def _alg(self, el: ET.Element) -> CombiningAlgorithm:
        uri = el.get("RuleCombiningAlgId") or el.get("PolicyCombiningAlgId") or ""
        for k, v in ALG_MAP.items():
            if uri.endswith(k):
                return v
        return CombiningAlgorithm.DENY_OVERRIDES

    def _walk(self, el: ET.Element, targets: list[list[list[Leaf]]], alg: CombiningAlgorithm,
              out: list[Rule], policy_id: str = "") -> None:
        tag = _local(el.tag)
        if tag in ("PolicySet", "Policy"):
            alg = self._alg(el)
            policy_id = el.get("PolicyId") or el.get("PolicySetId") or policy_id
            own = [self._target(c) for c in el if _local(c.tag) == "Target"]
            targets = targets + [t for t in own if t != [[]]]
            for c in el:
                self._walk(c, targets, alg, out, policy_id)
        elif tag == "Rule":
            out += self._rule(el, targets, alg, policy_id)

    # --- target ----------------------------------------------------------------
    def _target(self, t: ET.Element) -> list[list[Leaf]]:
        """DNF of a Target: AND over AnyOf of (OR over AllOf of (AND over Match))."""
        alts: list[list[Leaf]] = [[]]
        for anyof in (c for c in t if _local(c.tag) == "AnyOf"):
            allofs = [[Leaf(a) for a in self._matches(x)] for x in anyof if _local(x.tag) == "AllOf"]
            if not allofs:
                continue
            alts = [a + b for a in alts for b in allofs]
            if len(alts) > MAX_ALTERNATIVES:
                self.stats["untranslated"].append("target too large → TRUE")
                return [[TRUE]]
        # XACML 2.0 style <Subjects>/<Resources>/<Actions>
        for grp in (c for c in t if _local(c.tag) in ("Subjects", "Resources", "Actions", "Environments")):
            allofs = [[Leaf(a) for a in self._matches(x)] for x in grp]
            if allofs:
                alts = [a + b for a in alts for b in allofs]
        return alts

    def _matches(self, allof: ET.Element) -> list[Atom]:
        out = []
        for m in (e for e in allof if _local(e.tag).endswith("Match")):
            func = _fn(m.get("MatchId", ""))
            val_el = next((e for e in m if _local(e.tag) == "AttributeValue"), None)
            des = next((e for e in m if _local(e.tag) in ("AttributeDesignator", "AttributeSelector")), None)
            if val_el is None or des is None:
                continue
            name = _attr_name(des)
            value = _cast(val_el.text or "", val_el.get("DataType", ""))
            op = _cmp_op(func)
            if op is None:
                self.stats["untranslated"].append(func)
                self._note_type(name, value)
                out.append(Atom(name, "any"))
                continue
            self._note_type(name, value)
            out.append(Atom(name, op, value))
        return out

    def _note_type(self, name: str, value: Any) -> None:
        kind = "bool" if isinstance(value, bool) else "num" if isinstance(value, (int, float)) else "str"
        self._attr_types.setdefault(name, set()).add(kind)

    # --- condition -------------------------------------------------------------
    def _condition(self, c: ET.Element):
        applies = [e for e in c if _local(e.tag) in ("Apply", "AttributeDesignator", "AttributeSelector",
                                                      "AttributeValue", "VariableReference")]
        if not applies:
            return TAUT
        try:
            return self._as_bool(self._expr(applies[0]))
        except Exception as ex:  # pragma: no cover - defensive
            self.stats["untranslated"].append(f"condition error: {ex}")
            return TRUE

    def _expr(self, el: ET.Element):
        """Translate an element into: Leaf/Node (boolean) | _Ref | _Lit | _Arith."""
        tag = _local(el.tag)
        if tag in ("AttributeDesignator", "AttributeSelector"):
            return _Ref(_attr_name(el))
        if tag == "AttributeValue":
            return _Lit((_cast(el.text or "", el.get("DataType", "")),))
        if tag == "VariableReference":
            self.stats["untranslated"].append("VariableReference")
            return TRUE
        if tag != "Apply":
            return TRUE
        func = _fn(el.get("FunctionId", ""))
        kids = [c for c in el if _local(c.tag) != "Description"]
        if func == "and":
            return Node("and", tuple(self._as_bool(self._expr(k)) for k in kids))
        if func == "or":
            return Node("or", tuple(self._as_bool(self._expr(k)) for k in kids))
        if func == "not":
            return Node("not", (self._as_bool(self._expr(kids[0])),)) if kids else TRUE
        if func.endswith("-one-and-only"):
            inner = self._expr(kids[0]) if kids else TRUE
            if isinstance(inner, _Ref) and func.startswith("boolean"):
                return inner            # may be used directly as a predicate (handled by _as_bool)
            return inner
        if func.endswith("-bag"):
            vals = []
            for k in kids:
                v = self._expr(k)
                if isinstance(v, _Lit):
                    vals += list(v.values)
                else:
                    return self._unknown(func)
            return _Lit(tuple(vals))
        if func.endswith("-is-in"):                       # is-in(value, bag)
            a, b = (self._expr(k) for k in kids[:2])
            return self._membership(a, b, func)
        if func.endswith("-at-least-one-member-of") or func.endswith("-intersection"):
            a, b = (self._expr(k) for k in kids[:2])
            return self._membership(a, b, func)
        if func in ("any-of", "all-of", "any-of-any", "all-of-any", "any-of-all", "all-of-all"):
            fref = next((k for k in kids if _local(k.tag) == "Function"), None)
            inner = _fn(fref.get("FunctionId", "")) if fref is not None else "equal"
            args = [self._expr(k) for k in kids if _local(k.tag) != "Function"]
            if len(args) != 2:
                return self._unknown(func)
            op = _cmp_op(inner)
            if op is None:
                return self._unknown(f"{func}({inner})")
            return self._compare(args[0], args[1], op, func)
        arith = next((ARITH[k] for k in ARITH if func.endswith("-" + k)), None)
        if arith is not None:
            parts = [self._expr(k) for k in kids]
            smt, names = [], set()
            for p in parts:
                if isinstance(p, _Ref):
                    smt.append(_smt_sym(p.name)); names.add(p.name)
                    self._attr_types.setdefault(p.name, set()).add("num")
                elif isinstance(p, _Lit) and len(p.values) == 1 and isinstance(p.values[0], (int, float)):
                    smt.append(str(p.values[0]))
                elif isinstance(p, _Arith):
                    smt.append(p.smt); names |= p.names
                else:
                    return self._unknown(func)
            return _Arith(f"({arith} {' '.join(smt)})", names)
        op = _cmp_op(func)
        if op is not None and len(kids) >= 2:
            a, b = (self._expr(k) for k in kids[:2])
            return self._compare(a, b, op, func)
        return self._unknown(func)

    def _unknown(self, what: str):
        self.stats["untranslated"].append(what)
        return TRUE

    def _as_bool(self, e):
        if isinstance(e, _Ref):                 # boolean attribute used as a predicate
            self._attr_types.setdefault(e.name, set()).add("bool")
            return Leaf(Atom(e.name, "eq", True))
        if isinstance(e, (_Lit, _Arith)):
            if isinstance(e, _Lit) and len(e.values) == 1 and isinstance(e.values[0], bool):
                return TAUT if e.values[0] else Leaf(raw="false")
            return self._unknown("non-boolean expression as predicate")
        return e

    def _membership(self, a, b, func: str):
        """value ∈ bag / bags intersect: Atom `in` when one side is a literal bag and the other an attribute."""
        if isinstance(a, _Lit) and isinstance(b, _Ref):
            lit, ref = a, b
        elif isinstance(b, _Lit) and isinstance(a, _Ref):
            lit, ref = b, a
        else:
            return self._unknown(func)
        for v in lit.values:
            self._note_type(ref.name, v)
        if len(lit.values) == 1:
            return Leaf(Atom(ref.name, "eq", lit.values[0]))
        return Leaf(Atom(ref.name, "in", tuple(lit.values)))

    def _compare(self, a, b, op: str, func: str):
        if isinstance(a, _Lit) and isinstance(b, _Ref):
            a, b, op = b, a, FLIP[op]
        if isinstance(a, _Ref) and isinstance(b, _Lit):
            for v in b.values:
                self._note_type(a.name, v)
            if len(b.values) == 1:
                return Leaf(Atom(a.name, op, b.values[0]))
            if op == "eq":
                return Leaf(Atom(a.name, "in", tuple(b.values)))
            return self._unknown(func)
        # arithmetic comparison → SMT-LIB raw conjunct
        if isinstance(a, (_Arith, _Ref, _Lit)) and isinstance(b, (_Arith, _Ref, _Lit)):
            sa, sb = self._smt_operand(a), self._smt_operand(b)
            if sa is None or sb is None:
                return self._unknown(func)
            if isinstance(a, _Ref) and isinstance(b, _Ref):
                return self._unknown(f"{func} between two attributes")   # needs a shared value universe
            smt_op = {"eq": "=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}[op]
            return Leaf(raw=f"({smt_op} {sa} {sb})")
        return self._unknown(func)

    def _smt_operand(self, e) -> str | None:
        if isinstance(e, _Arith):
            return e.smt
        if isinstance(e, _Ref):
            self._attr_types.setdefault(e.name, set()).add("num")
            return _smt_sym(e.name)
        if isinstance(e, _Lit) and len(e.values) == 1 and isinstance(e.values[0], (int, float)) \
                and not isinstance(e.values[0], bool):
            return str(e.values[0])
        return None

    # --- rule assembly -----------------------------------------------------------
    def _rule(self, el: ET.Element, ancestors: list[list[list[Leaf]]], alg: CombiningAlgorithm,
              policy_id: str) -> list[Rule]:
        rid = el.get("RuleId", f"r{self.stats['rules']}")
        self.stats["rules"] += 1
        effect = Effect.PERMIT if el.get("Effect", "Permit").lower() == "permit" else Effect.DENY
        t_alts: list[list[Leaf]] = [[]]
        cond = TAUT
        for child in el:
            if _local(child.tag) == "Target":
                t_alts = self._target(child)
            elif _local(child.tag) == "Condition":
                cond = self._condition(child)
        target_alts = [[]]
        for anc in ancestors + [t_alts]:
            target_alts = [a + b for a in target_alts for b in anc]
            if len(target_alts) > MAX_ALTERNATIVES:
                self.stats["untranslated"].append(f"{rid}: target DNF too large → TRUE")
                target_alts = [[TRUE]]
                break
        cond_alts = _dnf(cond)
        rules: list[Rule] = []
        combos = list(itertools.product(target_alts, cond_alts))
        for k, (ta, ca) in enumerate(combos):
            approx = any(l.unknown for l in ta + ca)
            t_atoms = tuple(l.atom for l in ta if l.atom is not None)
            c_atoms = tuple(l.atom for l in ca if l.atom is not None)
            raws = [l.raw for l in ta + ca if l.raw is not None]
            raw = None if not raws else (raws[0] if len(raws) == 1 else "(and " + " ".join(raws) + ")")
            meta = {"source": "xacml", "xacml_rule": rid, "policy": policy_id, "combining": alg.value}
            if len(combos) > 1:
                meta["alt"] = k
            if approx:
                meta["approx"] = True
            sid = rid if len(combos) == 1 else f"{rid}#{k}"
            rules.append(Rule(sid, Predicate(t_atoms), Predicate(c_atoms, raw), effect, meta))
        self.stats["sub_rules"] += len(rules)
        self.stats["approx_rules"] += sum(1 for r in rules if r.meta.get("approx"))
        return rules


def _smt_sym(name: str) -> str:
    """Attribute name as an SMT-LIB simple symbol (letters, digits, . - _ allowed)."""
    return re.sub(r"[^A-Za-z0-9_.:-]", "_", name)

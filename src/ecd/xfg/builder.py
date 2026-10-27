"""Step [1b]: build the XACML-Feature Graph (XFG) of a rule.

Node types : RULE, TARGET, COND, ATOM, ATTR, VALUE, EFFECT
Edges      : RULE→TARGET, RULE→COND, RULE→EFFECT, TARGET/COND→ATOM,
             ATOM→ATTR, ATOM→VALUE  (all edges are added in both directions
             for message passing).

Node features
  * node type (one-hot id)
  * hashed token id (stable for unseen attribute names / values)
  * 4 numeric features for VALUE nodes of numeric atoms:
        [is_numeric, lo, hi, width]  normalised by the attribute bounds, so the
    GNN can learn *interval proximity* instead of treating "hour ≥ 17" and
    "hour ∈ 8‥18" as unrelated tokens.

Ontology (core.ontology) is injected structurally:
  * an atom on an alias attribute is also linked to the canonical ATTR node and
    its VALUE nodes are linked to the canonical VALUE nodes
    (`val:jobTitle=physician` —— `val:role=doctor`);
  * a VALUE node is linked to its parent in the value hierarchy
    (`val:role=senior_doctor` —— `val:role=doctor`).
So two rules written in different vocabularies share neighbourhoods *before*
any training, and contrastive training sharpens it.

Graph building modes
  build_rule_graph(r)            one rule, private leaves (what the embedder sees at inference)
  build_batch_graph(rules)       many rules, private leaves, one disconnected graph
                                 (what training sees — identical per-rule structure)
  build_policy_graph(p)          shared ATTR/VALUE/EFFECT leaves (analysis / visualisation only)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ecd.core import Atom, AttributeSpace, Policy, Rule

NODE_TYPES = ["RULE", "TARGET", "COND", "ATOM", "ATTR", "VALUE", "EFFECT"]
TYPE_IDX = {t: i for i, t in enumerate(NODE_TYPES)}
VOCAB_BUCKETS = 50_000
NUM_FEATS = 4


def tok_id(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % VOCAB_BUCKETS


@dataclass
class Graph:
    node_type: list[int] = field(default_factory=list)
    node_tok: list[int] = field(default_factory=list)
    node_label: list[str] = field(default_factory=list)
    node_num: list[tuple[float, float, float, float]] = field(default_factory=list)
    node_graph: list[int] = field(default_factory=list)        # which rule (sub-graph) a node belongs to
    edges: list[tuple[int, int]] = field(default_factory=list)
    rule_nodes: dict[str, int] = field(default_factory=dict)   # rule id -> RULE node index
    rule_graphs: dict[str, int] = field(default_factory=dict)  # rule id -> sub-graph index
    _shared: dict[str, int] = field(default_factory=dict)
    _cur_graph: int = -1

    def add(self, ntype: str, label: str, shared: bool = False,
            num: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> int:
        if shared and label in self._shared:
            return self._shared[label]
        idx = len(self.node_type)
        self.node_type.append(TYPE_IDX[ntype])
        self.node_tok.append(tok_id(label))
        self.node_label.append(label)
        self.node_num.append(num)
        self.node_graph.append(self._cur_graph)
        if shared:
            self._shared[label] = idx
        return idx

    @property
    def num_graphs(self) -> int:
        return self._cur_graph + 1

    def link(self, a: int, b: int) -> None:
        self.edges.append((a, b))
        self.edges.append((b, a))

    @property
    def num_nodes(self) -> int:
        return len(self.node_type)


_NUMERIC_OPS = {"lt", "le", "gt", "ge", "range"}


def _is_numeric_atom(a: Atom) -> bool:
    if a.op in _NUMERIC_OPS:
        return True
    return a.op == "eq" and isinstance(a.value, (int, float)) and not isinstance(a.value, bool)


def _num_feats(a: Atom, space: AttributeSpace | None) -> tuple[float, float, float, float]:
    attr = space.get(a.attr) if space else None
    lo_d = attr.lo if (attr and attr.lo is not None) else 0.0
    hi_d = attr.hi if (attr and attr.hi is not None) else 100.0
    span = max(hi_d - lo_d, 1e-9)
    v = a.value
    if a.op == "range":
        lo, hi = v
    elif a.op in ("lt", "le"):
        lo, hi = lo_d, v
    elif a.op in ("gt", "ge"):
        lo, hi = v, hi_d
    else:  # eq
        lo, hi = v, v
    n = lambda x: (float(x) - lo_d) / span  # noqa: E731
    return (1.0, n(lo), n(hi), n(hi) - n(lo))


def add_rule(g: Graph, r: Rule, share_leaves: bool = True,
             space: AttributeSpace | None = None, *, local_share: bool = True) -> int:
    """Add one rule. `local_share=True` shares ATTR/VALUE nodes *within* the rule
    (so an alias value and its canonical value node coincide when both occur)."""
    ont = space.ontology if space else None
    local: dict[str, int] = {}

    def leaf(ntype: str, label: str, num=(0.0, 0.0, 0.0, 0.0)) -> int:
        if share_leaves:
            return g.add(ntype, label, shared=True, num=num)
        if local_share and label in local:
            return local[label]
        idx = g.add(ntype, label, num=num)
        local[label] = idx
        return idx

    # RULE / TARGET / COND nodes carry *no* identity token: a per-rule token would
    # let the GNN memorise a lookup table for training rules and generalise to nothing.
    g._cur_graph += 1
    g.rule_graphs[r.id] = g._cur_graph
    rn = g.add("RULE", "rule")
    g.rule_nodes[r.id] = rn
    en = leaf("EFFECT", f"effect:{r.effect.value}")
    g.link(rn, en)
    for ptype, pred in (("TARGET", r.target), ("COND", r.condition)):
        pn = g.add(ptype, ptype)
        g.link(rn, pn)
        for a in pred.atoms:
            an = g.add("ATOM", f"atom:{a.attr}:{a.op}")
            g.link(pn, an)
            attr_n = leaf("ATTR", f"attr:{a.attr}")
            g.link(an, attr_n)
            canon_attr = ont.canonical_attr(a.attr) if ont else a.attr
            if canon_attr != a.attr:                       # alias → canonical attribute
                g.link(an, leaf("ATTR", f"attr:{canon_attr}"))
            if _is_numeric_atom(a):
                vn = leaf("VALUE", f"val:{a.attr}:num", _num_feats(a, space))
                g.link(an, vn)
                continue
            vals = a.value if isinstance(a.value, tuple) else (a.value,)
            for v in vals:
                vn = leaf("VALUE", f"val:{a.attr}={v}")
                g.link(an, vn)
                if ont is None:
                    continue
                cv = ont.canonical_value(a.attr, v)
                if canon_attr != a.attr or cv != v:        # alias value → canonical value
                    g.link(vn, leaf("VALUE", f"val:{canon_attr}={cv}"))
                # hierarchy: link each value to its ancestors
                cur, seen = cv, 0
                while seen < 8:
                    par = ont.parent(canon_attr, cur)
                    if par is None:
                        break
                    pn_ = leaf("VALUE", f"val:{canon_attr}={par}")
                    g.link(leaf("VALUE", f"val:{canon_attr}={cur}"), pn_)
                    cur, seen = par, seen + 1
        if pred.raw_expr:
            an = g.add("ATOM", f"raw:{pred.raw_expr}")
            g.link(pn, an)
            for name in _raw_names(pred.raw_expr):
                g.link(an, leaf("ATTR", f"attr:{ont.canonical_attr(name) if ont else name}"))
    return rn


def _raw_names(raw: str) -> list[str]:
    import re
    kw = {"and", "or", "not", "ite", "true", "false", "div", "mod", "abs", "distinct"}
    return [n for n in set(re.findall(r"[A-Za-z_][A-Za-z0-9_.:-]*", raw)) if n not in kw]


def build_rule_graph(r: Rule, space: AttributeSpace | None = None) -> Graph:
    g = Graph()
    add_rule(g, r, share_leaves=False, space=space)
    return g


def build_batch_graph(rules: list[Rule], space: AttributeSpace | None = None) -> Graph:
    """Disconnected union of per-rule graphs (identical structure to build_rule_graph)."""
    g = Graph()
    for r in rules:
        add_rule(g, r, share_leaves=False, space=space)
    return g


def build_policy_graph(p: Policy) -> Graph:
    g = Graph()
    for r in p:
        add_rule(g, r, share_leaves=True, space=p.space)
    return g

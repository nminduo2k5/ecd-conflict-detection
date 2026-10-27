"""Alias / hierarchy ontology: every exact step (Tier A, Z3, Event-B) must agree."""
from ecd.core import (AliasSpec, Atom, Attribute, AttrType, Effect, Ontology, Policy, Predicate, Rule)
from ecd.filtering import Pruner, decide
from ecd.verification import SMTChecker
from ecd.verification.eventb import base_context


def _policy():
    p = Policy("hetero")
    p.space.add(Attribute("role", AttrType.CATEGORICAL, ("doctor", "nurse", "senior_doctor")))
    p.space.add(Attribute("jobTitle", AttrType.CATEGORICAL, ("physician", "rn")))
    p.space.ontology.aliases["jobTitle"] = AliasSpec("role", {"physician": "doctor", "rn": "nurse"})
    p.space.ontology.hierarchy["role"] = {"senior_doctor": "doctor"}
    p.add(Rule("a", Predicate((Atom("role", "eq", "doctor"),)), Predicate(), Effect.PERMIT))
    p.add(Rule("b", Predicate((Atom("jobTitle", "eq", "physician"),)), Predicate(), Effect.DENY))
    p.add(Rule("c", Predicate((Atom("jobTitle", "eq", "rn"),)), Predicate(), Effect.DENY))
    p.add(Rule("d", Predicate((Atom("role", "eq", "senior_doctor"),)), Predicate(), Effect.DENY))
    return p


def test_canonicalisation():
    p = _policy()
    ont = p.space.ontology
    # alias → canonical, then hierarchy: doctor ⇒ {doctor, senior_doctor}
    assert ont.canonical_atom(Atom("jobTitle", "eq", "physician")) == Atom("role", "in", ("doctor", "senior_doctor"))
    assert ont.canonical_atom(Atom("role", "eq", "doctor")) == Atom("role", "in", ("doctor", "senior_doctor"))
    assert ont.canonical_atom(Atom("jobTitle", "eq", "rn")) == Atom("role", "eq", "nurse")
    assert ont.canonical_atom(Atom("role", "neq", "nurse")) == Atom("role", "neq", "nurse")
    assert ont.canonical_atom(Atom("role", "eq", "senior_doctor")) == Atom("role", "eq", "senior_doctor")


def test_smt_sees_alias_and_hierarchy():
    p = _policy()
    chk = SMTChecker(p.space)
    assert chk.check_pair(p.get("a"), p.get("b")).conflict          # doctor ≡ physician
    assert not chk.check_pair(p.get("a"), p.get("c")).overlap       # doctor vs nurse
    assert chk.check_pair(p.get("a"), p.get("d")).conflict          # senior_doctor ⊑ doctor
    cex = chk.check_pair(p.get("a"), p.get("b")).counterexample
    assert cex["role"] == "doctor"


def test_tier_a_sound_under_ontology():
    p = _policy()
    pr = Pruner(p.space.ontology)
    chk = SMTChecker(p.space)
    rules = list(p)
    for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
            if pr.decide(rules[i], rules[j]).pruned:
                assert not chk.check_pair(rules[i], rules[j]).conflict
    # alias-aware pruning: role=doctor vs jobTitle=rn is provably disjoint
    assert pr.decide(p.get("a"), p.get("c")).pruned
    # without ontology the same pair is NOT pruned (different attribute names) — still sound
    assert not decide(p.get("a"), p.get("c")).pruned


def test_eventb_uses_canonical_vocabulary():
    p = _policy()
    ctx = base_context(p, [(p.get("a"), p.get("b"))])
    txt = ctx.render()
    assert "jobTitle" not in txt                     # alias attribute never appears
    assert "app_b = {q · q ∈ REQ ∧ role(q) ∈ {role_doctor, role_senior_doctor} | q}" in txt
    assert "app_c = {q · q ∈ REQ ∧ role(q) = role_nurse | q}" in txt
    assert "role_senior_doctor" in txt               # hierarchy value is part of the carrier set


def test_policy_roundtrip_keeps_ontology(tmp_path):
    p = _policy()
    p.save(tmp_path / "p.json")
    q = Policy.load(tmp_path / "p.json")
    assert q.space.ontology.aliases["jobTitle"].canonical == "role"
    assert q.space.ontology.hierarchy["role"] == {"senior_doctor": "doctor"}
    assert Ontology.from_dict(q.space.ontology.to_dict()).to_dict() == q.space.ontology.to_dict()

from ecd.core import Atom, Predicate, Rule, Effect, Policy, Attribute, AttrType
from ecd.verification import SMTChecker


def test_conflict_with_counterexample(hospital):
    res = SMTChecker(hospital.space).check_pair(hospital.get("r1"), hospital.get("r2"))
    assert res.conflict
    assert res.counterexample["role"] == "doctor"
    assert 17 <= res.counterexample["hour"] <= 18


def test_no_overlap(hospital):
    res = SMTChecker(hospital.space).check_pair(hospital.get("r1"), hospital.get("r4"))
    assert not res.overlap


def test_raw_expr():
    p = Policy("raw")
    p.space.add(Attribute("x", AttrType.NUMERIC))
    p.space.add(Attribute("y", AttrType.NUMERIC))
    a = Rule("a", Predicate((Atom("x", "range", (0, 10)),)), Predicate((), "(> (+ x y) 15)"), Effect.PERMIT)
    b = Rule("b", Predicate((Atom("y", "lt", 3),)), Predicate(), Effect.DENY)
    res = SMTChecker(p.space).check_pair(a, b)
    assert not res.overlap  # x≤10 ∧ y<3 ⇒ x+y<13, so x+y>15 is UNSAT
    c = Rule("c", Predicate((Atom("y", "gt", 8),)), Predicate(), Effect.DENY)
    assert SMTChecker(p.space).check_pair(a, c).conflict  # x=10,y=9 works

from ecd.core import Atom, Predicate, Rule, Effect, Policy


def test_roundtrip(hospital):
    d = hospital.to_dict()
    p2 = Policy.from_dict(d)
    assert [r.id for r in p2] == ["r1", "r2", "r3", "r4"]
    assert p2.get("r1") == hospital.get("r1")


def test_content_hash_changes_with_effect():
    a = Rule("x", Predicate(), Predicate(), Effect.PERMIT)
    b = Rule("x", Predicate(), Predicate(), Effect.DENY)
    assert a.content_hash() != b.content_hash()


def test_bad_op():
    import pytest
    with pytest.raises(ValueError):
        Atom("a", "matches", 1)

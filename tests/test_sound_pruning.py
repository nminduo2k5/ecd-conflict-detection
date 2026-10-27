from ecd.filtering import decide, PruneReason
from ecd.verification import SMTChecker


def test_same_effect_pruned(hospital):
    d = decide(hospital.get("r2"), hospital.get("r3"))
    assert d.pruned and d.reason is PruneReason.SAME_EFFECT


def test_disjoint_categorical(hospital):
    d = decide(hospital.get("r1"), hospital.get("r3"))
    assert d.pruned and d.reason is PruneReason.DISJOINT_ATTR and d.attr == "role"


def test_disjoint_numeric(hospital):
    d = decide(hospital.get("r1"), hospital.get("r4"))
    assert d.pruned and d.attr == "hour"


def test_overlap_kept(hospital):
    assert not decide(hospital.get("r1"), hospital.get("r2")).pruned


def test_soundness_against_smt(hospital):
    """Theorem 1: everything pruned must be UNSAT for the SMT oracle."""
    chk = SMTChecker(hospital.space)
    rules = list(hospital)
    for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
            d = decide(rules[i], rules[j])
            if d.pruned:
                assert not chk.check_pair(rules[i], rules[j]).conflict

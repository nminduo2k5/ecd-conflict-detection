from benchmark.generator.synthetic import GenConfig, SyntheticGenerator
from ecd.verification import SMTChecker


def test_planted_conflicts_are_real():
    gen = SyntheticGenerator(GenConfig(n_rules=200, conflict_rate=0.1, seed=1))
    pol = gen.generate()
    chk = SMTChecker(pol.space)
    for a, b in gen.planted:
        assert chk.check_pair(pol.get(a), pol.get(b)).conflict, (a, b)


def test_delta_sequence_applies():
    from ecd.core import apply_delta
    gen = SyntheticGenerator(GenConfig(n_rules=100, n_deltas=30, seed=2))
    pol = gen.generate()
    for d in gen.deltas(pol):
        apply_delta(pol, d)
    assert pol.version == 30

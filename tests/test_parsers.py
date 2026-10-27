from pathlib import Path

import pytest

from ecd.core import Atom
from ecd.parsers import load_any
from ecd.parsers.xacml import XACMLParser
from ecd.verification import SMTChecker
from ecd.pipeline import Detector

FX = Path(__file__).parent / "fixtures"
RAW = Path(__file__).resolve().parents[1] / "data/raw/xacml/xpa/Experiments"


def test_xacml():
    pol = load_any(FX / "sample_xacml.xml")
    assert [r.id for r in pol] == ["doctor-read", "night-deny"]
    r = pol.get("night-deny")
    assert r.condition.atoms[0].op == "gt" and r.condition.atoms[0].value == 22
    assert not r.meta.get("approx")
    assert SMTChecker(pol.space).check_pair(*list(pol)).conflict


def test_iam():
    pol = load_any(FX / "sample_iam.json")
    assert len(pol) == 2
    assert SMTChecker(pol.space).check_pair(*list(pol)).conflict


def test_xacml_dnf_and_condition_translation():
    pol = load_any(FX / "sample_xacml_complex.xml")
    ids = [r.id for r in pol]
    # target with two AllOf alternatives × policy-level target → two sub-rules of `multi`
    assert "multi#0" in ids and "multi#1" in ids
    m0 = pol.get("multi#0")
    assert Atom("access-subject.role", "eq", "doctor") in m0.target.atoms
    assert Atom("resource.resource-id", "eq", "record") in m0.target.atoms      # inherited from <Policy><Target>
    # not(string-is-in("administrator", subject-id)) → neq
    nadm = pol.get("not-admin")
    assert Atom("access-subject.subject-id", "neq", "administrator") in nadm.condition.atoms
    # and(boolean-equal(one-and-only(x), true), integer-greater-than-or-equal(integer-subtract(a, b), 5))
    ar = pol.get("arith")
    assert Atom("resource.signed", "eq", True) in ar.condition.atoms
    assert ar.condition.raw_expr == "(>= (- access-subject.age environment.min-age) 5)"
    assert not ar.meta.get("approx")
    # regexp → over-approximated and flagged, but the policy still checks without crashing
    assert pol.get("regex").meta.get("approx") is True
    rep = Detector(pol, None, use_faiss=False).full_check()
    assert {(c.a, c.b) for c in rep.conflicts} >= {("multi#0", "not-admin")}


@pytest.mark.skipif(not RAW.exists(), reason="real XACML policies not downloaded")
@pytest.mark.parametrize("rel", ["itrust3/itrust3.xml", "HL7/HL7.xml", "gpms/gpms.xml",
                                 "pluto3/pluto3.xml", "fedora-rule3/fedora-rule3.xml", "Simpson1/Simpson1.xml"])
def test_real_xacml_parses_and_checks(rel):
    parser = XACMLParser()
    pol = parser.parse(RAW / rel)
    assert len(pol) >= parser.stats["rules"] > 0
    rep = Detector(pol, None, use_faiss=False).full_check()     # must not raise (Z3 encoding of every rule)
    assert rep.pairs_verified <= rep.pairs_total

from xml.etree import ElementTree as ET

from ecd.core import Atom, Delta, DeltaKind, Effect, Predicate, Rule
from ecd.embedding import HashedAtomEmbedder
from ecd.pipeline import Detector
from ecd.verification.eventb import base_context, predicate_to_eventb, raw_to_eventb, count_theorems


def test_context_render(hospital):
    rules = list(hospital)
    ctx = base_context(hospital, [(rules[0], rules[1])])
    txt = ctx.render()
    assert "context PolicyCtx_v0" in txt
    assert "sets REQ role_T" in txt
    assert "theorem @thm_r1_r2 app_r1 ∩ app_r2 = ∅" in txt
    assert "role(q) = role_doctor" in predicate_to_eventb(rules[0].applicability)


def test_buc_is_well_formed_xml(hospital):
    rules = list(hospital)
    ctx = base_context(hospital, [(rules[0], rules[1])])
    root = ET.fromstring(ctx.render_buc())
    assert root.tag == "org.eventb.core.contextFile"
    thms = [e for e in root if e.get("org.eventb.core.theorem") == "true"]
    assert len(thms) == 1 and thms[0].get("org.eventb.core.label") == "thm_r1_r2"
    ctx2 = base_context(hospital, [], name="PolicyCtx_v1", extends="PolicyCtx_v0", only_rules=set())
    root2 = ET.fromstring(ctx2.render_buc())
    assert root2[0].tag == "org.eventb.core.extendsContext"


def test_raw_arithmetic_translation():
    txt, ok = raw_to_eventb("(> (+ hour age) 120)")
    assert ok and txt == "(hour(q) + age(q)) > 120"
    txt, ok = raw_to_eventb("(and (>= (- age bart-age) 55) (not (= x 3)))")
    assert ok and "age(q) − bart_age(q)" in txt and "¬(x(q) = 3)" in txt
    txt, ok = raw_to_eventb("(xacml-condition 'foo')")
    assert not ok and "TRUE = TRUE" in txt


def test_modify_gets_versioned_constant(hospital, tmp_path):
    det = Detector(hospital, HashedAtomEmbedder(), eventb_out=tmp_path, use_faiss=False)
    det.full_check()
    r1 = hospital.get("r1")
    changed = Rule("r1", r1.target, Predicate((Atom("hour", "range", (8, 20)),)), r1.effect)
    det.apply(Delta(DeltaKind.MODIFY, changed, "r1"))
    v1 = (tmp_path / "PolicyCtx_v1.ctx").read_text(encoding="utf-8")
    assert "extends PolicyCtx_v0" in v1
    assert "constants app_r1__v1" in v1               # no redefinition of app_r1
    assert "@def_app_r1__v1" in v1 and "@def_app_r1 " not in v1
    assert "thm_r1__v1_r2 app_r1__v1 ∩ app_r2 = ∅" in v1
    # removal: empty extension, nothing re-declared
    det.apply(Delta(DeltaKind.REMOVE, rule_id="r2"))
    v2 = (tmp_path / "PolicyCtx_v2.ctx").read_text(encoding="utf-8")
    assert v2.strip().splitlines() == ["context PolicyCtx_v2", "extends PolicyCtx_v1", "end"]
    assert (tmp_path / "PolicyCtx_v2.buc").exists()
    st = det.eventb.po_stats()
    assert [s["po_new"] for s in st] == [count_theorems(tmp_path / f"PolicyCtx_v{i}.ctx") for i in range(3)]
    assert st[2]["po_inherited"] == st[0]["po_new"] + st[1]["po_new"]


def test_effect_only_pairs_get_theorems(hospital, tmp_path):
    det = Detector(hospital, None, eventb_out=tmp_path, use_faiss=False)
    rep = det.full_check()
    assert count_theorems(tmp_path / "PolicyCtx_v0.ctx") == rep.pairs_verified

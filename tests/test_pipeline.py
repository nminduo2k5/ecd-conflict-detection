import json
from ecd.core import Atom, Delta, DeltaKind, Effect, Predicate, Rule
from ecd.embedding import HashedAtomEmbedder
from ecd.filtering import Mode
from ecd.pipeline import Detector


def test_full_then_delta(hospital, tmp_path):
    det = Detector(hospital, HashedAtomEmbedder(), Mode.EXACT, eventb_out=tmp_path, use_faiss=False)
    rep = det.full_check()
    assert {(c.a, c.b) for c in rep.conflicts} == {("r1", "r2")}
    assert rep.pairs_pruned >= 3
    assert (tmp_path / "PolicyCtx_v0.ctx").exists()

    # add a rule that conflicts with r1
    new = Rule("r5", Predicate((Atom("role", "eq", "doctor"),)),
               Predicate((Atom("hour", "eq", 10),)), Effect.DENY)
    rep2 = det.apply(Delta(DeltaKind.ADD, new))
    assert {(c.a, c.b) for c in rep2.conflicts} == {("r5", "r1")}
    assert rep2.pairs_total == 4            # only pairs touching r5
    assert (tmp_path / "PolicyCtx_v1.ctx").read_text().startswith("context PolicyCtx_v1\nextends PolicyCtx_v0")

    # remove r1 → conflicts disappear from the ledger
    det.apply(Delta(DeltaKind.REMOVE, rule_id="r1"))
    assert det.all_conflicts() == []


def test_approximate_mode_truncates(hospital):
    det = Detector(hospital, HashedAtomEmbedder(), Mode.APPROXIMATE, top_k=1, use_faiss=False)
    rep = det.full_check()
    assert rep.pairs_verified <= 4

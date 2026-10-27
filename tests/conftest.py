import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pytest
from ecd.core import Atom, Attribute, AttrType, Effect, Policy, Predicate, Rule


@pytest.fixture
def hospital() -> Policy:
    """Motivating example: a small healthcare policy."""
    p = Policy("hospital")
    p.space.add(Attribute("role", AttrType.CATEGORICAL, ("doctor", "nurse", "admin")))
    p.space.add(Attribute("hour", AttrType.NUMERIC, lo=0, hi=23))
    p.add(Rule("r1", Predicate((Atom("role", "eq", "doctor"),)),
               Predicate((Atom("hour", "range", (8, 18)),)), Effect.PERMIT))
    p.add(Rule("r2", Predicate((Atom("role", "in", ("doctor", "nurse")),)),
               Predicate((Atom("hour", "ge", 17),)), Effect.DENY))         # overlaps r1 at hour 17-18
    p.add(Rule("r3", Predicate((Atom("role", "eq", "admin"),)), Predicate(), Effect.DENY))  # disjoint from r1
    p.add(Rule("r4", Predicate((Atom("role", "eq", "doctor"),)),
               Predicate((Atom("hour", "lt", 8),)), Effect.DENY))           # disjoint range from r1
    return p

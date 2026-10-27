"""Verified-pair cache = the set of discharged proof obligations.

Key: (rule_id_a, rule_id_b, hash_a, hash_b). The content hashes make the
cache self-invalidating: a MODIFY produces new hashes, so stale POs are
never reused by accident even if `invalidate` is skipped.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ecd.core import Rule, pair_key


@dataclass
class POEntry:
    conflict: bool
    counterexample: dict[str, Any]
    verified_at_version: int


@dataclass
class POCache:
    entries: dict[str, POEntry] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    @staticmethod
    def _key(a: Rule, b: Rule) -> str:
        (x, y) = pair_key(a.id, b.id)
        ha, hb = (a, b) if x == a.id else (b, a)
        return f"{x}|{y}|{ha.content_hash()}|{hb.content_hash()}"

    def get(self, a: Rule, b: Rule) -> POEntry | None:
        e = self.entries.get(self._key(a, b))
        if e is None:
            self.misses += 1
        else:
            self.hits += 1
        return e

    def put(self, a: Rule, b: Rule, entry: POEntry) -> None:
        self.entries[self._key(a, b)] = entry

    def invalidate_rule(self, rid: str) -> int:
        dead = [k for k in self.entries if k.split("|")[0] == rid or k.split("|")[1] == rid]
        for k in dead:
            del self.entries[k]
        return len(dead)

    @property
    def reuse_rate(self) -> float:
        t = self.hits + self.misses
        return 0.0 if t == 0 else self.hits / t

    def reset_counters(self) -> None:
        self.hits = self.misses = 0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({k: v.__dict__ for k, v in self.entries.items()}))

    @staticmethod
    def load(path: str | Path) -> "POCache":
        raw = json.loads(Path(path).read_text())
        return POCache({k: POEntry(**v) for k, v in raw.items()})

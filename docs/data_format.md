# Native policy JSON format

```json
{
  "name": "hospital",
  "version": 0,
  "combining": "deny-overrides",        // permit-overrides | first-applicable | only-one-applicable
  "attributes": {
    "role":  {"type": "categorical", "domain": ["doctor", "nurse", "admin"]},
    "hour":  {"type": "numeric", "lo": 0, "hi": 23},
    "onCall": {"type": "boolean"}
  },
  "rules": [
    {"id": "r1",
     "target":    {"atoms": [{"attr": "role", "op": "eq", "value": "doctor"}]},
     "condition": {"atoms": [{"attr": "hour", "op": "range", "value": [8, 18]}],
                   "raw_expr": "(> (+ hour riskScore) 30)"},     // optional SMT-LIB2 conjunct
     "effect": "Permit",
     "meta": {"source": "xacml"}}
  ]
}
```

Atom ops: `eq neq in notin lt le gt ge range any`. `range` value = `[lo, hi]` inclusive.

Delta file (`deltas.json`): list of
`{"kind": "add"|"modify"|"remove", "rule": {...}|null, "rule_id": "...", "position": null}`.

Dataset folder layout produced by `ecd gen`:
```
policy_v0.json  deltas.json  planted_conflicts.json  config.json  [truth_conflicts.json]
```
`truth_conflicts.json` is written by `experiments/baselines/full_smt.py` on first RQ2 run.

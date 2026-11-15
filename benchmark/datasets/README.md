# Datasets

Rules are counted after normalisation; conflicts are the exhaustive pairwise Z3 ground truth (paper, Table 1).
Synthetic policies come with 100 changes each, real policies with 50.

| Name | Rules / conflicts | Purpose in the paper | Source | Converter |
|---|---|---|---|---|
| iTrust 64, HL7 22, gpms 210, pluto 176, conference 15, fedora 15, kmarket 3, Simpson 4 (XACML 3.0) | 3–210 / 0–214 | real policies (exact-mode recall, pruning); RQ1 on gpms and pluto | XPA repository `Experiments/<name>/<name>.xml` | `benchmark/convert_real.py` → `data/processed/real/xacml_<name>` |
| obligation (XACML 3.0) | 2 | converted but **not used in the paper** | XPA repository | `convert_real.py` → `real/xacml_obligation` |
| AWS managed IAM policies (merges of 100 / 300 policies) | 239 / 240, 756 / 761 | real policies; RQ1 on the 300-policy merge (one rule per statement, wildcard actions expanded with the IAM action registry) | SummitRoute mirror + MAMIP `action-registry.json` | `convert_real.py` → `real/aws_managed_{100,300}` |
| Kubernetes bootstrap ClusterRoles (29 roles) | 138 / 0 | real policies, allow-only | kubernetes/kubernetes testdata | `convert_real.py` → `real/k8s_clusterroles` |
| `synthetic_1k` | 1 000 / 9 983 | RQ1, RQ2, RQ3: 12 attributes (9 categorical × 12 values, 3 numeric), two core attributes and four atoms per rule; 5 % planted conflicts, 5 % arithmetic conditions, 20 % in a second vocabulary with an alias ontology | `ecd gen` | `benchmark/generator/synthetic.py` |
| `complex_1k` | 1 000 / 9 785 | RQ2: as above, 60 % cross-attribute arithmetic conditions | `ecd gen` | `benchmark/generator/synthetic.py` |
| Synthetic 1k–100k | parametrised | RQ4 (30 seeds per size) | `ecd gen` | `benchmark/generator/synthetic.py` |
| `hetero_1k_h00/h25/h50` | 1 000 / 11 775 | RQ1 (`h50`), RQ5: 10 % planted and arithmetic, a role hierarchy, 0 / 25 / 50 % in a second vocabulary | `ecd gen --hetero h --hierarchy 0.2` | — |
| Amazon Employee Access (Kaggle) | 32k rows | optional, needs Kaggle credentials | Kaggle CLI | not used in the reported runs |

Continue, LMS and VMS (the XACML testing benchmarks of Martin & Xie) are not redistributed
publicly in XACML 3.0 form; the XPA set above (iTrust, pluto, gpms, HL7, …) is the standard
XACML 3.0 benchmark used by XPA / XACMUT-style work and covers the same purpose.

Every dataset folder contains `policy_v0.json` (+ `ontology` when aliases exist), `deltas.json`,
`planted_conflicts.json` (empty for real data), and — once an experiment has run —
`truth_conflicts.json` (exhaustive Z3 ground truth) and `labels.json` (Z3 overlap labels).

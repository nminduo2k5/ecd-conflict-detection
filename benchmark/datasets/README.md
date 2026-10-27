# Datasets

| Name | Size (rules after normalisation) | Purpose | Source | Converter |
|---|---|---|---|---|
| iTrust, HL7, gpms, pluto, conference, fedora, kmarket, obligation, Simpson (XACML 3.0) | 2 – 210 | correctness on real policies, RQ1 on real vocabularies | XPA repository `Experiments/<name>/<name>.xml` | `benchmark/convert_real.py` → `data/processed/real/xacml_<name>` |
| AWS managed IAM policies (100 / 300 policies merged) | 239 / 756 statements | near-real, heterogeneous (wildcard actions expanded with the IAM action registry) | SummitRoute mirror + MAMIP `action-registry.json` | `convert_real.py` → `real/aws_managed_{100,300}` |
| Kubernetes bootstrap ClusterRoles | 138 | allow-only overlap/redundancy, heterogeneous merge | kubernetes/kubernetes testdata | `convert_real.py` → `real/k8s_clusterroles` |
| Synthetic 1k–100k (`ecd gen`) | parametrised | RQ2–RQ4 | — | `benchmark/generator/synthetic.py` |
| Heterogeneous 1k (`hetero_1k_h00/h25/h50`) | 1 000 | RQ5: alias vocabularies + role hierarchy + raw conditions | `ecd gen --hetero h --hierarchy 0.2` | — |
| Amazon Employee Access (Kaggle) | 32k rows | optional, needs Kaggle credentials | Kaggle CLI | not used in the reported runs |

Continue, LMS and VMS (the XACML testing benchmarks of Martin & Xie) are not redistributed
publicly in XACML 3.0 form; the XPA set above (iTrust, pluto, gpms, HL7, …) is the standard
XACML 3.0 benchmark used by XPA / XACMUT-style work and covers the same purpose.

Every dataset folder contains `policy_v0.json` (+ `ontology` when aliases exist), `deltas.json`,
`planted_conflicts.json` (empty for real data), and — once an experiment has run —
`truth_conflicts.json` (exhaustive Z3 ground truth) and `labels.json` (Z3 overlap labels).

# Results summary

## RQ1 — overlap-awareness of the embedding (AUC on unseen rules, rule-level hold-out)

| dataset | rules | pairs (test) | SBERT | hash | hash+ontology | GNN untrained | **GNN+contrastive** |
|---|---|---|---|---|---|---|---|
| synthetic_1k | 1000 | 14184 | 0.482 | 0.715 | 0.807 | 0.586 | **0.850** |
| hetero_1k_h50 | 1000 | 13824 | 0.481 | 0.551 | 0.780 | 0.525 | **0.862** |
| aws_managed_300 | 756 | 7928 | 0.778 | 0.600 | – | 0.629 | **0.776** |
| xacml_gpms | 210 | 360 | 0.553 | 0.513 | – | 0.505 | **0.845** |
| xacml_pluto | 176 | 95 | 0.000 | 0.000 | – | 0.000 | **0.000** |
| xacml_itrust | 64 | 0 | nan | nan | – | – | **–** |

## RQ2 (rq2_pruning) — pair reduction / recall on synthetic_1k (1000 rules, 9983 true conflicts)

| variant | pairs to Z3 | pruning | recall | precision | Z3 calls to 1st conflict | time (s) |
|---|---|---|---|---|---|---|
| no_tierA | 249996 | 50.0 % | 100.0 % | 100.0 % | 156 | 114.5 |
| tierA_only | 10125 | 98.0 % | 100.0 % | 100.0 % | 1 | 17.7 |
| exact_hash | 10125 | 98.0 % | 100.0 % | 100.0 % | 1 | 18.3 |
| approx_hash_k10 | 6316 | 98.0 % | 62.3 % | 100.0 % | 1 | 12.4 |
| approx_hash_k25 | 9148 | 98.0 % | 90.3 % | 100.0 % | 1 | 20.6 |
| approx_hash_k50 | 10076 | 98.0 % | 99.5 % | 100.0 % | 1 | 18.4 |
| exact_hash+ont | 10125 | 98.0 % | 100.0 % | 100.0 % | 1 | 18.2 |
| approx_hash+ont_k10 | 6316 | 98.0 % | 62.3 % | 100.0 % | 1 | 21.1 |
| approx_hash+ont_k25 | 9148 | 98.0 % | 90.3 % | 100.0 % | 1 | 19.2 |
| approx_hash+ont_k50 | 10076 | 98.0 % | 99.5 % | 100.0 % | 1 | 22.2 |
| exact_gnn | 10125 | 98.0 % | 100.0 % | 100.0 % | 1 | 37.1 |
| approx_gnn_k10 | 6316 | 98.0 % | 62.3 % | 100.0 % | 1 | 17.0 |
| approx_gnn_k25 | 9148 | 98.0 % | 90.3 % | 100.0 % | 1 | 21.6 |
| approx_gnn_k50 | 10076 | 98.0 % | 99.5 % | 100.0 % | 1 | 23.4 |
| exact_index (candidates only) | 112479 | – | 98.8 % | – | – | – |
| exact_index+ont (candidates only) | 125323 | – | 99.9 % | – | – | – |

Δ path (77 Δs that introduce a conflict), exact mode:

| ordering | Z3 calls before 1st conflict (mean ± sd) | verified pairs / Δ | ms / Δ | Wilcoxon p vs none | A12 vs none |
|---|---|---|---|---|---|
| none | 1 ± 0 | 16 ± 15 | 35 ± 31 | – | – |
| hash | 1 ± 0 | 16 ± 15 | 34 ± 31 | nan | 0.500 |
| hash+ont | 1 ± 0 | 16 ± 15 | 35 ± 31 | nan | 0.500 |
| gnn | 1 ± 0 | 16 ± 15 | 44 ± 37 | 0.0833 | 0.519 |

## RQ2 (rq2_complex) — pair reduction / recall on complex_1k (1000 rules, 9785 true conflicts)

| variant | pairs to Z3 | pruning | recall | precision | Z3 calls to 1st conflict | time (s) |
|---|---|---|---|---|---|---|
| no_tierA | 249964 | 50.0 % | 100.0 % | 100.0 % | 100 | 102.3 |
| tierA_only | 10708 | 97.9 % | 100.0 % | 100.0 % | 1 | 18.3 |
| exact_hash | 10708 | 97.9 % | 100.0 % | 100.0 % | 1 | 18.9 |
| approx_hash_k10 | 6741 | 97.9 % | 62.6 % | 100.0 % | 1 | 13.2 |
| approx_hash_k25 | 9819 | 97.9 % | 91.6 % | 100.0 % | 1 | 17.9 |
| approx_hash_k50 | 10668 | 97.9 % | 99.6 % | 100.0 % | 1 | 18.8 |
| exact_hash+ont | 10708 | 97.9 % | 100.0 % | 100.0 % | 1 | 19.3 |
| approx_hash+ont_k10 | 6741 | 97.9 % | 62.7 % | 100.0 % | 1 | 13.1 |
| approx_hash+ont_k25 | 9819 | 97.9 % | 91.5 % | 100.0 % | 1 | 23.3 |
| approx_hash+ont_k50 | 10668 | 97.9 % | 99.6 % | 100.0 % | 1 | 33.3 |
| exact_gnn | 10708 | 97.9 % | 100.0 % | 100.0 % | 1 | 27.6 |
| approx_gnn_k10 | 6741 | 97.9 % | 63.1 % | 100.0 % | 1 | 22.7 |
| approx_gnn_k25 | 9819 | 97.9 % | 91.8 % | 100.0 % | 1 | 25.1 |
| approx_gnn_k50 | 10668 | 97.9 % | 99.6 % | 100.0 % | 1 | 31.4 |
| exact_index (candidates only) | 99807 | – | 73.9 % | – | – | – |
| exact_index+ont (candidates only) | 126628 | – | 99.9 % | – | – | – |

Δ path (80 Δs that introduce a conflict), exact mode:

| ordering | Z3 calls before 1st conflict (mean ± sd) | verified pairs / Δ | ms / Δ | Wilcoxon p vs none | A12 vs none |
|---|---|---|---|---|---|
| none | 1 ± 0 | 18 ± 17 | 40 ± 36 | – | – |
| hash | 1 ± 0 | 18 ± 17 | 43 ± 41 | 1.0000 | 0.506 |
| hash+ont | 1 ± 0 | 18 ± 17 | 41 ± 36 | 0.2763 | 0.493 |
| gnn | 1 ± 0 | 18 ± 17 | 51 ± 43 | 0.4962 | 0.482 |

## RQ3 — incremental Event-B on synthetic_1k (1000 rules, 50 Δ)

- full check v0: 10125 THM POs, 23.3 s
- per Δ: new POs 18 ± 15, PO reuse rate 0.9983, incremental time 66 ± 35 ms
- full regeneration would produce 253457 ± 2672 POs per Δ; from-scratch re-verification takes 18336 ± 531 ms
- Event-B contexts (.ctx/.buc) in `H:\ecd-project\ecd\results_rerun\eventb\rq3`

## RQ4 — scalability of the Δ path (10 Δ per run, 30 seeds, hash embedder)

| rules | exact ms/Δ | approx (top-50) ms/Δ | pairs verified/Δ exact | Tier A ms/Δ | Wilcoxon p | A12 (approx<exact) | GNN embed all rules (ms) |
|---|---|---|---|---|---|---|---|
| 1,000 | 60 ± 26 | 60 ± 26 | 18 ± 5 | 7 ± 3 | 0.6408 | 0.501 | 386 |
| 5,000 | 224 ± 80 | 120 ± 32 | 78 ± 30 | 34 ± 7 | 0.0000 | 0.062 | 1614 |
| 10,000 | 502 ± 125 | 179 ± 37 | 165 ± 45 | 80 ± 19 | 0.0000 | 0.023 | 4348 |
| 20,000 | 994 ± 254 | 278 ± 77 | 324 ± 76 | 162 ± 31 | 0.0000 | 0.000 | 7136 |
| 50,000 | 2517 ± 600 | 556 ± 68 | 847 ± 223 | 466 ± 59 | 0.0000 | 0.000 | 17328 |
| 100,000 | 5150 ± 1396 | 982 ± 136 | 1710 ± 531 | 855 ± 115 | 0.0000 | 0.000 | 34169 |

## RQ5 — heterogeneous vocabularies / hierarchies (recall %, top-50 budget where applicable)

| dataset | hetero | method | pairs to Z3 | recall all | recall cross-vocab (no shared literal) | recall alias-planted | recall hierarchy-planted |
|---|---|---|---|---|---|---|---|
| hetero_1k_h00 | 0 % | *true conflicts* | 11775 | – | n=104 | n=0 | n=0 |
| hetero_1k_h00 | 0 % | exact_index | 124155 | 99.2 | 14.4 | – | – |
| hetero_1k_h00 | 0 % | exact_index+ont | 124351 | 99.2 | 14.4 | – | – |
| hetero_1k_h00 | 0 % | approx_hash_k5 | 4116 | 34.4 | 0.0 | – | – |
| hetero_1k_h00 | 0 % | approx_hash+ont_k5 | 4116 | 34.4 | 0.0 | – | – |
| hetero_1k_h00 | 0 % | approx_gnn_k5 | 4116 | 34.5 | 1.0 | – | – |
| hetero_1k_h00 | 0 % | approx_hash_k10 | 6846 | 57.3 | 2.9 | – | – |
| hetero_1k_h00 | 0 % | approx_hash+ont_k10 | 6846 | 57.2 | 2.9 | – | – |
| hetero_1k_h00 | 0 % | approx_gnn_k10 | 6846 | 57.3 | 1.9 | – | – |
| hetero_1k_h00 | 0 % | approx_hash_k50 | 11816 | 99.0 | 63.5 | – | – |
| hetero_1k_h00 | 0 % | approx_hash+ont_k50 | 11816 | 99.0 | 63.5 | – | – |
| hetero_1k_h00 | 0 % | approx_gnn_k50 | 11816 | 99.0 | 75.0 | – | – |
| hetero_1k_h00 | 0 % | exact_gnn | 11934 | 100.0 | 100.0 | – | – |
| hetero_1k_h00 | 0 % | tierA_only | 11934 | 100.0 | 100.0 | – | – |
| hetero_1k_h25 | 25 % | *true conflicts* | 11775 | – | n=4416 | n=28 | n=0 |
| hetero_1k_h25 | 25 % | exact_index | 91715 | 65.6 | 8.3 | 96.4 | – |
| hetero_1k_h25 | 25 % | exact_index+ont | 124351 | 99.2 | 98.0 | 100.0 | – |
| hetero_1k_h25 | 25 % | approx_hash_k5 | 4116 | 34.4 | 13.3 | 78.6 | – |
| hetero_1k_h25 | 25 % | approx_hash+ont_k5 | 4116 | 34.4 | 30.6 | 100.0 | – |
| hetero_1k_h25 | 25 % | approx_gnn_k5 | 4116 | 34.5 | 31.7 | 60.7 | – |
| hetero_1k_h25 | 25 % | approx_hash_k10 | 6846 | 57.3 | 33.7 | 89.3 | – |
| hetero_1k_h25 | 25 % | approx_hash+ont_k10 | 6846 | 57.2 | 53.6 | 100.0 | – |
| hetero_1k_h25 | 25 % | approx_gnn_k10 | 6846 | 57.4 | 53.6 | 82.1 | – |
| hetero_1k_h25 | 25 % | approx_hash_k50 | 11816 | 99.0 | 97.3 | 100.0 | – |
| hetero_1k_h25 | 25 % | approx_hash+ont_k50 | 11816 | 99.0 | 98.3 | 100.0 | – |
| hetero_1k_h25 | 25 % | approx_gnn_k50 | 11816 | 99.0 | 98.5 | 100.0 | – |
| hetero_1k_h25 | 25 % | exact_gnn | 11934 | 100.0 | 100.0 | 100.0 | – |
| hetero_1k_h25 | 25 % | tierA_only | 11934 | 100.0 | 100.0 | 100.0 | – |
| hetero_1k_h50 | 50 % | *true conflicts* | 11775 | – | n=5553 | n=56 | n=0 |
| hetero_1k_h50 | 50 % | exact_index | 81810 | 56.6 | 8.0 | 98.2 | – |
| hetero_1k_h50 | 50 % | exact_index+ont | 124351 | 99.2 | 98.4 | 100.0 | – |
| hetero_1k_h50 | 50 % | approx_hash_k5 | 4116 | 34.5 | 12.0 | 83.9 | – |
| hetero_1k_h50 | 50 % | approx_hash+ont_k5 | 4116 | 34.4 | 30.4 | 100.0 | – |
| hetero_1k_h50 | 50 % | approx_gnn_k5 | 4116 | 34.6 | 33.3 | 64.3 | – |
| hetero_1k_h50 | 50 % | approx_hash_k10 | 6846 | 57.3 | 33.2 | 94.6 | – |
| hetero_1k_h50 | 50 % | approx_hash+ont_k10 | 6846 | 57.2 | 54.1 | 100.0 | – |
| hetero_1k_h50 | 50 % | approx_gnn_k10 | 6846 | 57.4 | 56.0 | 78.6 | – |
| hetero_1k_h50 | 50 % | approx_hash_k50 | 11816 | 99.0 | 97.9 | 100.0 | – |
| hetero_1k_h50 | 50 % | approx_hash+ont_k50 | 11816 | 99.0 | 98.7 | 100.0 | – |
| hetero_1k_h50 | 50 % | approx_gnn_k50 | 11816 | 99.0 | 98.7 | 100.0 | – |
| hetero_1k_h50 | 50 % | exact_gnn | 11934 | 100.0 | 100.0 | 100.0 | – |
| hetero_1k_h50 | 50 % | tierA_only | 11934 | 100.0 | 100.0 | 100.0 | – |

## Real policies (exact mode, hash embedder)

| dataset | rules (norm.) | source rules | pairs | Tier A pruned | pairs to Z3 | conflicts | classes | recall | full check (ms) | Δ ms (mean ± sd) | approx top-50 recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xacml_itrust | 64 | 64 | 2016 | 100.0 % | 0 | 0 | {} | 100.0 % | 1 | 3 ± 6 | 100.0 % |
| xacml_hl7 | 22 | 19 | 231 | 92.6 % | 17 | 17 | {'order-dependent': 17} | 100.0 % | 31 | 3 ± 4 | 100.0 % |
| xacml_gpms | 210 | 97 | 21945 | 99.0 % | 214 | 214 | {'masked': 214} | 100.0 % | 489 | 7 ± 12 | 100.0 % |
| xacml_pluto | 176 | 21 | 15400 | 98.9 % | 175 | 175 | {'masked': 175} | 100.0 % | 309 | 4 ± 7 | 100.0 % |
| xacml_conference | 15 | 15 | 105 | 86.7 % | 14 | 14 | {'masked': 14} | 100.0 % | 21 | 4 ± 3 | 100.0 % |
| xacml_fedora | 15 | 12 | 105 | 81.9 % | 19 | 19 | {'masked': 19} | 100.0 % | 24 | 3 ± 3 | 100.0 % |
| xacml_kmarket | 3 | 3 | 3 | 33.3 % | 2 | 2 | {'masked': 2} | 100.0 % | 5 | 2 ± 1 | 100.0 % |
| xacml_simpson | 4 | 4 | 6 | 50.0 % | 3 | 3 | {'masked': 3} | 100.0 % | 19 | 2 ± 2 | 100.0 % |
| aws_managed_100 | 239 | 100 | 28441 | 99.2 % | 240 | 240 | {'masked': 240} | 100.0 % | 926 | 51 ± 117 | 100.0 % |
| aws_managed_300 | 756 | 300 | 285390 | 99.7 % | 761 | 761 | {'masked': 761} | 100.0 % | 3800 | 238 ± 554 | 36.9 % |
| k8s_clusterroles | 138 | – | 9453 | 100.0 % | 0 | 0 | {} | 100.0 % | 7 | 14 ± 28 | 100.0 % |

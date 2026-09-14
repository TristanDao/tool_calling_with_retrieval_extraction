# Method 2 — legacy run re-scored

Protocol: unseen negative exposure; candidate-scope. Cost unavailable.

| Dataset | N | Strict ArgA | Normalized ArgA | Oracle ArgA | Gap | Arg F1 | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benchmark | 10555 | 39.40% | 39.84% | 42.66% | 2.82% | 61.51% | 56.30 | 79.59 |
| custom_seen | 800 | 53.50% | 53.50% | 64.25% | 10.75% | 85.13% | 58.66 | 89.31 |
| custom_unseen | 800 | 7.50% | 7.50% | 19.25% | 11.75% | 58.60% | 58.41 | 93.28 |

All ArgA rates use positive samples. Oracle gap uses the same positive denominator.
Strict and normalized score the same postprocessed predictions; they do not measure inference ablation.
Review human_review_queue.jsonl before attributing W/T/P/I to linguistic causes.

# Baseline Results

- **Split**: test
- **Candidate K**: 50
- **Generated**: 2026-05-24 16:29
- **Ground-truth injection**: enabled (see `src/baselines/adapter.py`)

## Baseline Descriptions

| Baseline | Description |
|----------|-------------|
| Random | Uniformly random shuffle of candidates (lower bound) |
| Popularity | Rank by recent entry count (global popularity, 60-day window) |
| CategoryMatch | Rank by worker's historical category affinity |
| QualityWeighted | Blend popularity with award density, weighted by worker quality (requester-oriented) |

## Results


### Generic Ranking Metrics

| Metric | Random | Popularity | CategoryMatch | QualityWeighted |
|---|---|---|---|---|
| HR@1 | 0.0206 | 0.0495 | 0.0158 | 0.0461 |
| HR@5 | 0.1074 | 0.2125 | 0.0621 | 0.2102 |
| HR@10 | 0.2167 | 0.3579 | 0.1290 | 0.3531 |
| NDCG@1 | 0.0206 | 0.0495 | 0.0158 | 0.0461 |
| NDCG@5 | 0.0627 | 0.1307 | 0.0388 | 0.1276 |
| NDCG@10 | 0.0976 | 0.1774 | 0.0597 | 0.1734 |
| MRR | 0.0944 | 0.1524 | 0.0768 | 0.1485 |
| Precision@1 | 0.0206 | 0.0495 | 0.0158 | 0.0461 |
| Precision@5 | 0.0215 | 0.0425 | 0.0124 | 0.0420 |
| Precision@10 | 0.0217 | 0.0358 | 0.0129 | 0.0353 |
| Recall@1 | 0.0206 | 0.0495 | 0.0158 | 0.0461 |
| Recall@5 | 0.1074 | 0.2125 | 0.0621 | 0.2102 |
| Recall@10 | 0.2167 | 0.3579 | 0.1290 | 0.3531 |


### Worker-Objective Metrics

| Metric | Random | Popularity | CategoryMatch | QualityWeighted |
|---|---|---|---|---|
| avg_award_value@1 | 67.9288 | 156.9550 | 59.6834 | 178.0509 |
| avg_award_value@5 | 67.4875 | 131.2475 | 57.4067 | 137.3110 |
| avg_award_value@10 | 67.7383 | 113.8419 | 66.0140 | 120.2591 |
| finalist_rate@1 | 0.0017 | 0.0032 | 0.0027 | 0.0041 |
| finalist_rate@5 | 0.0019 | 0.0027 | 0.0024 | 0.0029 |
| finalist_rate@10 | 0.0019 | 0.0027 | 0.0021 | 0.0028 |
| winner_rate@1 | 0.0017 | 0.0028 | 0.0026 | 0.0037 |
| winner_rate@5 | 0.0018 | 0.0026 | 0.0023 | 0.0028 |
| winner_rate@10 | 0.0019 | 0.0026 | 0.0021 | 0.0027 |
| category_match_rate@1 | 0.8915 | 0.9657 | 0.9990 | 0.9503 |
| category_match_rate@5 | 0.8909 | 0.9537 | 0.9895 | 0.9379 |
| category_match_rate@10 | 0.8908 | 0.9430 | 0.9797 | 0.9219 |


### Requester-Objective Metrics

| Metric | Random | Popularity | CategoryMatch | QualityWeighted |
|---|---|---|---|---|
| avg_recommender_worker_quality | 0.8894 | 0.8888 | 0.8918 | 0.8854 |
| project_coverage | 1.0000 | 0.5130 | 0.3525 | 0.5679 |

## Analysis

### Key Findings

- **Generic Ranking**: Popularity leads on 13/13 metrics
- **Worker-Objective**: QualityWeighted leads on 9/12 metrics
- **Requester-Objective**: Random leads on 1/2 metrics

### Notes

- Random baseline establishes the lower bound. Any meaningful recommender should significantly outperform it.
- All baselines use the same JOB-06 candidate set (K=50) with ground-truth injection for fair comparison.
- Anti-leakage: all history-based baselines only use data strictly before each evaluation timestamp.

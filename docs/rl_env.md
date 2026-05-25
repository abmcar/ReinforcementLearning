# RL Environment

## Selected Formulation

The DQN mainline uses a contextual bandit replay formulation. Each logged
entry becomes one transition with state `s`, logged action `a`, reward `r`,
candidate set `candidates`, and `s_next = None`.

This is the default selected in `docs/roadmap.md`: the dataset is an offline
recommendation log, the reward is immediately observed from the logged
worker-project interaction, and there is no simulator for online transitions.

## State

`OfflineRecommendationEnv.state_dim == 9`.

State features are built from JOB-03 logged feature rows and only use
information available before the entry timestamp:

| Feature | Source |
|---|---|
| `worker_quality` | normalized worker quality |
| `log1p(hist_entries)` | prior worker entries |
| `log1p(hist_wins)` | prior worker wins |
| `hist_win_rate` | prior win rate |
| `log1p(hist_avg_award)` | prior average award |
| yearly sin/cos | timestamp context |
| weekday sin/cos | timestamp context |

## Action And Candidates

Actions are project IDs from the JOB-06 candidate set. Candidate project
features are constructed at timestamp `t` for every candidate:

| Feature | Source |
|---|---|
| category | project metadata |
| sub-category | project metadata |
| duration | project start/deadline |
| days remaining | deadline minus `t` |
| current entries | entries strictly before `t` |
| active flag | `start_date <= t <= deadline` |

The logged ground-truth project is injected when JOB-06 does not recall it.
This matches the baseline evaluator policy and keeps `Q(s, a_data)` defined for
TD and CQL losses.

## Reward

Two reward functions are implemented in `src/rl/rewards.py`.

Worker objective:

```text
award_value / award_scale
+ finalist_bonus * finalist
+ winner_bonus * winner
+ category_match_bonus * category_match
```

Requester objective:

```text
worker_quality
```

`RewardConfig` makes the worker reward coefficients injectable.

## Transition

`Transition` fields:

| Field | Meaning |
|---|---|
| `s` | state vector |
| `a` | logged project ID |
| `r` | selected objective reward |
| `s_next` | always `None` for contextual bandit |
| `candidates` | action set used by DQN |
| `candidate_features` | per-candidate action features |
| `action_index` | position of logged action in candidates |
| `info` | worker, project, timestamp, objective metadata |

## Verification

Executed during implementation:

```text
.venv/bin/python -m pytest tests/test_env.py tests/test_dqn.py -q
5 passed

.venv/bin/python - <<'PY'
from itertools import islice
from src.rl.env import OfflineRecommendationEnv
count = sum(1 for _ in islice(OfflineRecommendationEnv(split='train', objective='worker', candidate_k=50).iter_transitions(), 10000))
print('transition_count_sample', count)
PY
transition_count_sample 10000
```

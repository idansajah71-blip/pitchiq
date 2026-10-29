# Data summary

Source: StatsBomb Open Data via `statsbombpy`.

- Competitions: 3
- Matches: 479
- Shots: 12,241
- Goals: 1,294
- Base rate: 10.57%
- Penalty share: 1.36% (conversion 75.4%)
- Non-penalty base rate: 9.67%

## Competitions

```text
   competition    season  matches  shots  goals  conversion
FIFA World Cup      2022       64   1494    195    0.130522
       La Liga 2019/2020       35    839    111    0.132300
Premier League 2015/2016      380   9908    988    0.099717
```

## Column audit

| column | role | dtype | missing_pct |
|---|---|---|---|
| event_id | meta | str | 0.0% |
| match_id | meta (group split) | int64 | 0.0% |
| minute | safe (pre-shot) | int64 | 0.0% |
| second | safe (pre-shot) | int64 | 0.0% |
| period | safe (pre-shot) | int64 | 0.0% |
| player_id | meta | float64 | 0.0% |
| player | meta | str | 0.0% |
| team_id | meta | int64 | 0.0% |
| team | meta | str | 0.0% |
| possession_team | meta | str | 0.0% |
| x | safe (pre-shot location) | float64 | 0.0% |
| y | safe (pre-shot location) | float64 | 0.0% |
| xg_statsbomb | BENCHMARK only - never a feature | float64 | 0.0% |
| outcome | LEAKAGE / target source | str | 0.0% |
| is_goal | TARGET (derived from outcome) | bool | 0.0% |
| is_penalty | flag (penalties excluded from model) | bool | 0.0% |
| body_part | safe (pre-shot) | str | 0.0% |
| shot_type | safe (pre-shot) | str | 0.0% |
| technique | safe (pre-shot) | str | 0.0% |
| play_pattern | safe (pre-shot) | str | 0.0% |
| under_pressure | safe (pre-shot) | bool | 0.0% |
| first_time | safe (pre-shot) | bool | 0.0% |
| open_goal | safe (pre-shot) | bool | 0.0% |
| one_on_one | safe (pre-shot) | bool | 0.0% |
| aerial_won | safe (pre-shot) | bool | 0.0% |
| freeze_frame | safe (snapshot at shot time) | str | 1.2% |

`outcome` is only used to derive the target. `xg_statsbomb` is a benchmark and is never passed to a model. `shot_end_location` is not stored at all.

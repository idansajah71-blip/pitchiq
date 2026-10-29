# PitchIQ — Expected Goals Intelligence

Shot-level expected goals (xG) built from StatsBomb Open Data and turned into
**actionable insight for coaches and analysts** — not just an accuracy
leaderboard.

> Data provided by **StatsBomb** via their Open Data initiative
> (https://github.com/statsbomb/open-data). All analysis and errors are ours.
> PitchIQ is an independent hackathon project and is not affiliated with
> StatsBomb.

## Live demo

Deployed on **Streamlit Community Cloud** — link to be added before
submission. Until then, run it locally:

```bash
streamlit run app/streamlit_app.py
```

| ![Shot map](reports/figures/fig1_shot_map.png) |
|---|
| *Shot Map — every shot sized/coloured by PitchIQ xG; in the app this is a fully interactive Plotly visual with hover details and filters.* |

## The problem

A coach asks: *which shots should we be taking, and which finishing cases are
real skill or luck?* Raw shot counts understate quality — a 40-yard daisycutter
and a one-touch finish goal count the same in the box score, but carry wildly
different probabilities. PitchIQ answers with a transparent xG model (every
input is known **before** the shot is taken) plus player/team/zone views that
lead directly to concrete instructions.

## What it does

1. **Shot Map** — every shot on an interactive pitch; size/colour = xG;
   hover for player, minute, context and outcome.
2. **xG Estimator** — pick position + body part + context, get the goal
   probability in real time, with distance/angle and a goal-mouth visual.
3. **Model Evaluation** — honest metrics (ROC-AUC, log loss, Brier,
   calibration curves) explained in plain language, benchmarked against
   StatsBomb's own xG on the same held-out matches.
4. **Insight Engine** — player/creator/finisher leaderboards, team shot
   selection, zone efficiency, and the five recommendations below.

## Results (seed 42, penalties excluded, split grouped by match)

| Model | ROC-AUC | Log loss | Brier |
|---|---|---|---|
| Logistic Regression (chosen) | 0.772 | 0.263 | 0.0741 |
| XGBoost (tuned) | 0.781 | 0.259 | 0.0730 |
| Random Forest | 0.775 | 0.260 | 0.0731 |
| StatsBomb xG (benchmark) | 0.783 | 0.255 | 0.0711 |

* Ablation: distance+angle alone → AUC 0.748; + shot context → 0.761;
  **+ freeze-frame defender/GK features → 0.772** (+0.024 over geometry only).
* XGBoost's edge is inside CV noise (Δ AUC ≈ 0.009), so the transparent
  Logistic Regression is shipped and trees remain the performance fallback.

### Headline insights (coach language)

* Counter-attack shots average **0.170 xG** (14.7% conversion) vs 0.104 in
  regular play — tempo is the cheapest upgrade.
* Aston Villa take **55%** of their shots at < 0.05 xG; Stoke City shoot from
  30+ units out for 14% of attempts (xG/shot 0.016, 0 goals). Killing those
  attempts is worth ~**+5.7 xG** per season.
* Zone `110+/y20-40` delivers 0.184 mean xG but only sees 14.1% of shots.
* Messi: 30 goals from 22.6 xG (**+7.4**, 217 shots); Özil: 142 assisted shots
  (14.7 xG created).

## Data

| Competition | Season | Matches | Shots | Goals | Conversion |
|---|---|---|---|---|---|
| Premier League | 2015/2016 | 380 | 9,908 | 988 | 9.97% |
| La Liga (Messi framed) | 2019/2020 | 35 | 839 | 111 | 13.23% |
| FIFA World Cup | 2022 | 64 | 1,494 | 195 | 13.05% |
| **Total** | | **479** | **12,241** | **1,294** | **10.57%** |

Penalties (167) are reported but excluded from the model — a 12-yard
one-on-one is a different task. Freeze frames exist for 98.8% of shots.

### Leakage controls

* Split **grouped by `match_id`** (GroupShuffleSplit 80/20 + GroupKFold-5 CV).
* `shot.outcome` only builds the target; `shot_statsbomb_xg` is a benchmark,
  never an input; `shot_end_location` never enters the table.
* Every feature is justified in `reports/feature_justification.md`;
  11 feature/geometry tests + 2 headless app smoke tests guard the pipeline.

## How to run

```bash
pip install -r requirements.txt

# full reproducible pipeline (downloads/caches StatsBomb events first run)
# PowerShell:  ./scripts/run_pipeline.ps1
# or step-by-step:
python -m src.data && python -m src.features && python -m src.eda
python -m src.train
python -m src.creators && python -m src.insights

# the app
streamlit run app/streamlit_app.py
```

Requires Python 3.11+. Tests: `python -m pytest tests/ -q` (13 tests).

## Project structure

```
pitchiq/
├── app/streamlit_app.py        # 4-page Streamlit app (dark theme)
├── src/
│   ├── data.py                 # StatsBomb download + caching + audit
│   ├── features.py             # geometry, freeze-frame, preprocessor
│   ├── creators.py             # shot -> key pass -> assist creator
│   ├── train.py                # splits, ablation, tuning, calibration
│   ├── insights.py             # SHAP, permutations, leaderboards
│   └── eda.py                  # conversion rates + figures
├── models/                     # best_model.joblib + metadata.json
├── reports/                    # CSVs, markdown, figures (fig1..fig11)
├── scripts/run_pipeline.ps1    # end-to-end runner
└── tests/                      # geometry + app smoke tests
```

## Limitations & next steps

* 3 competitions, 479 matches — club data is 2015/16 & 2019/20; xG style
  margins may drift with rule changes (2025: shot-clock tendencies).
* Freeze-frame features are pre-shot snapshots, not tracked pressure quality;
  estimator ignores defender/GK placement (median-imputed, flagged).
* Player tables treat the shooter as credential — "creators" use assisted
  shots, not weighted value-added (would need event-level pass chains).
* goals − xG on < 30 shots is noise; the app flags small samples.
* Next: step-based EPV instead of a static angle model, calibration
  monitoring per competition, and a pass-chain xG added for creators.

## Credits & license

* Data: **StatsBomb Open Data** (CC BY-NC) via `statsbombpy`.
* Code: MIT License (see `LICENSE`).

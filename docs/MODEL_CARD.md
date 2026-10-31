# ShotWorth Model Card

*Version: post-audit regeneration, October 29 2026 (`audit_protocol_version: 2026-10-29-v1`)*

## 1. Identity

- **Task:** binary shot-level goal probability (xG) for non-penalty open-play and
  free-kick shots.
- **Deployment model:** Logistic Regression (pre-declared before holdout scoring,
  chosen for transparency).
- **Exploratory comparators:** tuned RandomForest / XGBoost over identical splits
  (CV-scaled hyperparameters; nested CV not yet implemented, so their numbers are
  context, not claims).
- **Benchmark only:** StatsBomb's published xG is scored on the same holdout for
  context. It is never an input feature and never a training target.

## 2. Data

- **Source:** StatsBomb Open Data (free tier), via `statsbombpy` — Premier League
  2015/16, La Liga 2019/20, 2022 World Cup.
- **Scope:** 479 matches, 12,241 non-penalty shots (2,459-shot holdout,
  96 matches); penalties reported separately.
- **Unit:** one row = one shot, with pre-shot context (position, body part, shot
  type, technique, play pattern, pressure/first-time/open-goal flags) and —
  when a freeze-frame exists — per-opponent defender/goalkeeper positions.
  Missing freeze-frames are flagged (`has_freeze_frame = 0`) rather than silently
  imputed into progress narratives.

## 3. Features (pre-shot only)

Geometry (distance to goal, open-goal angle, log-distance, distance-to-line,
lateral distance from centre), shot context (body part, type, technique, play
pattern, one-on-one, open goal, aerial, pressure, first-time, kickoff/last-minute
flags), freeze-frame counts (defenders in the shot corridor, keeper factor,
gk-in-line) and simple defensive spatial aggregates. No post-shot information
(outcome, end location, keeper reaction) ever enters the feature matrix.

## 4. Protocol

- **Split:** `GroupShuffleSplit` grouped by `match_id` (80/20, seed `20261029`,
  9,615 train shots / 383 train matches vs 2,459 test shots / 96 test matches,
  zero match overlap).
- **Leakage controls:** no outcome or post-shot features; StatsBomb xG excluded
  from inputs; grouped CV inside model selection; reports regenerated
  end-to-end after an external audit (`PITCHIQ_Audit_Report.md`).
- **Calibration:** isotonic calibration fitted with **grouped out-of-fold
  predictions on the training set only**; the deployment path always applies
  the saved calibrator.

## 5. Metrics (holdout, seed 20261029)

| Model | ROC-AUC | Log loss | Brier |
| --- | --- | --- | --- |
| Logistic Regression (deployed) | **0.796** | 0.257 | 0.0732 |
| RandomForest (exploratory) | 0.780 | 0.264 | 0.0747 |
| XGBoost (exploratory) | 0.784 | 0.263 | 0.0747 |
| StatsBomb xG (benchmark) | 0.803 | 0.255 | 0.0724 |

Ablation (LogReg): geometry-only 0.756 → + shot context 0.757/0.781 steps →
+ freeze-frame defenders 0.796. ECE of the deployed path: 0.0204.

## 6. Intended use

- Support analysts and coaches with pre-shot chance-quality estimates,
  opportunity-cost framing ("what is a low-quality shot?"), zone valuation and
  scouting-style questions.
- NOT intended for: betting, player contracts, disciplinary or fitness
  decisions, or any claim about individual causality. Insights in the app are
  labelled as descriptive associations or assumption-based scenarios.

## 7. Limitations

- Event-data freeze-frame coverage varies by competition/season; defender
  aggregates are coarse.
- Lookup-free counterfactuals in the Estimator ("Why this xG?") are
  one-at-a-time approximations, not SHAP-exact decompositions.
- Beach-ball/noisy zones and very small player samples are filtered with
  min-sample controls, but remain noisy even above thresholds.
- No retraining on new seasons yet; model assumes feature distributions similar
  to the training window.

## 8. Reproduction

`./scripts/run_pipeline.ps1` (Windows) or the `python -m src.data …` sequence in
`README.md` — data → features → eda → train → creators → insights — then
`pytest tests/`. Every quoted number in the UI/README/Devpost comes from the
post-audit artifacts only.

Data remains © StatsBomb under their open-data terms; ShotWorth is an
independent hackathon project, not affiliated with StatsBomb.

# Devpost description (final)

**Title:** PitchIQ — Expected Goals Intelligence

**Live demo:** <https://pitchiq-xg.streamlit.app> · **Source:** <https://github.com/idansajah71-blip/pitchiq>

## Problem
Every shot counts as one in a basic box score, but chances differ dramatically. Coaches and analysts need to understand where shots come from, what context affects chance quality, and which patterns are worth investigating—not just receive another chart of raw counts.

## Solution
PitchIQ turns about 12,000 StatsBomb Open Data shots from the Premier League 2015/16, La Liga 2019/20 and the 2022 World Cup into a transparent, coach-facing expected-goals (xG) workflow. It uses pre-shot context such as distance, angle, shot type, and pressure, plus defender/goalkeeper positioning when a freeze-frame is available (missing frames are flagged). Match-grouped splits keep shots from one match together. Penalties are excluded from model training and handled separately in the shot map.

## Features
- **Shot Map:** explore shots by team, player, context and outcome.
- **Match Timeline:** replay any match — cumulative xG race per minute, goal markers, benchmark overlay and an auto-playing live replay mode.
- **xG Estimator:** change shot location and context to see how estimated scoring probability responds, with a one-at-a-time "Why this xG?" counterfactual explainer.
- **Model Evaluation:** compare a transparent Logistic Regression baseline with tree-model experiments and StatsBomb's xG benchmark; inspect discrimination, probability error and calibration.
- **Beat the Model:** price five unseen holdout shots yourself; log loss grades you against the model on identical data.
- **Insights:** team head-to-head radar, an auto-generated scout card, shot selection by zone, player and creator summaries, and assumption-labelled coaching scenarios.

## Practical value
PitchIQ is designed to help analysts ask better follow-up questions: are low-quality attempts concentrated in particular zones, do patterns persist across samples, and which clips should a coach review? Counter-attack differences are presented as descriptive associations rather than causal claims. Scenario-based xG changes are assumptions, not guaranteed extra goals. Player leaderboard rows are separated by team and competition, avoiding club/national-team totals being shown under one team.

## Evaluation (regenerated post-audit, Oct 2026)

Logistic Regression is the **pre-declared** deployment baseline for
transparency; tuned tree models are exploratory comparators (nested CV is not
yet implemented), and no claim is made that PitchIQ beats StatsBomb's
specialist xG. On a match-grouped holdout (96 matches / 2,459 shots, seed
20261029): ROC-AUC **0.796**, log loss **0.257**, Brier **0.0732** (Logistic
Regression), StatsBomb benchmark 0.803 / 0.255 / 0.0724. Ablation:
geometry-only 0.756 → +shot context 0.781 → +freeze-frame defenders 0.796.
Insights come wrapped in explicit framing, e.g. Aston Villa's 54% sub-0.05-xG
shots feed a typed replacement scenario (≈ +5.2 xG, an assumption not a
forecast) and counter-attack differences are labelled descriptive
associations.

Data: StatsBomb Open Data. Code: MIT. PitchIQ is an independent project and is not affiliated with StatsBomb.

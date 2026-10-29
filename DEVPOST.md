# Devpost description (max ~300 words)

**Title:** PitchIQ — Expected Goals Intelligence

## Problem
Shot counts lie. A 40-yard hope and a one-touch tap-in both count as "a shot",
yet turn into goals ~1% vs ~40% of the time. Coaches and analysts need to know
which chances are worth taking, which finishing streaks are real skill, and
where their team leaks — but public xG tools are black boxes or dashboards
that end in numbers, not instructions.

## Solution
PitchIQ turns ~12,000 StatsBomb Open Data shots (Premier League 2015/16,
La Liga 2019/20, World Cup 2022; 479 matches) into a transparent xG model and
a coach-facing app. Every input exists **before** the shot: distance, goal
angle, freeze-frame defenders/keeper geometry, pressure and finish context.
Split grouped by match, penalties handled separately — no leakage.

## Features
* **Shot Map** — every shot sized/coloured by xG, filterable by team, player,
  body part, type, outcome.
* **xG Estimator** — pick a position + context, see the probability, distance
  and goal-mouth angle update live.
* **Evaluation** — ROC-AUC 0.772 / Brier 0.0741 (Logistic Regression, seed 42;
  XGBoost within noise at 0.781), calibration curves and plain-language
  explanations, with StatsBomb's xG as an external benchmark on identical data.
* **Insight engine** — SHAP + permutation importance, player/creator/finisher
  leaderboards, team shot selection, zone efficiency and number-backed
  recommendations.

## Actionable impact
Counter shots return 0.170 xG vs 0.104 in regular play; Aston Villa take 55%
of shots below 0.05 xG and Stoke's long-range habit (0% on 59 shots) costs a
team ~5.7 xG per season if kept; the highest-value zone (0.184 mean xG) gets
only 14.1% of shots. These reorder patterns and drills directly.

## Limitations & next steps
Three competitions; players appear once; goals − xG is noisy under 30 shots
(flagged in-app). Next: pass-chain value for creators and per-competition
calibration.

Data: StatsBomb Open Data. Code: MIT.

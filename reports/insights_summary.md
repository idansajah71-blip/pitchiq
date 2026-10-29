# Insights summary (Phase 3)

Model output: PitchIQ xG = saved pipeline + optional calibration.
Penalties excluded everywhere (reported separately). Player and team tables use **descriptive averages**, not guarantees - goals - xG is noisy below ~30 shots.

## Top 5 factors (SHAP, grouped to original features)

1. **distance** (mean |SHAP| 1.606) - Closer to goal = more of the goal is open and the keeper has less time. This is the single biggest driver.
2. **log_distance** (mean |SHAP| 0.571) - A transformed distance that matches how fast quality decays.
3. **angle_deg** (mean |SHAP| 0.371) - A wide angle means the shooter sees more goal between the posts; tight angles are almost never worth taking.
4. **n_defenders_cone** (mean |SHAP| 0.248) - Defenders standing between the shot and the goal block it; one extra defender measurably lowers quality.
5. **nearest_defender_dist** (mean |SHAP| 0.216) - Time and space: the closer the nearest defender, the worse the shot.

Permutation importance agrees on order: distance, log_distance, angle_deg, distance_to_goal_line, body_part (see `permutation_importance.csv`).

### Top chance creators (assists received as shooter, min 10)

| creator | team | assisted shots | assisted goals | assisted xG |
|---|---|---|---|---|
| Mesut Özil | Arsenal | 142 | 19 | 14.68 |
| Lionel Andrés Messi Cuccittini | Barcelona | 96 | 13 | 11.90 |
| Dimitri Payet | West Ham United | 112 | 12 | 11.44 |
| Riyad Mahrez | Leicester City | 66 | 10 | 10.24 |
| Christian Dannemann Eriksen | Tottenham Hotspur | 116 | 12 | 10.16 |
| Kevin De Bruyne | Manchester City | 86 | 9 | 9.87 |
| Alexis Alejandro Sánchez Sánchez | Arsenal | 57 | 4 | 8.80 |
| Francesc Fàbregas i Soler | Chelsea | 69 | 7 | 8.68 |
| Troy Deeney | Watford | 56 | 7 | 8.64 |
| Dušan Tadić | Southampton | 86 | 13 | 8.57 |

### Top finishers (min 15 shots, ranked by goals - PitchIQ xG)

| player | team | shots | xG | xG/shot | goals | goals-xG |
|---|---|---|---|---|---|---|
| Lionel Andrés Messi Cuccittini | Barcelona | 217 | 22.60 | 0.104 | 30 | +7.40 |
| Sergio Leonel Agüero del Castillo | Manchester City | 113 | 14.85 | 0.131 | 20 | +5.15 |
| Riyad Mahrez | Leicester City | 81 | 7.97 | 0.098 | 13 | +5.03 |
| André Ayew Pelé | Swansea City | 70 | 8.16 | 0.117 | 13 | +4.84 |
| Anthony Martial | Manchester United | 57 | 6.67 | 0.117 | 11 | +4.33 |
| Marcus Rashford | Manchester United | 27 | 3.86 | 0.143 | 8 | +4.14 |
| Kelechi Promise Iheanacho | Manchester City | 29 | 4.38 | 0.151 | 8 | +3.62 |
| Dimitri Payet | West Ham United | 69 | 5.77 | 0.084 | 9 | +3.23 |
| Nathan Redmond | Norwich City | 53 | 2.86 | 0.054 | 6 | +3.14 |
| Kylian Mbappé Lottin | France | 29 | 2.92 | 0.101 | 6 | +3.08 |

### Best shot quality (min 15 shots, ranked by PitchIQ xG/shot)

| player | team | shots | xG | xG/shot | goals | goals-xG |
|---|---|---|---|---|---|---|
| Frenkie de Jong | Barcelona | 17 | 5.86 | 0.345 | 3 | -2.86 |
| Martin Braithwaite Christensen | Barcelona | 23 | 5.18 | 0.225 | 2 | -3.18 |
| Alberto Paloschi | Swansea City | 15 | 3.23 | 0.216 | 2 | -1.23 |
| Diafra Sakho | West Ham United | 46 | 8.27 | 0.180 | 5 | -3.27 |
| Diego da Silva Costa | Chelsea | 70 | 12.49 | 0.178 | 12 | -0.49 |
| Dwight Gayle | Crystal Palace | 22 | 3.91 | 0.178 | 3 | -0.91 |
| Jamie Vardy | Leicester City | 112 | 19.69 | 0.176 | 19 | -0.69 |
| Daniel Nii Tackie Mensah Welbeck | Arsenal | 20 | 3.48 | 0.174 | 4 | +0.52 |
| Fernando Francisco Reges | Manchester City | 16 | 2.78 | 0.174 | 2 | -0.78 |
| Antoine Griezmann | Barcelona | 68 | 11.81 | 0.174 | 11 | -0.81 |

Goals - xG is noisy on small samples; treat single-season finishes (especially under ~30 shots) as luck-flavoured evidence.

## Teams with the most low-quality shooting

- Aston Villa: 54.9% of shots below xG 0.05 (390 shots)
- Crystal Palace: 53.8% of shots below xG 0.05 (472 shots)
- Liverpool: 51.0% of shots below xG 0.05 (633 shots)
- Stoke City: 51.0% of shots below xG 0.05 (420 shots)
- Tottenham Hotspur: 50.2% of shots below xG 0.05 (651 shots)

## Recommendations for coaches (short version)

- Aston Villa take 55% of their shots below 0.05 xG (their PitchIQ xG/shot 0.073 vs league 0.098). If just one in three of those hopeless attempts were replaced with a medium-quality (~0.10 xG) look, that alone adds about 5.7 xG per season. Instruction: fewer low-probability shots, more ball progression into the box.
- Stoke City shoot from 30+ units out for 14% of their attempts (59 shots: xG/shot 0.016, conversion 0.0%). Their xG difference for vs against is -13.11. Instruction: keep long shots as a rebound option only, not a first choice.
- 'From Counter' shots average 0.170 xG (conversion 14.7%) vs 0.104 for regular play. Increase tempo and shots out of this pattern: 558 shots now, each extra 20 shots in this pattern is approximately +3.40 xG.
- Inside the attacking bands, the band 110+ / 20-40 is the highest-value zone (mean xG 0.184, conversion 18.3%) but only takes 14.1% of all shots. Drill a higher share of entries into that corridor before shooting.
- Lionel Andrés Messi Cuccittini has 30 goals on 22.60 xG (217 shots, +7.40 vs expectation, 0.104 xG/shot). Design volume for this finisher; if shots can be steered to him, expected output rises.

Files: `shap_top_factors.csv`, `permutation_importance.csv`, `player_summary.csv`, `team_summary.csv`, `zone_summary.csv`, `context_quality.csv`, `coach_recommendations.md`.

Data: StatsBomb Open Data. PitchIQ is an independent hackathon project and is not affiliated with StatsBomb.
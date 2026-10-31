# Insights summary (Phase 3)

Model output: ShotWorth xG = saved pipeline + optional calibration.
Penalties excluded everywhere (reported separately). Player and team tables use **descriptive averages**, not guarantees - goals - xG is noisy below ~30 shots.

## Top 5 factors (SHAP, grouped to original features)

1. **distance** (mean |SHAP| 1.317) - Closer to goal = more of the goal is open and the keeper has less time. This is the single biggest driver.
2. **log_distance** (mean |SHAP| 0.555) - A transformed distance that matches how fast quality decays.
3. **angle_deg** (mean |SHAP| 0.375) - A wide angle means the shooter sees more goal between the posts; tight angles are almost never worth taking.
4. **n_defenders_cone** (mean |SHAP| 0.216) - Counts defenders between shooter and goal; interpret alongside the shot location and freeze-frame availability.
5. **nearest_defender_dist** (mean |SHAP| 0.193) - Distance to the nearest defender is a proxy for space, not a complete measure of pressure.

Permutation importance agrees on order: distance, log_distance, angle_deg, body_part, gk_off_line (see `permutation_importance.csv`).

### Top chance creators (assists received as shooter, min 10)

| creator | team | competition | assisted shots | assisted goals | assisted xG |
|---|---|---|---|---|---|
| Mesut Özil | Arsenal | Premier League | 142 | 19 | 14.50 |
| Dimitri Payet | West Ham United | Premier League | 112 | 12 | 11.37 |
| Riyad Mahrez | Leicester City | Premier League | 66 | 10 | 10.04 |
| Lionel Andrés Messi Cuccittini | Barcelona | La Liga | 77 | 10 | 10.00 |
| Christian Dannemann Eriksen | Tottenham Hotspur | Premier League | 107 | 12 | 9.70 |
| Kevin De Bruyne | Manchester City | Premier League | 78 | 9 | 8.77 |
| Francesc Fàbregas i Soler | Chelsea | Premier League | 69 | 7 | 8.70 |
| Troy Deeney | Watford | Premier League | 56 | 7 | 8.45 |
| Alexis Alejandro Sánchez Sánchez | Arsenal | Premier League | 57 | 4 | 8.28 |
| Erik Lamela | Tottenham Hotspur | Premier League | 72 | 8 | 8.20 |

### Top finishers (min 15 shots, ranked by goals - ShotWorth xG)

| player | team | competition | shots | xG | xG/shot | goals | goals-xG |
|---|---|---|---|---|---|---|---|
| Lionel Andrés Messi Cuccittini | Barcelona | La Liga | 190 | 19.94 | 0.105 | 27 | +7.06 |
| Riyad Mahrez | Leicester City | Premier League | 81 | 7.61 | 0.094 | 13 | +5.39 |
| Sergio Leonel Agüero del Castillo | Manchester City | Premier League | 113 | 14.61 | 0.129 | 20 | +5.39 |
| André Ayew Pelé | Swansea City | Premier League | 69 | 7.61 | 0.110 | 12 | +4.39 |
| Anthony Martial | Manchester United | Premier League | 57 | 6.89 | 0.121 | 11 | +4.11 |
| Kelechi Promise Iheanacho | Manchester City | Premier League | 29 | 4.34 | 0.150 | 8 | +3.66 |
| Dimitri Payet | West Ham United | Premier League | 69 | 5.61 | 0.081 | 9 | +3.39 |
| Kylian Mbappé Lottin | France | FIFA World Cup | 29 | 2.74 | 0.094 | 6 | +3.26 |
| Nathan Redmond | Norwich City | Premier League | 53 | 2.96 | 0.056 | 6 | +3.04 |
| Graziano Pellè | Southampton | Premier League | 82 | 7.97 | 0.097 | 11 | +3.03 |

### Best shot quality (min 15 shots, ranked by ShotWorth xG/shot)

| player | team | competition | shots | xG | xG/shot | goals | goals-xG |
|---|---|---|---|---|---|---|---|
| Frenkie de Jong | Barcelona | La Liga | 15 | 5.11 | 0.341 | 2 | -3.11 |
| Martin Braithwaite Christensen | Barcelona | La Liga | 22 | 4.98 | 0.226 | 2 | -2.98 |
| Alberto Paloschi | Swansea City | Premier League | 15 | 3.13 | 0.209 | 2 | -1.13 |
| Fernando Francisco Reges | Manchester City | Premier League | 16 | 2.88 | 0.180 | 2 | -0.88 |
| Diafra Sakho | West Ham United | Premier League | 46 | 8.26 | 0.180 | 5 | -3.26 |
| Antoine Griezmann | Barcelona | La Liga | 62 | 10.72 | 0.173 | 11 | +0.28 |
| Diego da Silva Costa | Chelsea | Premier League | 70 | 12.08 | 0.173 | 12 | -0.08 |
| Jamie Vardy | Leicester City | Premier League | 112 | 19.25 | 0.172 | 19 | -0.25 |
| Daniel Nii Tackie Mensah Welbeck | Arsenal | Premier League | 20 | 3.40 | 0.170 | 4 | +0.60 |
| Dieumerci Mbokani Bezua | Norwich City | Premier League | 37 | 6.27 | 0.169 | 7 | +0.73 |

Goals - xG is noisy on small samples; treat single-season finishes (especially under ~30 shots) as luck-flavoured evidence.

## Teams with the most low-quality shooting

- Aston Villa: 54.4% of shots below xG 0.05 (390 shots)
- Liverpool: 50.9% of shots below xG 0.05 (633 shots)
- Crystal Palace: 50.8% of shots below xG 0.05 (472 shots)
- Tottenham Hotspur: 49.9% of shots below xG 0.05 (651 shots)
- Stoke City: 48.8% of shots below xG 0.05 (420 shots)

## Recommendations for coaches (short version)

- Aston Villa take 54% of their shots below 0.05 xG (ShotWorth xG/shot 0.073 vs dataset mean 0.098). Scenario, not a causal forecast: replacing one-third of these 212 shots (mean current xG 0.027) with chances rated 0.10 xG would add about +5.2 expected goals if other factors stayed constant. Coaching hypothesis: look for better progression into the box rather than treating low xG alone as proof a shot was a mistake.
- Stoke City attempt shots from 30+ StatsBomb pitch-coordinate units on 14% of their attempts (59 shots: xG/shot 0.017, conversion 0.0%). Their xG difference for vs against is -13.97. Coaching hypothesis: review whether progression or a pass creates a better option in these situations; low xG alone does not prove a shot was a mistake.
- 'From Counter' shots average 0.146 xG (conversion 14.7%) vs 0.097 for regular play, across 558 observed shots. This association is descriptive—not proof that the pattern or tempo causes better chances; shot location and game state may explain part of the gap. Review the transition clips before prescribing a tactical change.
- The attacking band 110+ / inside channel (mirrored y-bands grouped together) has the highest mean xG among qualified paired zones: 0.178 (conversion 17.7%; 3317 shots, 27.5% of all shots). Treat this as a location pattern to review on video, not proof that forcing shots into that band will improve outcomes.
- Lionel Andrés Messi Cuccittini (Barcelona, La Liga) recorded 27 goals on 19.94 ShotWorth xG (190 shots, +7.06 goals minus xG, 0.105 xG/shot). This is an observed residual, not proof it will persist; validate on a larger or later sample before treating it as repeatable finishing skill.

Files: `shap_top_factors.csv`, `permutation_importance.csv`, `player_summary.csv`, `team_summary.csv`, `zone_summary.csv`, `context_quality.csv`, `coach_recommendations.md`.

Data: StatsBomb Open Data. ShotWorth is an independent hackathon project and is not affiliated with StatsBomb.
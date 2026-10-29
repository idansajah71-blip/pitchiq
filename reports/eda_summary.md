# EDA summary

Source: StatsBomb Open Data via `statsbombpy`. Non-penalty shots unless stated.

- Matches: 479
- Shots: 12,241
- Base rate: 10.57%
- Non-penalty base rate: 9.67%
- Penalties: 167 (1.36%), conversion 75.4%

## Conversion by distance

| distance | shots | goals | conversion |
|---|---|---|---|
| 0-6 | 384 | 150 | 39.06% |
| 6-12 | 2532 | 432 | 17.06% |
| 12-18 | 2782 | 341 | 12.26% |
| 18-24 | 2821 | 147 | 5.21% |
| 24-30 | 2472 | 75 | 3.03% |
| 30+ | 1083 | 23 | 2.12% |

## Conversion by angle

| angle_deg | shots | goals | conversion |
|---|---|---|---|
| 0-5 | 26 | 1 | 3.85% |
| 5-10 | 233 | 13 | 5.58% |
| 10-15 | 2466 | 55 | 2.23% |
| 15-20 | 3506 | 174 | 4.96% |
| 20-25 | 1733 | 172 | 9.92% |
| 25-35 | 1761 | 238 | 13.52% |
| 35+ | 2349 | 515 | 21.92% |

## Conversion by body part

| body_part | shots | goals | conversion |
|---|---|---|---|
| Right Foot | 6351 | 578 | 9.10% |
| Left Foot | 3723 | 357 | 9.59% |
| Head | 1965 | 219 | 11.15% |

## Conversion by shot type

| shot_type | shots | goals | conversion |
|---|---|---|---|
| Free Kick | 501 | 31 | 6.19% |
| Open Play | 11570 | 1136 | 9.82% |

## StatsBomb xG benchmark

- ROC-AUC: 0.800
- Brier: 0.0723
- Log loss: 0.2575

## Freeze-frame effect

| defender_in_cone | shots | goals | conversion |
|---|---|---|---|
| 1+ defender in cone | 6362 | 389 | 6.11% |
| No defender in cone | 5712 | 779 | 13.64% |

## Figures

- `fig1_shot_map.png`
- `fig2_distance_angle.png`
- `fig3_body_part_type.png`
- `fig4_xg_calibration.png`

# Feature justification

Every feature is recorded before or at the moment of the shot.

| Feature | Source column | Football reason | Leakage risk |
|---|---|---|---|
| `distance` | location [x, y] | Closer shots beat the keeper more often. | None - known before the shot. |
| `angle_deg` | location [x, y] | Wider angle to the posts means more goal to aim at. | None - known before the shot. |
| `log_distance` | derived from distance | Linearises the strong distance decay for the LR baseline. | None - deterministic transform. |
| `distance_to_goal_line` | 120 - location x | How deep the shooter is; separates wide-close from central-far. | None - known before the shot. |
| `abs_y_from_center` | location y vs goal centre 40 | Central shots have a better angle than wide ones. | None - known before the shot. |
| `minute` | event minute | Late-game fatigue and game-state effects. | None - timestamp of the shot. |
| `n_defenders_cone` | shot_freeze_frame | Defenders between shooter and goal block the shot. | None - snapshot at shot time, not after. |
| `n_defenders_near` | shot_freeze_frame | Pressure from nearby defenders lowers shot quality. | None - snapshot at shot time. |
| `nearest_defender_dist` | shot_freeze_frame | Time and space on the ball. | None - snapshot at shot time. |
| `gk_off_line` | shot_freeze_frame (Goalkeeper) | A keeper caught off the line concedes easier finishes. | None - position at shot time. |
| `gk_y_offset` | shot_freeze_frame (Goalkeeper) | A keeper shaded to one side leaves the other side open. | None - position at shot time. |
| `has_freeze_frame` | shot_freeze_frame presence | Makes missing defensive context explicit rather than silent. | None - data availability indicator. |
| `under_pressure` | under_pressure flag | Pressure reduces technique and placement. | None - recorded during the event. |
| `first_time` | shot_first_time flag | First-time finishes are harder to control. | None - known before contact. |
| `open_goal` | shot_open_goal flag | Empty-net situations are near-certain goals. | None - known before the shot. |
| `one_on_one` | shot_one_on_one flag | A clean look against the keeper is a different shot type. | None - known before the shot. |
| `aerial_won` | shot_aerial_won flag | Aerial duels won before the shot signal a cleaner contact. | None - occurs before the shot. |
| `body_part` | shot_body_part | Contact type changes control; headers cluster at close range, so read it with location. | None - known at contact. |
| `shot_type` | shot_type | Open play and free kicks have different geometries; penalties are removed. | None; penalty rows are removed before training. |
| `technique` | shot_technique | Volleys, lobs and backheels are distinct contact types selected by situation. | None - known at contact. |
| `play_pattern` | play_pattern | Counters and set pieces create different shot contexts. | None - known before the shot. |

## Excluded from the model

| Column | Why it is excluded |
|---|---|
| `outcome` / `is_goal` | Target only. |
| `xg_statsbomb` | StatsBomb benchmark; using it as a feature would copy the answer. |
| `shot_end_location` | Only exists after the shot; never stored. |
| `player`, `team` | Identity shortcuts that do not generalise to new matches. |
| penalty shots | Fixed 12-yard one-on-one; excluded from the main model and reported separately. |

## Missing values

Freeze-frame features are missing when StatsBomb has no freeze frame for a shot (1.2% of shots). They are median-imputed inside the sklearn pipeline and flagged with `has_freeze_frame`, so the model can learn how much to trust them. Goalkeeper features are missing in 1.3% of shots. The same preprocessor is used at inference time.

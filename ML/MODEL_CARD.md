# RoadWeave baseline model card

## Status

This is a research prototype and demonstration baseline, not a controller for
a physical vehicle. The deterministic collision supervisor must remain active.

Training completed on 2026-08-25 with Python 3.9.6 and scikit-learn 1.6.1. All
splits are group-disjoint: rows from the same nuScenes scene or K-Risk event
cannot appear on both sides of a train/validation/test boundary.

## Risk model

- Model: histogram gradient-boosting classifier
- Rows: 41,668
- Independent groups: 26,901
- Classes: LOW 7,746; MODERATE 21,774; HIGH 8,627; EXTREME 3,521
- Test accuracy: 0.9181
- Test macro-F1: 0.9083
- Leave-one-source-out macro-F1: 0.0758 (train nuScenes → test K-Risk),
  0.0746 (train K-Risk → test nuScenes)

nuScenes risk is now graded into all four levels from physical severity (ego
acceleration, front TTC, pedestrian gap, and overlap) instead of only LOW, and
the K-Risk conversion now includes inD and rounD alongside highD and CitySim.
This removes the "LOW always means nuScenes" shortcut for the within-domain
task: random group-separated macro-F1 rose from 0.82 to 0.91.

The leave-one-source-out numbers are the honest measure of generalization:
they stay near 0.08 because the two datasets are genuinely different domains
(urban sensor-rich driving vs. German/Chinese trajectory segments). The model
still cannot transfer across that gap, so treat every within-domain score as
optimistic evidence.

## Policy model

- Model: histogram gradient-boosting classifier
- Rows: 13,426
- Classes: KEEP 11,889; ACCELERATE 658; DECELERATE 632; CHANGE_LEFT 136;
  CHANGE_RIGHT 111
- Test accuracy: 0.9206
- Test macro-F1: 0.7612
- Per-class test F1: KEEP 0.957; ACCELERATE 0.566; DECELERATE 0.531;
  CHANGE_LEFT 0.906; CHANGE_RIGHT 0.846
- Leave-one-source-out macro-F1: 0.1426 (train nuScenes → test K-Risk),
  0.2023 (train K-Risk → test nuScenes)

Lane changes are now learnable. highD contributes direction only when its
native `yaw_left`/`yaw_right` signal is paired with a true clip-level
`lane_diff`; nuScenes lane changes are detected by integrating lateral motion
per frame heading (curvature-invariant) over a 3 s window. This avoids treating
21 notable lateral movements as completed lane changes while preserving strong
held-out lane-change scores.

The policy score is still within-domain evidence. Its leave-one-source-out
results remain weak, so it must not be described as a source-independent or
real-vehicle-ready controller.

## Controller benchmark

The original six-seed, 150-second benchmark found the ML controller more
conservative than the deterministic controller. After the lane-label cleanup,
a two-seed, 90-second regression run confirmed the same direction: rule/ML
progress was 716/641 m, minimum TTC 1.352/1.394 s, emergency stops 2/2,
deadlock time 1.85/3.55 s, and mean absolute acceleration 1.027/1.089 m/s².
This shorter post-clean run is a regression check, not a replacement for a
larger statistical evaluation.

## Weather-speed model

- Model: histogram gradient-boosting regressor
- Dataset: Extreme Driving Conditions Dataset
- Conditions: `DRY`, `RAIN`, `SNOW`, `FOG`
- Converted rows: 18,787 from 413 unique episodes
- Training rows/groups: 16,869 / 379
- Official validation rows/groups: 1,918 / 34
- Inputs: 16 summaries of the previous three seconds of longitudinal
  acceleration and yaw rate, plus one-hot weather context
- Output: a continuous cruise-speed factor clipped to 0.35–1.0
- Five-fold, episode-separated CV MAE: 0.1263 (fixed rules: 0.2106)
- Official validation MAE: 0.1401 (fixed rules: 0.1843)
- Official validation R²: 0.2033

The target is the following second's mean vehicle speed divided by a 50 km/h
reference. Current-speed features are intentionally excluded: including them
made future-speed prediction a nearly identical copy of the present speed and
produced a misleadingly small error. Condition-balanced sample weights prevent
the numerous fog frames from completely overwhelming rain and snow.

Fog is included because the episode annotations contain explicit fog/haze
descriptions. The converter found 14,236 fog rows, 1,528 rain rows, 262 snow
rows, and 2,761 dry rows across train and validation. The official validation
split is highly uneven (fog 1,765, dry 124, rain 15, snow 14), so the aggregate
score is useful but the rain and snow condition scores are not yet strong
evidence of generalization. Additional episode-level validation is required
before making safety claims.

At runtime the weather model is a cautious speed-cap advisor. It does not steer,
brake directly, classify risk, or replace the deterministic collision envelope.
Unity forces dry weather to factor 1.0, smooths adverse factor changes, and uses
the former fixed rain/snow/fog percentages only when the model service is absent
or stale. This dataset has no ground-truth sensor-health label, so this model
must not be described as measuring camera, LiDAR, radar, GPS, or IMU reliability.

## Online safety behavior

The learned model requests a maneuver; it never writes a transform.
`online_policy.py` applies these constraints before a request reaches the
simulated vehicle:

- emergency stopping envelope for pedestrians and dangerously close front actors;
- a deterministic deadlock escape: a full stop behind a stationary obstacle with
  a clear adjacent lane changes lanes instead of waiting forever (this mirrors
  the rule controller, because the learned policy alone could not reliably
  initiate an overtake);
- rejection of a lane change when its front or rear clearance is unsafe;
- three consecutive model votes before accepting a lane change;
- conservative deceleration for low-confidence, high-risk, or extreme-risk requests;
- continued 30 Hz safety/physics updates around the 5 Hz learned decisions.

## Virtual-sensor reliability models

- Models: three histogram gradient-boosting regressors (camera, LiDAR, radar)
- Training source: reproducible RoadWeave object-level fault simulation
- Dataset: 12,000 group-disjoint scenarios / 36,000 sensor windows
- Window: 3 seconds
- Conditions: dry, rain, snow, fog
- Fault families: dropout, range noise, velocity noise, bias, latency, false
  positives, calibration drift, complete failure, and nominal operation
- Test MAE: camera 0.0137; LiDAR 0.0171; radar 0.0168
- Test R²: camera 0.986; LiDAR 0.984; radar 0.989
- `FAILED` status recall: camera 0.931; LiDAR 0.984; radar 1.000
- Status macro-F1: camera 0.957; LiDAR 0.962; radar 0.713

The continuous target is calculated from held-out ground truth using detection
recall and precision, range and velocity accuracy, message freshness, and track
continuity. Injected fault name and severity are retained only for evaluation;
they are excluded from training inputs. The learned scores substantially beat
the mean-confidence baseline on MAE, while the lower discrete status F1—most
noticeably radar—reflects rare boundary classes rather than poor continuous
regression. The continuous score is therefore the primary metric.

These are virtual-sensor models, not hardware-certified estimators. They prove
that RoadWeave can detect and react to simulated degradation through a stable
interface. They do not prove transfer to a particular physical camera, LiDAR,
or radar. Hardware deployment requires synchronized gateway health windows,
controlled real fault injection, calibration checks, and external validation.

## Required next research work

1. Improve cross-domain generalization and calibration; the risk model's ~0.08
   and policy model's ~0.14–0.20 leave-one-source-out scores show that labeling
   alone does not close the domain gap.
2. Add an NGSIM adapter with an explicit feet→metres unit conversion. Its
   K-Risk release retains native feet despite the `info.txt` labelling them as
   metres, so it needs a separately verified unit test before use.
3. Compare this baseline against the deterministic controller (see
   `Tools/benchmark_controllers.py`) on minimum TTC, clearance, intervention
   count, deadlock time, route progress, and comfort.
4. Keep the ML controller in Test Lab/simulation until those checks are complete.
5. Validate the reliability feature distributions against a real sensor gateway
   and retrain/calibrate the three regressors before using hardware scores.

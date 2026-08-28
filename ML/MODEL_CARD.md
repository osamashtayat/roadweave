# RoadWeave baseline model card

## Status

This is a research prototype and demonstration baseline, not a controller for
a physical vehicle. The deterministic collision supervisor must remain active.

Shared-label v2 training completed on 2026-08-28 with Python 3.9.6 and
scikit-learn 1.6.1. All
splits are group-disjoint: rows from the same nuScenes scene or K-Risk event
cannot appear on both sides of a train/validation/test boundary.

## Risk model

- Model: histogram gradient-boosting classifier
- Rows: 39,918 (15,908 nuScenes; 24,010 K-Risk)
- Independent groups: 24,844
- Test accuracy: 0.8423
- Test macro-F1: 0.7580
- Per-class test F1: LOW 0.923; MODERATE 0.639; HIGH 0.703; EXTREME 0.767
- Leave-one-source-out macro-F1: 0.5011 (train nuScenes → K-Risk),
  0.2882 (train K-Risk → nuScenes)
- Vehicle-overlap subset for K-Risk → nuScenes: 0.4411 macro-F1 (7,575 rows)

Both adapters now call the same future physical label function. The target is
the measured next-second outcome using current-path TTC, required
deceleration, acceleration, pedestrian clearance, and overlap. The input is
only the previous/current second. Compared with the old 0.0758/0.0746 transfer
scores, this is a large improvement, but the asymmetric K-Risk → nuScenes
result still reflects pedestrians and urban interactions absent from K-Risk.
It is not evidence of real-vehicle generalization.

## Policy model

- Model: histogram gradient-boosting classifier
- Model: flat five-way histogram gradient-boosting classifier
- Rows: 39,918
- Test accuracy: 0.7965
- Test macro-F1: 0.5609
- Per-class test F1: KEEP 0.880; ACCELERATE 0.686; DECELERATE 0.635;
  CHANGE_LEFT 0.267; CHANGE_RIGHT 0.336
- Leave-one-source-out macro-F1: 0.3766 (train nuScenes → K-Risk),
  0.4876 (train K-Risk → nuScenes)
- Vehicle-overlap subset for K-Risk → nuScenes: 0.5063 macro-F1

Every core K-Risk row is now labelled from actual next-second ego motion. The
372 GPT-4.1 recommended actions remain in a separate audit-only Parquet file
and are never fitted. This makes the task harder and explains why the ordinary
macro-F1 is below the former 0.7612, but transfer is substantially more honest
and improves from the former 0.1426/0.2023. A two-stage lateral/longitudinal
classifier was also evaluated; it fell to 0.5244 ordinary and about 0.36 in
both source holdouts, so the flat model was retained.

## Controller benchmark

The selected v2 pair was evaluated on six unseen procedural seeds for 120
seconds each. Rule/ML mean progress was 861.14/940.53 m; minimum clearance was
-3.47/1.14 m; minimum TTC was 0.009/0.493 s; deadlock time was 2.92/1.20 s;
mean absolute acceleration was 1.216/1.099 m/s²; and mean absolute yaw was
2.228/2.185 degrees/s. The ML controller therefore improved this closed-loop
benchmark, although six procedural seeds are still a prototype-scale study and
not a safety certification.

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
- artifact-specific OOD envelopes that request cautious deceleration when too
  many physical features fall outside the training distribution;
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

1. Continue cross-domain calibration. Shared physical labels raised source
   holdouts substantially, but risk remains asymmetric (0.501/0.288) and policy
   remains moderate (0.377/0.488). A source classifier still identifies the
   dataset at about 0.99 macro-F1, proving residual covariate shift.
2. Add an NGSIM adapter with an explicit feet→metres unit conversion. Its
   K-Risk release retains native feet despite the `info.txt` labelling them as
   metres, so it needs a separately verified unit test before use.
3. Expand the six-seed closed-loop benchmark with more routes, confidence
   intervals, ablations, and unseen scenario families.
4. Keep the ML controller in Test Lab/simulation until hardware-domain OOD,
   calibration, and controlled fault tests are complete.
5. Validate the reliability feature distributions against a real sensor gateway
   and retrain/calibrate the three regressors before using hardware scores.

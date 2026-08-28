# RoadWeave ML pipeline

This folder contains the reproducible baseline that turns nuScenes and K-Risk
trajectories into a shared observation format, trains the risk and maneuver
models, and optionally uses those models inside the RoadWeave procedural live
stream. A third, independent weather-speed model is trained from the Extreme
Driving Conditions Dataset and is used by the Unity Test Lab for rain, snow,
and fog. Three additional regressors estimate the reliability of RoadWeave's
object-level virtual camera, LiDAR, and radar from rolling health features.

The Unity project is not coupled to pandas, Parquet, scikit-learn, or either
training dataset. Unity still receives the same `TwinSnapshot` UDP messages.
The learned controller lives on the simulated-source side and changes the ego
vehicle's requested high-level behavior before the source publishes a
snapshot.

## Data flow

```text
nuScenes + CAN ----------------> build_nuscenes.py ---+
                                                       +--> canonical 1 s / 2 Hz tables
K-Risk trajectories ----------> build_krisk.py ------+             |
                                                                    v
                                                   risk_model + policy_model
                                                                    |
                           deterministic safety checks <--- online_policy.py
                                                                    |
                                     simulated_twin_stream.py --> TwinSnapshot
                                                                    |
                                                                  Unity

Extreme Driving episode tars --> build_extreme_weather.py
                                      |
                                      +--> 3 s motion + weather features
                                                    |
                                                    v
                                          weather_model (speed factor)
                                                    |
Unity Test Lab observation --> test_lab_ml_service.py --> Unity controller

Unity actor truth --> virtual camera/LiDAR/radar --> fused observation
                              |                         |
                              +--> rolling health -----+
                                        |
                camera/LiDAR/radar reliability models --> safety mode
```

Each training row has dataset, underlying-domain, group, timing, and label
origin metadata plus canonical physical summaries. Both adapters use exactly
one measured second of past/current state, resampled to 2 Hz. Measurements in
the following second create the risk and actual-motion targets but never enter
the feature matrix. No raw risk score, folder name, future pose, GPT response,
or dataset-native action label is admitted as a feature.

## Current labels

- Risk: `LOW`, `MODERATE`, `HIGH`, `EXTREME`
- Policy: `KEEP`, `ACCELERATE`, `DECELERATE`, `CHANGE_LEFT`, `CHANGE_RIGHT`
- Speed: metres/second
- Acceleration: metres/second squared
- Distance: metres
- Yaw rate: degrees/second
- TTC: seconds, capped at 20

`common_labels.py` assigns both datasets with the same physical rules. Risk is
derived from future TTC, required deceleration, ego acceleration, pedestrian
clearance, and overlap. Policy is the ego's measured future speed and
curvature-normalized lateral motion. The 372 GPT-4.1 recommendations are now
written only to `policy_krisk_recommended.parquet` for audit; the deployable
policy model never trains on them. This corrects the old mismatch where
nuScenes meant “what the driver did” but K-Risk meant “what GPT recommends.”

The K-Risk converter supports highD, CitySim ExpresswayA/FreewayB (including
the extreme TTC folders), inD, and rounD. NGSIM is deliberately not mixed in:
the release retains native feet despite its `info.txt` unit label, so its
feet-to-metres conversion needs a separately verified adapter and unit test.

## Reproduce everything

Run these commands from `/Users/asus/Desktop/roadweave`:

```bash
source ML/.venv/bin/activate

python ML/src/build_krisk.py
python ML/src/build_nuscenes.py --overwrite --validate
python ML/src/inspect_data.py

python ML/src/train_model.py --task risk --feature-profile all
python ML/src/train_model.py --task policy --feature-profile transfer
python ML/src/audit_domains.py --task risk
python ML/src/audit_domains.py --task policy

python ML/src/build_extreme_weather.py
python ML/src/train_weather_model.py

python ML/src/generate_sensor_reliability.py
python ML/src/train_sensor_reliability.py

python ML/src/test_prediction.py --task risk
python ML/src/test_prediction.py --task policy

python -m unittest discover -s ML -p 'test*.py' -v
python Tools/simulated_twin_stream.py --self-test
python Tools/simulated_twin_stream.py --controller ml --self-test
python Tools/benchmark_controllers.py --seeds 6 --seconds 120
```

The converters default to the dataset locations used on this Mac:

- `/Users/asus/Downloads/v1.0-trainval`
- `/Users/asus/Downloads/32896772/K-Risk_data`
- `/Users/asus/Downloads/RoadWeaveData/Extreme_Driving_Conditions_Dataset`

Use `--dataroot` or `--data-root` when those folders move. Both builders also
provide small smoke-test options; run each script with `--help` for the exact
arguments.

## Run the ML controller with Unity

First stop any older Python simulator that is using UDP ports 5055 or 5056.
Then run:

```bash
cd /Users/asus/Desktop/roadweave
source ML/.venv/bin/activate
python Tools/simulated_twin_stream.py --controller ml
```

Open Unity, press Play, and then press Drive. Unity's existing simulated-stream
adapter continues to receive `TwinSnapshot` messages. To use the established
rule controller again, omit `--controller ml` (rule mode remains the default).

The ML policy makes a high-level decision at 5 Hz while the world, sensors,
collision supervisor, and outgoing stream continue at 30 Hz. A reset creates a
new controller and clears its one-second history. Emergency pedestrian/front
object braking, lane-clearance rejection, and three-vote lane-change
confirmation remain deterministic safety constraints. Each model artifact also
contains a learned feature envelope. A substantially out-of-distribution
observation requests cautious deceleration and exposes its OOD score in the
runtime diagnostics; it never disables the deterministic supervisor.

## Run the ML models in the Test Lab

The Test Lab uses a separate local UDP inference service so all three models
can influence the autonomous test vehicle without coupling Unity to Python.
Start it before pressing Create Test / Run Test:

```bash
cd /Users/asus/Desktop/roadweave
source ML/.venv/bin/activate
python Tools/test_lab_ml_service.py
```

The service listens on `127.0.0.1:5075` (override with `--host`/`--port`) and
answers the versioned `roadweave.testlab-ml/1.0` protocol. It loads the risk,
policy, weather, camera-reliability, LiDAR-reliability, and radar-reliability
artifacts. While it runs, the
Test Lab's `TestLabMlDecisionBridge` requests a decision at 5 Hz and feeds the
returned target speed, requested lane change, and learned adverse-weather speed
cap into `AutonomousTestVehicleController`. Rain, snow, and fog use the
weather model; dry weather deliberately stays at factor 1.0. If the service or
weather artifact is unavailable, the Test Lab keeps driving on its existing
deterministic weather fallback, so a missing Python process never blocks a
test. The model's normalized output is converted back through its 50 km/h
training reference into an absolute target speed. Unity then smooths the
corresponding factor and clips it to 0.35–1.0; weather alone therefore cannot
command a complete stop.

The Test Lab vehicle also creates an object-level virtual camera (15 Hz),
LiDAR (10 Hz), and radar (20 Hz). Each channel has different range, latency,
noise, weather sensitivity, and injected fault behavior. Their observations
are confidence-fused before the risk/policy models see them. A rolling
three-second window supplies dropout, freshness, confidence, continuity,
variance, innovation, disagreement, ego-motion, and weather features to the
three reliability regressors. The service returns per-sensor scores plus one
redundancy-aware safety mode:

- `NORMAL`: all available channels are acceptable;
- `CAUTIOUS`: at least two channels remain acceptable;
- `RESTRICTED`: only limited sensing remains, so speed is capped and new
  overtakes are rejected;
- `MINIMAL_RISK`: no usable channel remains, so the prototype stops.

The learned policy and lane planner consume the noisy fused observations. The
uncorrupted Unity actor scan is retained only as the final collision envelope;
this keeps the research prototype safe without leaking perfect information
into the learned decisions.

For the risky-driver demonstration, press **Create Test** and then the runtime
**Risk Driving** button. The simulated driver requests 80 km/h while the ML
policy continues to receive the safe 30 km/h cruise reference and current
sensor observations. The results panel shows the driver request, current ML
target, ML intervention count, and deterministic safety-intervention count.
The button is created at runtime from the existing Run Test style, so the saved
Unity Canvas and the normal Live Twin layout are not modified.

Validate the service without Unity:

```bash
python Tools/test_lab_ml_service.py --self-test
```

As in the live twin, the learned maneuver never writes a transform: it is
supervised by the same deterministic safety envelope (pedestrian/front
emergency braking, lane-clearance rejection, and three-vote lane-change
confirmation).

## Outputs from the completed run

Prepared data, saved models, and reports are generated locally under:

- `ML/data/processed/`
- `ML/models/`
- `ML/reports/`

They are ignored by Git because they are reproducible artifacts. The completed
shared-label run produced 39,918 rows for each task: 15,908 nuScenes windows
and 24,010 K-Risk events. See `MODEL_CARD.md` for the measured within-domain,
leave-one-source-out, leave-one-domain-out, and closed-loop performance
and the limitations that matter before presenting or publishing this baseline.

## File guide

- `src/features.py`: canonical current/history feature contract
- `src/labels.py`: stable numeric and named target definitions
- `src/common_labels.py`: identical future physical targets for every adapter
- `src/build_krisk.py`: K-Risk discovery, conversion, validation, and report
- `src/build_nuscenes.py`: nuScenes/CAN conversion and future-only policy labels
- `src/nuscenes_lite.py`: read-only metadata/CAN loader without plotting dependencies
- `src/inspect_data.py`: read-only pre-training quality gate
- `src/model_support.py`: schema, transfer profile, OOD envelope, and prediction code
- `src/train_model.py`: balanced fitting plus group/source/domain holdout evaluation
- `src/audit_domains.py`: source classifier and per-feature distribution-drift audit
- `src/hierarchical_policy.py`: reproducible rejected two-stage policy experiment
- `src/test_prediction.py`: explains one saved-model prediction
- `src/online_policy.py`: 5 Hz learned policy plus deterministic safety wrapper
- `src/weather_features.py`: canonical three-second weather/motion feature contract
- `src/build_extreme_weather.py`: streaming Extreme Driving tar converter
- `src/train_weather_model.py`: group-separated weather regressor training/evaluation
- `src/weather_model.py`: weather artifact validation and inference helper
- `src/sensor_reliability_features.py`: canonical virtual-sensor health contract
- `src/generate_sensor_reliability.py`: reproducible fault-injection data generator
- `src/train_sensor_reliability.py`: one group-split reliability regressor per sensor
- `src/sensor_reliability_model.py`: reliability artifact validation and inference
- `Tools/benchmark_controllers.py`: rule-vs-ML safety/comfort comparison
- `Tools/test_lab_ml_service.py`: local decision, weather, and reliability service
- `tests/`: feature, splitting, safety, and runtime unit tests

## Transition to a future sensor gateway

Do not send raw hardware-specific fields directly into the models. A future
sensor gateway should first map GPS/IMU/CAN/perception outputs into the same
ego-relative physical signals used by `features.py` (ego motion, current-front
object, adjacent front/rear objects, pedestrian, and lane-clear flags). The
one-second 2 Hz summarizer and saved model interface can then remain unchanged.
Unity remains downstream of `TwinSnapshot`, so replacing the Python simulator
does not require Unity to understand the training datasets or model library.
The reliability models have the same boundary: a gateway may publish the
optional `sensorHealth` contract after computing equivalent health statistics.
The current regressors are trained only on RoadWeave virtual-sensor faults,
however, so they must be validated or retrained with hardware fault-injection
data before their scores are interpreted as real sensor reliability.

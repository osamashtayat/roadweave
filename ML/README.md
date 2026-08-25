# RoadWeave ML pipeline

This folder contains the reproducible baseline that turns nuScenes and K-Risk
trajectories into a shared observation format, trains two classical machine
learning models, and optionally uses those models inside the RoadWeave
procedural live stream.

The Unity project is not coupled to pandas, Parquet, scikit-learn, or either
training dataset. Unity still receives the same `TwinSnapshot` UDP messages.
The learned controller lives on the simulated-source side and changes the ego
vehicle's requested high-level behavior before the source publishes a
snapshot.

## Data flow

```text
nuScenes + CAN ----------------> build_nuscenes.py ---+
                                                       +--> canonical 3 s feature tables
selected K-Risk trajectories --> build_krisk.py ------+             |
                                                                    v
                                                   risk_model + policy_model
                                                                    |
                           deterministic safety checks <--- online_policy.py
                                                                    |
                                     simulated_twin_stream.py --> TwinSnapshot
                                                                    |
                                                                  Unity
```

Each training row has source/group metadata, one target, and 210 canonical
features. The features are six summaries (`now`, `min`, `max`, `mean`, `std`,
and `trend`) of 35 physical signals over up to three seconds. No raw risk
score, folder name, future pose, GPT response, or action label is admitted as
a feature.

## Current labels

- Risk: `LOW`, `MODERATE`, `HIGH`, `EXTREME`
- Policy: `KEEP`, `ACCELERATE`, `DECELERATE`, `CHANGE_LEFT`, `CHANGE_RIGHT`
- Speed: metres/second
- Acceleration: metres/second squared
- Distance: metres
- Yaw rate: degrees/second
- TTC: seconds, capped at 20

nuScenes supplies conservatively selected `LOW` risk windows and routine
policy examples. K-Risk supplies `MODERATE`, `HIGH`, and `EXTREME` risk events.
The K-Risk policy subset uses the 372 available GPT-4.1 recommended-action
labels; it does not learn to imitate a dangerous driver's recorded maneuver.

The K-Risk converter currently supports highD and CitySim ExpresswayA/FreewayB
(including the extreme TTC folders). inD, NGSIM, and rounD are deliberately not
mixed in yet because their coordinate systems, units, sampling rates, and
schemas require separately verified adapters.

## Reproduce everything

Run these commands from `/Users/asus/Desktop/roadweave`:

```bash
source ML/.venv/bin/activate

python ML/src/build_krisk.py --fail-on-error
python ML/src/build_nuscenes.py --overwrite --validate
python ML/src/inspect_data.py

python ML/src/train_model.py --task risk
python ML/src/train_model.py --task policy

python ML/src/test_prediction.py --task risk
python ML/src/test_prediction.py --task policy

python -m unittest discover -s ML -p 'test*.py' -v
python Tools/simulated_twin_stream.py --self-test
python Tools/simulated_twin_stream.py --controller ml --self-test
```

The converters default to the dataset locations used on this Mac:

- `/Users/asus/Downloads/v1.0-trainval`
- `/Users/asus/Downloads/32896772/K-Risk_data`

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
new controller and clears its three-second history. Emergency pedestrian/front
object braking, lane-clearance rejection, and three-vote lane-change
confirmation remain deterministic safety constraints.

## Run the ML models in the Test Lab

The Test Lab uses a separate local UDP inference service so the same two models
can influence the autonomous test vehicle without coupling Unity to Python.
Start it before pressing Create Test / Run Test:

```bash
cd /Users/asus/Desktop/roadweave
source ML/.venv/bin/activate
python Tools/test_lab_ml_service.py
```

The service listens on `127.0.0.1:5075` (override with `--host`/`--port`) and
answers the versioned `roadweave.testlab-ml/1.0` protocol. While it runs, the
Test Lab's `TestLabMlDecisionBridge` requests a decision at 5 Hz and feeds the
returned target speed (and a requested lane change) into
`AutonomousTestVehicleController`. If the service is not running, the Test Lab
keeps driving on its deterministic rule fallback, so a missing service never
blocks a test.

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
run produced 26,376 risk rows and 15,045 policy rows. See `MODEL_CARD.md` for
the measured performance and the limitations that matter before presenting or
publishing this baseline.

## File guide

- `src/features.py`: canonical current/history feature contract
- `src/labels.py`: stable numeric and named target definitions
- `src/build_krisk.py`: K-Risk discovery, conversion, validation, and report
- `src/build_nuscenes.py`: nuScenes/CAN conversion and future-only policy labels
- `src/inspect_data.py`: read-only pre-training quality gate
- `src/model_support.py`: shared loading, schema validation, and prediction code
- `src/train_model.py`: group-separated fitting, evaluation, artifacts, and reports
- `src/test_prediction.py`: explains one saved-model prediction
- `src/online_policy.py`: 5 Hz learned policy plus deterministic safety wrapper
- `tests/`: feature, splitting, safety, and runtime unit tests

## Transition to a future sensor gateway

Do not send raw hardware-specific fields directly into the models. A future
sensor gateway should first map GPS/IMU/CAN/perception outputs into the same
ego-relative physical signals used by `features.py` (ego motion, current-front
object, adjacent front/rear objects, pedestrian, and lane-clear flags). The
three-second summarizer and saved model interface can then remain unchanged.
Unity remains downstream of `TwinSnapshot`, so replacing the Python simulator
does not require Unity to understand the training datasets or model library.


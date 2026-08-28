# RoadWeave virtual-sensor reliability implementation

## What was implemented

RoadWeave now has an object-level camera, LiDAR, and radar reliability pipeline
inside Test Lab. This is deliberately smaller and more defensible than claiming
raw photorealistic sensor simulation: Unity begins from actors it already knows,
then three virtual channels independently miss, delay, bias, and corrupt those
actor observations.

The system answers two different questions:

1. What does the autonomous controller currently perceive after noisy sensor
   fusion?
2. How trustworthy does each sensor channel appear from its recent behavior?

The first answer is sent to the existing risk and policy models. The second is
sent to three new reliability regressors. Perfect Unity actor truth is not sent
to either learned decision. It is kept only for the final deterministic
collision supervisor.

## Runtime data flow

```text
Unity scenario actors
        |
        v
SimulatedVehicleSensorSuite (object-level ground truth)
        |
        +-------------------------------> final collision safety envelope
        |
        v
VirtualPerceptionSensorRig
  camera 15 Hz | LiDAR 10 Hz | radar 20 Hz
  weather + noise + latency + faults
        |
        +--> confidence-weighted fused actor observations
        |         |
        |         +--> risk/policy feature history
        |
        +--> three-second health windows
                  |
                  v
TestLabMlDecisionBridge -- UDP roadweave.testlab-ml/1.0 --> Python service
                                                             |
                           +---------------------------------+----------------+
                           |                 |               |                |
                        risk model       policy model   weather model   3 reliability
                           |                 |               |           regressors
                           +-----------------+---------------+----------------+
                                                             |
                                                             v
             action + target speed + risk + weather cap + per-sensor reliability
                                                             |
                                                             v
AutonomousTestVehicleController
  NORMAL / CAUTIOUS / RESTRICTED / MINIMAL_RISK
  deterministic collision supervisor remains final authority
```

## The three simulated channels

`Assets/Scripts/TestLab/VirtualPerceptionSensorRig.cs` adds itself to the Test
Lab ego vehicle at runtime. No Canvas or saved scene layout was changed.

- Camera: 55 m nominal range, 15 Hz, most sensitive to fog/snow and moderate
  range and velocity noise.
- LiDAR: 70 m nominal range, 10 Hz, accurate range but degraded by precipitation.
- Radar: 100 m nominal range, 20 Hz, best closing-speed measurement and strongest
  resistance to weather.

Fault profiles are reproducible from seed 2026 and rotate every 25 seconds.
Available faults are dropout, range noise, velocity noise, bias, latency, false
positive, calibration drift, and complete failure. A nominal profile is also
available. The three channels are fused by confidence, so one failed channel
does not automatically erase a valid obstacle detected by the other two.

## Health feature contract

Each channel summarizes its previous three seconds into:

- dropout rate;
- mean and maximum message age;
- mean and standard deviation of detection count;
- mean and standard deviation of confidence;
- track continuity;
- range and velocity variance;
- innovation mean and standard deviation;
- disagreement with the other sensor channels;
- ego speed and yaw-rate mean and standard deviation;
- one-hot dry/rain/snow/fog context.

Fault type and fault severity are never included. The model must infer degraded
reliability from symptoms observable at runtime.

## Dataset and label construction

`ML/src/generate_sensor_reliability.py` creates 12,000 independent scenarios.
Each scenario produces one camera, one LiDAR, and one radar row, for 36,000
total rows. All three rows from a scenario stay in the same train, validation,
or test split, preventing scenario leakage.

The continuous reliability target is calculated against known simulation truth:

- observation quality combines detection recall, precision, range accuracy,
  velocity accuracy, and track continuity;
- availability combines actual message delivery and message freshness;
- final reliability is observation quality multiplied by availability.

A valid message on an empty road is treated as healthy. A missing or stale
message is not. This prevents the model from confusing “nothing is nearby”
with “the sensor has failed.”

This label design is why the dataset is useful: reliability is a measurable
quality score, not the injected fault name renamed as a target.

Generated artifacts are local and reproducible:

- `ML/data/processed/sensor_reliability.parquet`;
- `ML/reports/sensor_reliability_generation.json`.

## Model training

`ML/src/train_sensor_reliability.py` trains one
`HistGradientBoostingRegressor` for each sensor. A separate model is appropriate
because the same symptom has a different meaning for camera, LiDAR, and radar.
The fixed scenario-group split is used for all three models.

The trained artifacts are:

- `ML/models/sensor_camera_reliability_model.joblib`;
- `ML/models/sensor_lidar_reliability_model.joblib`;
- `ML/models/sensor_radar_reliability_model.joblib`.

The evaluation report and row-level predictions are:

- `ML/reports/sensor_reliability_metrics.json`;
- `ML/reports/sensor_reliability_predictions.csv`.

The full evaluation includes continuous regression metrics, thresholded status
confusion matrices, weather and fault breakdowns, a confidence-only baseline,
permutation importance, and leave-one-fault-out experiments.

## Runtime safety modes

The Python service returns every sensor score and a redundancy-aware mode:

- `NORMAL`: every available sensor is at least acceptable;
- `CAUTIOUS`: two or more sensors remain acceptable; Unity caps speed to 75%;
- `RESTRICTED`: at least one sensor remains usable; Unity caps speed to 45% and
  rejects new overtakes;
- `MINIMAL_RISK`: all channels are failed; Unity commands a stop.

The overall score is 75% of the median channel and 25% of the weakest channel.
This preserves redundancy while making a failed channel visible.

## Reproduction

From `/Users/asus/Desktop/roadweave`:

```bash
source ML/.venv/bin/activate
python ML/src/generate_sensor_reliability.py
python ML/src/train_sensor_reliability.py
python -m unittest discover -s ML/tests -v
python Tools/test_lab_ml_service.py --self-test
python Tools/test_lab_ml_service.py
```

Then open Unity, enter Test Lab, and run a scenario. The existing ML status text
reports the aggregate sensor percentage and safety mode; no Canvas element was
added or moved.

## Future real sensor gateway

`TwinSnapshot.sensorHealth` is optional so older replay and simulated-stream
sources remain compatible. A future gateway can compute the same rolling
features from real camera, LiDAR, and radar diagnostics and publish them without
changing Unity's visual or test architecture.

The saved virtual models should not immediately be reused as real-hardware
truth. First collect synchronized gateway windows under healthy operation and
controlled degradations (occlusion, packet loss, timing delay, calibration
offset, and weather), calculate targets against a trusted reference, compare
feature distributions, then retrain or calibrate. The interface is reusable;
the current numerical model is a simulation baseline.

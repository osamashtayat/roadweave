# RoadWeave weather-speed model implementation

## Scope

RoadWeave now has a third ML artifact for Test Lab weather behavior. It is
separate from the existing risk classifier and maneuver-policy classifier. The
model supports rain, snow, and fog. Dry weather bypasses the weather model and
retains the requested cruise speed.

This is a research prototype. The Extreme Driving Conditions Dataset does not
provide a ground-truth sensor-health or sensor-reliability label. It can support
a later proxy reliability experiment using synthetic camera/LiDAR degradation
or cross-sensor disagreement, but the current weather-speed model must not be
presented as a direct measurement of sensor reliability.

## Dataset audit and conversion

Dataset root:

`/Users/asus/Downloads/RoadWeaveData/Extreme_Driving_Conditions_Dataset`

`ML/src/build_extreme_weather.py` scans the episode tar files without extracting
their large camera, depth, or LiDAR payloads. It reads `metadata.jsonl` and
`episode_annotation.json` directly from each archive. The converter:

1. identifies dry, rain, snow, and fog/haze episodes from category and weather
   annotations;
2. reads the 4 Hz ego state (`speed`, `yaw rate`, and `longitudinal acceleration`);
3. forms a past-only three-second history for every usable decision point;
4. calculates the following one-second mean speed only as the supervised target;
5. keeps every episode as a group so frames from one episode cannot cross an
   evaluation split;
6. writes the canonical table to
   `ML/data/processed/weather_extreme.parquet`.

Completed conversion:

- 589 archives scanned;
- 413 selected episodes and 18,787 rows;
- fog: 310 episodes / 14,236 rows;
- dry: 65 episodes / 2,761 rows;
- rain: 22 episodes / 1,528 rows;
- snow: 16 episodes / 262 rows;
- zero conversion errors.

The detailed machine-readable conversion record is
`ML/reports/weather_conversion_report.json`.

## Feature and target contract

The schema is `roadweave.weather-observation/1.0`, implemented in
`ML/src/weather_features.py`.

The trained model uses 16 features:

- six past-three-second summaries of longitudinal acceleration: current,
  minimum, maximum, mean, standard deviation, and trend;
- the same six summaries for yaw rate;
- one-hot dry, rain, snow, and fog context.

The target is:

`mean speed during the next second / 50 km/h`

and is clipped to 0.35–1.0. This represents a learned cautious speed cap between
17.5 and 50 km/h. Current-speed features were deliberately removed. When they
were included, the model mostly copied current speed into future speed and
produced an unrealistically excellent score without learning useful weather or
motion behavior.

## Training and evaluation

`ML/src/train_weather_model.py` trains a scikit-learn
`HistGradientBoostingRegressor`. Weather-condition balancing gives rare snow and
rain rows more influence than their raw counts would provide. Five-fold
`StratifiedGroupKFold` evaluation keeps complete episodes together.

Outputs:

- model: `ML/models/weather_model.joblib`;
- metrics: `ML/reports/weather_metrics.json`;
- row predictions: `ML/reports/weather_test_predictions.csv`.

Measured results:

- training set: 16,869 rows / 379 episodes;
- official validation: 1,918 rows / 34 episodes;
- episode-separated five-fold CV MAE: 0.1263;
- prior fixed-percentage rule CV MAE: 0.2106;
- official validation MAE: 0.1401;
- prior fixed-percentage rule validation MAE: 0.1843;
- official validation R²: 0.2033.

The official validation set is imbalanced: fog has 1,765 rows, dry 124, rain
15, and snow 14. The aggregate result and fog result are useful prototype
evidence, but the rain and snow validation subsets are too small for strong
generalization or safety claims.

## Runtime data flow

```text
Unity Test Lab
  -> TestLabMlDecisionBridge (5 Hz UDP observation)
  -> Tools/test_lab_ml_service.py
  -> TestLabInferenceEngine
       -> risk_model.joblib
       -> policy_model.joblib
       -> weather_model.joblib for Rain/Snow/Fog
  -> JSON decision response
       -> risk + maneuver target
       -> weather model used flag
       -> normalized weather factor
       -> absolute weather target speed
  -> AutonomousTestVehicleController
       -> smooth speed cap
       -> sensor and collision safety supervisor
       -> vehicle motion
```

The weather inference reuses the same Unity ego history already sent to the
risk and policy service. The normalized model result is multiplied by the
artifact's 50 km/h training reference to produce an absolute speed cap. This
prevents a 0.5 prediction from meaning different physical speeds during normal
cruise, overtaking, and the 80 km/h risky-driver demonstration.

The Unity controller does not permit the weather model to write transforms,
steer, or command a zero-speed stop. It converts the absolute cap to the current
driving mode, clips the resulting factor to 0.35–1.0, and approaches it smoothly.
Pedestrian and vehicle collision envelopes remain deterministic and have final
authority.

If the weather artifact, Python service, or a fresh matching response is absent,
Unity uses the previous rain/snow/fog percentages from `TestWeatherController`
as a fallback. The Canvas was not edited.

## Reproduce

From `/Users/asus/Desktop/roadweave`:

```bash
source ML/.venv/bin/activate
python ML/src/build_extreme_weather.py
python ML/src/train_weather_model.py
python -m unittest discover -s ML -p 'test*.py' -v
python Tools/test_lab_ml_service.py --self-test
python Tools/test_lab_ml_service.py
```

Start the final service command before opening the Test Lab. In Unity, choose
Rain, Snow, or Fog and run the test. The ML status includes the active weather
condition and the learned target speed. Dry does not invoke the weather model.

## Sensor-reliability follow-up

A later sensor-reliability model needs an explicit target such as known camera
occlusion, injected LiDAR point loss, localization error, or measured
cross-sensor inconsistency. A defensible experiment can corrupt clean synchronized
samples at known severities and train the model to estimate that known severity.
That would be a new model and evaluation task; it is intentionally not hidden
inside the weather-speed model implemented here.

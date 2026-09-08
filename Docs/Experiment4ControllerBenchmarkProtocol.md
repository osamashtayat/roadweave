# RoadWeave Experiment 4 — Rule versus Hybrid ML Controller

## Research question

Under identical closed-loop road and actor scenarios, how does the RoadWeave
Hybrid ML Controller change safety, progress, comfort, maneuver completion, and
scenario success compared with the deterministic Rule Controller?

This is an offline benchmark. Unity does not need to be open. The benchmark uses
the same procedural world, simulated sensors, vehicle dynamics, risk model,
policy model, and safety layer used by RoadWeave's live simulated source.

## Fixed experimental conditions

- Independent variable: controller (`Rule Controller` or `Hybrid ML Controller`).
- 30 paired seeds: 2026000 through 2026029.
- 300 simulated seconds per controller/seed.
- Fixed simulation step: 0.05 seconds.
- Fixed ML decision rate: 5 Hz.
- Dry weather and normal sensor reliability.
- The risk and policy model files are not retrained between runs.
- A complete immutable scenario manifest is generated once per seed and copied
  to both controllers.

The configuration file records SHA-256 hashes of both model artifacts and the
benchmark/controller source files, plus the Python and package versions. These
values prove exactly which trained models and software environment were
evaluated. Do not edit these files between starting and resuming an experiment.

## Metric definitions

- `route_progress_m`: final distance along the route coordinate.
- `collision_count`: unique actors for which the world had to prevent an
  attempted overlap. The stabilizing non-penetration correction remains active.
- `minimum_ttc_s`: lowest finite front/pedestrian time-to-collision.
- `minimum_clearance_m`: closest signed vehicle-to-actor boundary distance;
  zero is contact and a negative value is overlap.
- `emergency_stops`: transitions into emergency behavior, not frames stopped.
- `deadlock_duration_s`: accumulated time below 0.3 m/s with a vehicle blocking
  the current lane within 15 m.
- `overtake_attempts`: distinct actors for which a lane-change pass was committed.
- `successful_overtakes`: target passed by at least 10 m and ego stabilized in a lane.
- `mean_absolute_acceleration_mps2`: mean acceleration magnitude after a
  two-second startup exclusion.
- `maximum_absolute_jerk_mps3`: maximum step-to-step acceleration change divided
  by 0.05 seconds, also after the startup exclusion.
- `mean_absolute_yaw_rate_deg_s`: steering activity after startup.
- `scenario_completion_rate`: successful evaluable scenarios divided by
  evaluable scenarios.

A scenario must be encountered at least 30 seconds before the run ends to be
evaluable. It succeeds only if it resolves, has no collision, and accumulates no
more than five seconds of scenario deadlock. The heatmap uses only scenario IDs
that were evaluable for both controllers and prints both percentage and sample
count.

## Stage 1 — Environment check

```bash
cd /Users/asus/Desktop/roadweave
source ML/.venv/bin/activate
python -c "import numpy, pandas, scipy, sklearn, matplotlib; print('Experiment 4 environment ready')"
```

## Stage 2 — Short pilot

Run this before the final experiment:

```bash
python Tools/benchmark_controllers.py \
  --seeds 3 \
  --seed-start 2026000 \
  --seconds 120 \
  --dt 0.05 \
  --output-dir ExperimentResults/experiment4/pilot
```

Expected result:

- Six completed rows: three seeds × two controllers.
- No error text in `experiment4_runs.csv`.
- Two rows per seed with the same `manifest_id`.
- Graphs 11–15 appear automatically in the pilot folder.

The pilot is a software check, not part of the paper's final statistics.

## Stage 3 — Final 30-seed experiment

Use a new output directory. Connect the Mac to power, close heavy applications,
and run this command once. `caffeinate -i` keeps the Mac awake only while the
benchmark process is running:

```bash
caffeinate -i python Tools/benchmark_controllers.py \
  --seeds 30 \
  --seed-start 2026000 \
  --seconds 300 \
  --dt 0.05 \
  --output-dir ExperimentResults/experiment4/final
```

The benchmark is faster than real time because it does not render Unity. On the
current Mac, allow approximately 45–75 minutes; actual time is recorded in
`experiment4_config.json`. Terminal prints and checkpoints each completed
controller/seed, so visible progress is expected throughout the run.

The program runs the Rule and Hybrid ML controllers automatically. Do not open
Unity, start the UDP stream, change model files, or retrain a model while it is
running. Results are checkpointed after every controller/seed run.

If Terminal or the computer interrupts the experiment, run the exact same
command with `--resume` added:

```bash
caffeinate -i python Tools/benchmark_controllers.py \
  --seeds 30 \
  --seed-start 2026000 \
  --seconds 300 \
  --dt 0.05 \
  --output-dir ExperimentResults/experiment4/final \
  --resume
```

The resume check rejects different seeds, duration, step size, controllers,
model hashes, code hashes, or software versions. Completed seed/controller
checkpoints are not repeated.

## Stage 4 — Output validation

The final folder must contain:

- `experiment4_config.json`
- `experiment4_runs.csv` — exactly 60 completed run rows.
- `experiment4_scenarios.csv` — scenario-level evidence.
- `experiment4_valid_paired_runs.csv` — the 60 analysis rows.
- `experiment4_invalid_runs.csv` — header only when nothing failed.
- `experiment4_controller_summary.csv`
- `experiment4_paired_statistics.csv`
- `experiment4_scenario_success.csv`
- Graphs 11–15 as both PNG and PDF.

Check the run file:

```bash
python -c "import pandas as pd; p='ExperimentResults/experiment4/final/experiment4_runs.csv'; d=pd.read_csv(p); print(d.groupby('controller').size()); print('completed=', d.run_completed.astype(str).str.lower().eq('true').sum()); print('errors=', d.error.notna().sum()); print('manifest mismatches=', (d.groupby('seed').manifest_id.nunique()!=1).sum())"
```

Expected values:

- Rule Controller: 30
- Hybrid ML Controller: 30
- completed: 60
- errors: 0
- manifest mismatches: 0

## Stage 5 — Recreate analysis without rerunning simulations

The benchmark already analyzes automatically. If figures need to be recreated:

```bash
python Tools/analyze_experiment4.py \
  --results-dir ExperimentResults/experiment4/final \
  --expected-pairs 30
```

## Paper figures

- Graph 11: four panels of controller means and 95% bootstrap confidence intervals.
- Graph 12: paired-run minimum TTC distributions.
- Graph 13: paired-run minimum-clearance distributions.
- Graph 14: safety-versus-progress scatter plot.
- Graph 15: paired common-scenario success heatmap.

The statistics table reports the mean paired Hybrid-minus-Rule difference, its
95% bootstrap confidence interval, a two-sided Wilcoxon signed-rank p-value,
and paired effect size. Interpret direction using the table's
`preferred_direction` column; a positive difference is not automatically better
for metrics such as collisions, deadlock, acceleration, or jerk.

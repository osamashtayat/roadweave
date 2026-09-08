# RoadWeave Experiment 2: Real-Time Update-Rate Protocol

## Purpose

This experiment changes only the publication rate of the simulated real-time
source. It measures whether RoadWeave continues to receive, apply, and display
canonical `TwinSnapshot` state correctly at 2, 10, 30, 50, and 100 Hz.

The Python publisher and Unity run on the same Mac. The latency results are a
local pipeline baseline covering serialization, loopback transport, validation,
Unity scheduling, and state application. They are not measurements of Ethernet,
Wi-Fi, cellular, or wide-area-network latency. Zero missing scheduled sequences
therefore means the local implementation sustained the tested workload; it does
not establish loss-free operation on a physical network.

The formal rate experiment uses a clear road (`--no-actors`). This isolates the
data pipeline from autonomous-driving decisions and prevents a random scenario
from making one update-rate run different from another. Scenario-heavy stress
testing can be reported separately.

## One-time Unity setup

The `DigitalTwinSystem` object in `RoadWeave.unity` is already configured as
follows:

- `SourceExperimentRecorder` is disabled. It belongs to Experiment 1.
- `UpdateRateExperimentRecorder` is enabled.
- Warm-up duration is 30 seconds.
- Measurement duration is 300 seconds.
- A missing stream for more than 5 seconds stops and invalidates the run.
- A canonical ego/actor overlap stops and invalidates the run.

Do not enable both experiment recorders at the same time. No Canvas or UI
objects need to be changed.

## Run commands

From the RoadWeave project directory, activate the existing environment:

```bash
cd /Users/asus/Desktop/roadweave
source ML/.venv/bin/activate
```

Start one rate at a time. Use the same seed and the clear-road option for every
formal run:

```bash
python Tools/simulated_twin_stream.py --rate 2 --seed 20260831 --no-actors
python Tools/simulated_twin_stream.py --rate 10 --seed 20260831 --no-actors
python Tools/simulated_twin_stream.py --rate 30 --seed 20260831 --no-actors
python Tools/simulated_twin_stream.py --rate 50 --seed 20260831 --no-actors
python Tools/simulated_twin_stream.py --rate 100 --seed 20260831 --no-actors
```

Only one of these commands should run at a time.

## Exact procedure for each run

1. Start the Python command for the chosen rate.
2. In Unity, enter Play mode.
3. Click **Drive** once.
4. Do not click other controls during the run.
5. Unity automatically performs a 30-second warm-up.
6. Unity then automatically records exactly 300 seconds.
7. Wait for the Console message saying the three Experiment 2 CSV files were
   saved.
8. Exit Play mode and stop the Python process.
9. Repeat the same rate until it has three valid runs.
10. Continue with the next rate.

The complete study is 15 valid runs: five rates multiplied by three
repetitions. The measured part of each run is five minutes, plus the 30-second
warm-up.

## Output files

Files are saved automatically in:

```text
ExperimentResults/experiment2/
```

Each run produces:

- `experiment2_030hz_run_001_snapshots.csv`: one row per accepted snapshot.
- `experiment2_030hz_run_001_frames.csv`: one row per rendered Unity frame.
- `experiment2_030hz_run_001_summary.csv`: run-level statistics and validity.

The number increases automatically. Existing results are never overwritten.

## What happens if the car crashes or the stream stops early?

The recorder immediately saves the partial CSV files and sets
`valid_for_analysis=false`. Its `completion_reason` identifies the cause, for
example `collision_detected`, `source_data_timeout`, or
`source_stopped_before_window_completed`.

Keep the partial files as an audit trail, but do not count them as one of the
three repetitions. Correct the cause if necessary and repeat that rate. The
analysis script automatically excludes invalid runs.

The formal `--no-actors` command should prevent a scenario collision. If a
collision still occurs on a clear road, treat that as a software defect rather
than ordinary experimental loss and investigate it before continuing.

## Generate the tables and figures

After completing the runs:

```bash
MPLBACKEND=Agg python Tools/analyze_experiment2.py --strict
```

The analysis is written to:

```text
ExperimentResults/experiment2/analysis_publication/
```

It contains:

- A valid-run summary CSV.
- A per-rate mean and standard-deviation CSV using runs as the independent units.
- A pipeline-accounting table based on unique source sequences. Raw transport
  counters remain in the audit because their measurement-window boundaries can
  differ by one callback.
- A run-level latency figure that reports median, P95, P99, and maxima without
  treating callbacks as independent repetitions.
- A displayed-state-age figure with individual-run and pooled descriptive ECDFs.
- A Unity frame-time figure with every run shown.
- Motion-continuity diagnostics (held-frame percentage and displayed speed),
  explicitly not presented as a perceptual smoothness score.
- An excluded-run CSV documenting incomplete, crashed, or invalid runs.

New publication outputs are written to `analysis_publication`. The earlier
`analysis` folder is retained unchanged for provenance.

## Interpretation limits

- The independent sample size is three runs per rate. Frames and callbacks
  within a run are correlated observations, not additional repetitions.
- The sharp frame-time change between 2 and 10 Hz and rare 0.5-1 second local
  latency maxima are reported as observations. Their causes are not assigned
  without profiler or operating-system evidence.
- Update rate is only one load factor. The clear-road protocol does not test
  scaling by actor count, payload size, scene complexity, or multiple clients.

The `--strict` option refuses to call the experiment complete unless every rate
has three valid runs. Without `--strict`, the script can create preliminary
figures from the valid runs collected so far.

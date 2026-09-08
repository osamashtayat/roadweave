# RoadWeave Experiment 3 — Missing and Delayed Data

## Purpose

This experiment measures how RoadWeave behaves when its 30 Hz simulated live
source loses, delays, duplicates, reorders, or omits data. The autonomous-world
content is disabled so the only independent variable is data transport quality.

The publisher and Unity run on the same Mac. This is a controlled local
fault-response experiment: it isolates the deliberately injected conditions,
but it is not validation over Ethernet, Wi-Fi, cellular, or a wide-area
network. Packet loss is independent Bernoulli loss, delay is fixed, and the
disconnection window is scheduled. Real networks can add burst loss, variable
jitter, bandwidth limits, clock differences, and reconnection behavior that
are outside this protocol.

The Unity recorder starts only when the Python command contains a unique
`--run-id`. Ordinary RoadWeave demonstrations therefore do not create
Experiment 3 files.

## What is automatic

After **Drive** is pressed:

1. Python and Unity run a 30-second warm-up.
2. Unity records for 300 seconds (or the duration supplied to Python).
3. Unity automatically saves snapshot, frame, event, and summary CSV files in
   `ExperimentResults/experiment3/unity`.
4. Python continuously saves independent ground truth in
   `ExperimentResults/experiment3/python`.
5. Stop Python with Control-C after Unity reports that the run was saved. Python
   then writes its source-side summary CSV.

Do not pause Unity or interact with the vehicle during a measured run. Use a
new run ID if a run is interrupted; existing evidence is never overwritten.

## Common preparation for every run

From the RoadWeave project directory:

```bash
cd /Users/asus/Desktop/roadweave
source ML/.venv/bin/activate
```

For every command below:

1. Start the Python command.
2. In Unity, press Play.
3. Press Drive once.
4. Wait for `RoadWeave Experiment 3 CSV files saved` in the Unity Console.
5. Stop Play mode.
6. Return to Terminal and press Control-C once.
7. Confirm that both `unity` and `python` folders received files with the same
   run ID.

## Repetition seeds

Use matched seeds so every fault level sees the same road trajectory:

| Repetition | World seed | Fault seed | Run suffix |
|---|---:|---:|---|
| 1 | 301 | 901 | `r01` |
| 2 | 302 | 902 | `r02` |
| 3 | 303 | 903 | `r03` |

Replace the seed values and suffix in each example for repetitions 2 and 3.

## A. Baseline

Run once per repetition. The same baseline is reused for the loss and delay
graphs.

```bash
python Tools/simulated_twin_stream.py --rate 30 --no-actors --controller rule --seed 301 --fault-seed 901 --run-id baseline_r01 --profile-id baseline
```

## B. Packet loss

Run 5%, 10%, and 20% loss for each repetition. The baseline supplies 0%.

```bash
python Tools/simulated_twin_stream.py --rate 30 --no-actors --controller rule --seed 301 --fault-seed 901 --run-id loss05_r01 --profile-id loss05 --packet-loss-percent 5
python Tools/simulated_twin_stream.py --rate 30 --no-actors --controller rule --seed 301 --fault-seed 901 --run-id loss10_r01 --profile-id loss10 --packet-loss-percent 10
python Tools/simulated_twin_stream.py --rate 30 --no-actors --controller rule --seed 301 --fault-seed 901 --run-id loss20_r01 --profile-id loss20 --packet-loss-percent 20
```

These runs answer whether random loss makes the displayed state old or stale.
At 30 Hz, even 20% independent loss may produce **zero stale time**, because
RoadWeave becomes stale only after one continuous second without an accepted
snapshot. That is a valid and useful result, not a failed experiment.

## C. Network delay

Run 50, 100, 250, and 500 ms for every repetition. The baseline supplies 0 ms.
Delay uses a delivery queue; it never freezes the simulated world.

```bash
python Tools/simulated_twin_stream.py --rate 30 --no-actors --controller rule --seed 301 --fault-seed 901 --run-id delay050_r01 --profile-id delay050 --delay-ms 50
python Tools/simulated_twin_stream.py --rate 30 --no-actors --controller rule --seed 301 --fault-seed 901 --run-id delay100_r01 --profile-id delay100 --delay-ms 100
python Tools/simulated_twin_stream.py --rate 30 --no-actors --controller rule --seed 301 --fault-seed 901 --run-id delay250_r01 --profile-id delay250 --delay-ms 250
python Tools/simulated_twin_stream.py --rate 30 --no-actors --controller rule --seed 301 --fault-seed 901 --run-id delay500_r01 --profile-id delay500 --delay-ms 500
```

## D. Temporary disconnection

For a useful recovery curve, test 0.5, 1, 2, and 5 seconds. Every outage begins
120 seconds after the measurement window starts.

```bash
python Tools/simulated_twin_stream.py --rate 30 --no-actors --controller rule --seed 301 --fault-seed 901 --run-id disconnect05_r01 --profile-id disconnect05 --disconnect-at-measurement-s 120 --disconnect-duration-s 0.5
python Tools/simulated_twin_stream.py --rate 30 --no-actors --controller rule --seed 301 --fault-seed 901 --run-id disconnect1_r01 --profile-id disconnect1 --disconnect-at-measurement-s 120 --disconnect-duration-s 1
python Tools/simulated_twin_stream.py --rate 30 --no-actors --controller rule --seed 301 --fault-seed 901 --run-id disconnect2_r01 --profile-id disconnect2 --disconnect-at-measurement-s 120 --disconnect-duration-s 2
python Tools/simulated_twin_stream.py --rate 30 --no-actors --controller rule --seed 301 --fault-seed 901 --run-id disconnect5_r01 --profile-id disconnect5 --disconnect-at-measurement-s 120 --disconnect-duration-s 5
```

## E. Contract and ordering checks

These are shorter 60-second conformance tests. Run each at least once; three
repetitions are preferred if the results will be reported statistically.

```bash
python Tools/simulated_twin_stream.py --rate 30 --no-actors --seed 301 --fault-seed 901 --run-id duplicate_r01 --profile-id duplicate --duplicate-percent 5 --experiment-duration 60
python Tools/simulated_twin_stream.py --rate 30 --no-actors --seed 301 --fault-seed 901 --run-id outoforder_r01 --profile-id outoforder --out-of-order-percent 5 --experiment-duration 60
python Tools/simulated_twin_stream.py --rate 30 --no-actors --seed 301 --fault-seed 901 --run-id missingvehicle_r01 --profile-id missingvehicle --missing-vehicle-percent 5 --experiment-duration 60
python Tools/simulated_twin_stream.py --rate 30 --no-actors --seed 301 --fault-seed 901 --run-id invalidvehicle_r01 --profile-id invalidvehicle --invalid-vehicle-percent 5 --experiment-duration 60
python Tools/simulated_twin_stream.py --rate 30 --no-actors --seed 301 --fault-seed 901 --run-id missingactors_r01 --profile-id missingactors --missing-actors-percent 5 --experiment-duration 60
```

Expected policy:

- Duplicate and older sequence numbers are rejected and counted.
- A missing required vehicle group is rejected and the last accepted state is held.
- A present vehicle group marked `Invalid`, with overall snapshot `Partial`, is
  accepted because the ego pose remains usable.
- An omitted actor list currently becomes an empty list. Record this honestly as
  “no actor observations supplied”; it does not prove that the road is empty.
- After one second without an accepted streaming snapshot, state freshness is
  `Stale`. A newly accepted snapshot returns it to `Fresh`.

The source summary records separate counts for missing-vehicle, invalid-vehicle,
and missing-actors injections in newly collected runs. Older retained runs can
still be audited from Unity contract rejections and accepted snapshot validity.

## Produce publication analyses

After all main runs are complete:

```bash
cd /Users/asus/Desktop/roadweave
source ML/.venv/bin/activate
python Tools/analyze_experiment3.py --expected-runs 3 --strict
```

The analysis creates:

- A packet-loss figure pairing achieved injected loss with run-level P95 state age.
- A disconnection-impact figure pairing omitted updates with the maximum
  one-frame recovery jump. First-receipt time is retained in the run table but
  is not treated as smooth visual recovery.
- A delay figure that separates the configured 160 ms presentation buffer from
  the additional injected-delay contribution using seed-matched baselines.
- `experiment3_run_level_metrics.csv`: one analysis-ready row per valid run.
- `experiment3_conformance_audit.csv`: expected and observed contract outcomes.
- `experiment3_delay_error_decomposition.csv`: presentation and injected-delay
  components for every delay run.
- `experiment3_excluded_runs.csv`: interrupted or unmatched runs that were excluded.
- `experiment3_run_counts.csv`: repetition audit by profile.

New outputs are written to `ExperimentResults/experiment3/analysis_publication`.
The earlier `analysis` folder remains unchanged for provenance.

The position error compares the displayed Unity vehicle at each frame with the
Python ground-truth position at the same UTC time. It therefore includes the
local transport/scheduling delay plus RoadWeave's configured 160 ms streaming
presentation buffer. At the 13.9 m/s test speed, the buffer predicts 2.224 m of
lag before injected delay, explaining almost all of the observed zero-delay
baseline. The metric is temporal display lag for this trajectory, not
localization accuracy.

The unit of replication is the run. The main loss, delay, and disconnection
conditions have three runs each. The contract/ordering profiles currently have
one run each and must be described as conformance demonstrations rather than
statistical evidence.

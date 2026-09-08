# RoadWeave Experiment 1: source integration baseline

## Purpose

Experiment 1 checks whether two source adapters can publish the canonical
`TwinSnapshot` contract into the same RoadWeave state and presentation path.
It is an integration baseline, not a controlled comparison of source quality,
throughput, or autonomous-driving performance.

The evaluated sources are:

- nuScenes replay, sampled by the Unity replay adapter on rendered frames;
- the local simulated UDP stream, configured at 30 Hz.

The replay rows are presentation samples created by the adapter. They are not
independent nuScenes sensor messages and must not be described as native sensor
frequency or network throughput.

## Scope and controls

- The publisher and Unity run on the same Mac. Timing therefore measures local
  process, serialization, transport, validation, and Unity scheduling overhead.
- Each source has 20 independent 15-second runs in the retained dataset.
- The sources retain their native experiment workloads. Their update rates and
  actor counts are not matched, so cross-source runtime differences are
  descriptive and cannot be assigned to the adapter alone.
- The experiment validates the shared contract and state path. It does not by
  itself establish that an arbitrary third-party model or source is easy to
  replace. That claim requires a separate adapter implementation study with
  recorded code changes, reused components, and integration effort.

## Recorded outcomes

For every run, record:

- completion reason and errors/warnings;
- adapter-ready and drive-to-first-snapshot timing;
- accepted snapshot count and achieved update rate;
- inter-arrival timing and sequence gaps;
- snapshot/frame freshness;
- actor count and Unity frame-time summaries.

The unit of replication is the run. Snapshot and frame samples within a run are
correlated and are not independent experimental repetitions.

## Reproduce the analysis

From the project directory:

```bash
MPLBACKEND=Agg python Tools/analyze_experiment1.py --strict
```

The script writes new derived outputs to
`ExperimentResults/experiment1/analysis_publication`. Existing raw CSV files
and the earlier analysis folder are not modified.

## Claim boundary

Supported: both tested adapters completed the retained short local runs and
published usable canonical snapshots through the same state/presentation path.

Not supported: one source is faster or better than the other; replay rate is a
native nuScenes sensing rate; arbitrary sources are plug-and-play; the system
has been validated over a physical network; or 15-second runs establish
long-duration stability.

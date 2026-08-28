# RoadWeave shared-label and cross-domain ML implementation

## Outcome

RoadWeave ML schema v2 removes the largest semantic mismatch between nuScenes
and K-Risk. Both adapters now produce the same observation, use the same
physical future-label functions, retain the underlying collection domain, and
are evaluated with group, source, and domain holdouts. Runtime inference adds
an out-of-distribution (OOD) guard before the existing deterministic safety
supervisor.

This improves cross-source transfer substantially, but does not eliminate
dataset shift. The system must still be described as a simulation/research
prototype rather than a real-vehicle controller.

## Exact data contract

Each row contains:

- `source`: `nuscenes` or `krisk`;
- `domain`: nuScenes location or K-Risk origin (`highd`, `ind`, `round`,
  `expresswaya`, `freewayb`);
- group ID and timestamp;
- `label_origin=observed_future`;
- one second of history and one second of label horizon;
- canonical 2 Hz sampling metadata;
- ego-relative SI physical summaries from `ML/src/features.py`.

High-frequency K-Risk and runtime histories are reduced to the same 2 Hz grid
as nuScenes before summaries are calculated. The model sees only past/current
states. Future states are used inside the converter to create a target and are
discarded before training.

## Shared targets

`ML/src/common_labels.py` is the only definition used by both raw adapters.

Risk uses measured next-second current-path evidence:

- collision/physical overlap;
- minimum time to collision;
- required deceleration from closing speed and gap;
- maximum absolute ego acceleration;
- pedestrian clearance.

Policy uses measured next-second ego behavior:

- curvature-normalized lateral displacement for left/right changes;
- speed change for accelerate/decelerate;
- otherwise keep.

The K-Risk folder severity, `total_risk`, behavior flags, and GPT response are
not input features or core targets. The 372 GPT recommendations are preserved
in `policy_krisk_recommended.parquet` only for audit and future comparison.

## Prepared data

- nuScenes: 15,908 risk rows and 15,908 policy rows from 834 usable scenes;
- K-Risk: 24,010 risk rows and 24,010 policy rows;
- combined: 39,918 rows per task and 24,844 independent groups;
- excluded K-Risk clips: 2,057 (7.9%) because the ego trajectory did not have
  a complete measured second on both sides of the observation. No missing
  future was padded or fabricated.

K-Risk policy distribution after physical relabeling:

- KEEP 17,213;
- ACCELERATE 2,990;
- DECELERATE 2,145;
- CHANGE_LEFT 811;
- CHANGE_RIGHT 851.

## Training and evaluation

`ML/src/train_model.py` uses source/class-balanced sample weights and strictly
group-disjoint train/validation/test splits. It additionally trains diagnostic
models for:

- leave-one-source-out (nuScenes versus K-Risk);
- leave-one-domain-out (highD, inD, rounD, CitySim locations, and each
  nuScenes location);
- a vehicle-overlap subset with no pedestrian in the observation history.

The risk model retained all 210 physical summaries because the 87-feature
source-overlap experiment materially reduced both ordinary and transfer risk
performance. The policy model uses the 87-feature transfer profile, which
removes pedestrian-only fields and sampling-sensitive binary transition
summaries.

Selected risk results:

- test accuracy 0.8423; macro-F1 0.7580;
- class F1: LOW 0.923, MODERATE 0.639, HIGH 0.703, EXTREME 0.767;
- nuScenes → K-Risk macro-F1 0.5011 (previously 0.0758);
- K-Risk → nuScenes macro-F1 0.2882 (previously 0.0746);
- K-Risk → nuScenes vehicle-overlap subset macro-F1 0.4411.

Selected policy results:

- test accuracy 0.7965; macro-F1 0.5609;
- class F1: KEEP 0.880, ACCELERATE 0.686, DECELERATE 0.635,
  CHANGE_LEFT 0.267, CHANGE_RIGHT 0.336;
- nuScenes → K-Risk macro-F1 0.3766 (previously 0.1426);
- K-Risk → nuScenes macro-F1 0.4876 (previously 0.2023);
- K-Risk → nuScenes vehicle-overlap subset macro-F1 0.5063.

## Audits and rejected alternatives

`ML/src/audit_domains.py` trains a dataset/domain classifier and measures
missingness plus normalized Wasserstein feature drift. The full source
classifier remains around 0.99 macro-F1. This proves that shared labels and
sampling reduce task mismatch but do not erase covariate shift.

Two alternatives were evaluated and rejected for deployment:

1. The compact transfer profile for risk fell from 0.7580 to 0.5179 test
   macro-F1 and from 0.5011 to 0.3947 on nuScenes → K-Risk.
2. A hierarchical lateral/longitudinal policy fell from 0.5609 to 0.5244 test
   macro-F1 and to about 0.36 on both source holdouts.

The experimental hierarchical classifier remains reproducible in
`ML/src/hierarchical_policy.py`, but `flat` is the default and selected model.
RoadWeave procedural data was deliberately used for closed-loop evaluation,
not supervised training, because copying the rule controller's actions would
create circular evidence.

## Runtime integration and OOD policy

Every v2 artifact contains per-feature training quantiles, missing rates, and
an OOD threshold. `ML/src/model_support.py` evaluates each observation against
that envelope. `ML/src/online_policy.py` and
`ML/src/testlab_inference.py` request cautious deceleration when the envelope
is exceeded and expose `outOfDistribution` plus `oodScore`. A physical
emergency still has higher priority, and OOD never authorizes a lane change or
direct transform write.

The Test Lab service self-test passed with the selected artifacts. Its normal
observation produced OOD false with score 0.0115; its pedestrian emergency was
still stopped by the deterministic envelope.

## Closed-loop procedural benchmark

Six unseen seeds were run for 120 seconds per controller:

| Metric | Rule | Selected ML |
|---|---:|---:|
| Mean route progress | 861.14 m | 940.53 m |
| Minimum clearance | -3.47 m | 1.14 m |
| Minimum TTC | 0.009 s | 0.493 s |
| Mean emergency stops | 2.50 | 2.83 |
| Mean deadlock time | 2.92 s | 1.20 s |
| Mean absolute acceleration | 1.216 m/s² | 1.099 m/s² |
| Mean absolute yaw | 2.228 deg/s | 2.185 deg/s |

The negative rule clearance indicates overlap in at least one benchmark run.
The selected ML pair retained positive clearance across these seeds. This is
useful prototype evidence, but the sample is too small for a safety claim.

## Files created or materially changed

- `ML/src/common_labels.py`: shared physical target definitions;
- `ML/src/nuscenes_lite.py`: focused metadata/CAN reader;
- `ML/src/features.py`: canonical 2 Hz history resampling;
- `ML/src/build_nuscenes.py`: domain metadata and future physical risk/policy;
- `ML/src/build_krisk.py`: actual-future targets and audit-only GPT output;
- `ML/src/model_support.py`: schema v2, transfer profile, OOD profile/guard;
- `ML/src/train_model.py`: source/domain weighting and holdout evaluation;
- `ML/src/audit_domains.py`: domain classifier and drift report;
- `ML/src/hierarchical_policy.py`: rejected but reproducible experiment;
- `ML/src/online_policy.py`: live-stream OOD supervision;
- `ML/src/testlab_inference.py`: Test Lab OOD supervision;
- `Assets/Scripts/TestLab/TestLabMlDecisionBridge.cs`: OOD response fields;
- tests, `ML/README.md`, and `ML/MODEL_CARD.md`.

Generated Parquet data, joblib artifacts, predictions, and JSON reports remain
under the gitignored `ML/data/processed`, `ML/models`, and `ML/reports`
directories. They are reproducible from the commands in `ML/README.md`.

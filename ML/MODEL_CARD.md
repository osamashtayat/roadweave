# RoadWeave baseline model card

## Status

This is a research prototype and demonstration baseline, not a controller for
a physical vehicle. The deterministic collision supervisor must remain active.

Training completed on 2026-08-25 with Python 3.9.6 and scikit-learn 1.6.1. All
splits are group-disjoint: rows from the same nuScenes scene or K-Risk event
cannot appear on both sides of a train/validation/test boundary.

## Risk model

- Model: histogram gradient-boosting classifier
- Rows: 26,376
- Independent groups: 19,330
- Classes: LOW 7,746; MODERATE 13,229; HIGH 4,365; EXTREME 1,036
- Test accuracy: 0.8985
- Test balanced accuracy: 0.8110
- Test macro-F1: 0.8223
- Per-class test F1: LOW 0.9987; MODERATE 0.9080; HIGH 0.7485; EXTREME 0.6339

The score is useful as an initial pipeline check, but it is optimistic evidence
of real-world generalization. LOW examples come from nuScenes while all other
risk classes come from K-Risk, so dataset/domain differences can act as a
shortcut. A publishable evaluation should add low and hazardous examples from
the same domains or use leave-one-dataset-out validation.

## Policy model

- Model: histogram gradient-boosting classifier
- Rows: 15,045
- Independent groups: 1,206
- Classes: KEEP 13,571; ACCELERATE 755; DECELERATE 707; CHANGE_LEFT 5;
  CHANGE_RIGHT 7
- Test accuracy: 0.9300
- Test balanced accuracy: 0.4385
- Test macro-F1: 0.4464
- Per-class test F1: KEEP 0.9623; ACCELERATE 0.6403; DECELERATE 0.6294;
  CHANGE_LEFT 0.0000; CHANGE_RIGHT 0.0000

Accuracy is high because KEEP dominates. Macro-F1 and per-class results are the
honest measures: this baseline can learn routine longitudinal choices, but the
available labels are not sufficient to claim learned lane-changing ability.
The saved model is therefore experimental and rule mode remains RoadWeave's
default presentation mode.

## Online safety behavior

The learned model requests a maneuver; it never writes a Unity transform.
`online_policy.py` applies these constraints before a request reaches the
simulated vehicle:

- emergency stopping envelope for pedestrians and dangerously close front actors;
- rejection of a lane change when its front or rear clearance is unsafe;
- three consecutive model votes before accepting a lane change;
- conservative deceleration for low-confidence, high-risk, or extreme-risk requests;
- continued 30 Hz safety/physics updates around the 5 Hz learned decisions.

Both the standard rule-controller self-test and saved-model live-world
self-test pass without a geometric overlap.

## Required next research work

1. Obtain substantially more safe recommended lane-change labels; do not simply
   duplicate the current 12 examples and call the imbalance solved.
2. Add verified inD, NGSIM, and rounD adapters with explicit coordinate/unit
   tests before using those K-Risk folders.
3. Evaluate cross-domain generalization and calibration, not only random
   group-separated performance.
4. Compare this baseline against the existing deterministic controller on
   collision rate, minimum TTC, route progress, intervention count, comfort,
   and scenario completion.
5. Keep the ML controller in Test Lab/simulation until those safety and
   generalization checks are complete.


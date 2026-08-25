# RoadWeave baseline model card

## Status

This is a research prototype and demonstration baseline, not a controller for
a physical vehicle. The deterministic collision supervisor must remain active.

Training completed on 2026-08-25 with Python 3.9.6 and scikit-learn 1.6.1. All
splits are group-disjoint: rows from the same nuScenes scene or K-Risk event
cannot appear on both sides of a train/validation/test boundary.

## Risk model

- Model: histogram gradient-boosting classifier
- Rows: 41,668
- Independent groups: 26,901
- Classes: LOW 7,746; MODERATE 21,774; HIGH 8,627; EXTREME 3,521
- Test accuracy: 0.9181
- Test macro-F1: 0.9083
- Leave-one-source-out macro-F1: 0.0758 (train nuScenes → test K-Risk),
  0.0746 (train K-Risk → test nuScenes)

nuScenes risk is now graded into all four levels from physical severity (ego
acceleration, front TTC, pedestrian gap, and overlap) instead of only LOW, and
the K-Risk conversion now includes inD and rounD alongside highD and CitySim.
This removes the "LOW always means nuScenes" shortcut for the within-domain
task: random group-separated macro-F1 rose from 0.82 to 0.91.

The leave-one-source-out numbers are the honest measure of generalization:
they stay near 0.08 because the two datasets are genuinely different domains
(urban sensor-rich driving vs. German/Chinese trajectory segments). The model
still cannot transfer across that gap, so treat every within-domain score as
optimistic evidence.

## Policy model

- Model: histogram gradient-boosting classifier
- Rows: 13,447
- Classes: KEEP 11,889; ACCELERATE 658; DECELERATE 632; CHANGE_LEFT 150;
  CHANGE_RIGHT 118
- Test macro-F1: 0.7582
- Per-class test F1: KEEP 0.956; ACCELERATE 0.519; DECELERATE 0.597;
  CHANGE_LEFT 0.862; CHANGE_RIGHT 0.857

Lane changes are now learnable. Two fixes supplied the previously missing
signal: highD's native `yaw_left`/`yaw_right` ground-truth flags are used as
policy labels (150+118 examples instead of 12), and nuScenes lane changes are
detected by integrating lateral motion per frame heading (curvature-invariant)
over a 3 s window. Lane-change F1 went from 0.0 to ~0.86.

## Online safety behavior

The learned model requests a maneuver; it never writes a transform.
`online_policy.py` applies these constraints before a request reaches the
simulated vehicle:

- emergency stopping envelope for pedestrians and dangerously close front actors;
- a deterministic deadlock escape: a full stop behind a stationary obstacle with
  a clear adjacent lane changes lanes instead of waiting forever (this mirrors
  the rule controller, because the learned policy alone could not reliably
  initiate an overtake);
- rejection of a lane change when its front or rear clearance is unsafe;
- three consecutive model votes before accepting a lane change;
- conservative deceleration for low-confidence, high-risk, or extreme-risk requests;
- continued 30 Hz safety/physics updates around the 5 Hz learned decisions.

## Required next research work

1. Evaluate cross-domain generalization and calibration; the ~0.08
   leave-one-source-out scores show the domain gap is not closed by labeling
   alone.
2. Add an NGSIM adapter with an explicit feet→metres unit conversion. Its
   K-Risk release retains native feet despite the `info.txt` labelling them as
   metres, so it needs a separately verified unit test before use.
3. Compare this baseline against the deterministic controller (see
   `Tools/benchmark_controllers.py`) on minimum TTC, clearance, intervention
   count, deadlock time, route progress, and comfort.
4. Keep the ML controller in Test Lab/simulation until those checks are complete.

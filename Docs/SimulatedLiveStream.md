# RoadWeave Procedural Live-Stream Simulator

## Purpose

`Tools/simulated_twin_stream.py` is a temporary software data source for the
RoadWeave Live Twin. It proves that Unity can consume a real-time stream without
depending on nuScenes. A future sensor gateway can replace this process by
publishing the same canonical `TwinSnapshot` JSON contract.

The simulator is deterministic when a seed is supplied and procedural when no
seed is supplied. It is a research/demo simulator, not a certified autonomous
driving stack.

## Data flow

```text
Unlimited procedural route and scenario planner
        -> simulated road actors
        -> simulated sensor observations
        -> autonomous behavior controller
        -> ego/actor world update
        -> canonical TwinSnapshot JSON
        -> UDP 5055
        -> SimulatedJsonUdpTransport
        -> SimulatedStreamTwinSource
        -> DigitalTwinStateManager
        -> Unity vehicle, road, actors and dashboard
```

Unity sends `RESET_START`, `RESET_PAUSE`, `START`, and `PAUSE` commands to the
Python process on UDP port 5056.

Ego telemetry and surrounding actors are published at 30 Hz. Route geometry is
published at 1 Hz and retained by Unity between route messages. Route points are
automatically thinned if necessary so every UDP datagram remains below the safe
8,000-byte transport budget used on macOS.

## Unlimited random world

Every Unity reset creates a new seed, a different road shape, and a continuously
refilled queue of encounters. The stream does not finish after the initial
look-ahead queue. Passed actors are retired and new slow cars, stopped cars,
trucks, pedestrians, and supporting traffic are generated farther ahead for as
long as Unity keeps the session running. Their order, spacing, lane, speed, and
supporting traffic vary. Every generated actor has a stable, unique ID for the
lifetime of that source session.

The road is represented internally by distance along a procedural route. Its
world-space heading changes smoothly left and right. Sensors make decisions in
lane-relative coordinates, while the canonical snapshot publishes curved Unity
positions, rotations, velocities, and an optional look-ahead `route` section.
Unity renders that route before the ego reaches it instead of extending one
straight road forever.

The planner only creates the world. It does not command the ego vehicle. The
controller receives only the output of `SimulatedSensorSuite`.

## Simulated sensors

The sensor suite reports:

- closest actor ahead in the current lane;
- left-front and left-rear traffic;
- right-front and right-rear traffic;
- bumper gap and relative speed;
- time to collision;
- a pedestrian-crossing hazard.

## Driving behavior

The deterministic controller uses these states:

- `CRUISE`
- `FOLLOWING`
- `BRAKING_FOR_PEDESTRIAN`
- `CHANGE_LEFT`
- `PASSING`
- `CHANGE_RIGHT`
- `EMERGENCY_STOP`

It accelerates to approximately 50 km/h on a clear road, follows when the other
lane is blocked, overtakes only when front/rear clearance is available, returns
to the right after passing, and stops for crossing pedestrians. A final
non-penetration guard prevents the ego and surrounding traffic from occupying
the same space if a discrete update reaches a safety boundary.

## Run commands

Normal random run:

```bash
cd "/Users/asus/Desktop/roadweave"
python3 -B Tools/simulated_twin_stream.py
```

Keep a larger or smaller future encounter queue (the stream remains unlimited):

```bash
python3 -B Tools/simulated_twin_stream.py --encounters 10
```

Reproduce a particular run for a presentation or bug report:

```bash
python3 -B Tools/simulated_twin_stream.py --seed 20260824
```

Clear-road baseline:

```bash
python3 -B Tools/simulated_twin_stream.py --no-actors
```

Change stream rate:

```bash
python3 -B Tools/simulated_twin_stream.py --rate 30
```

Run the built-in deterministic safety check:

```bash
python3 -B Tools/simulated_twin_stream.py --self-test
```

The console prints the generated seed and initial look-ahead queue at reset.
During driving it prints speed, lane, controller behavior, and the nearest front
detection. New encounters continue beyond that initial printed queue. Save the
seed whenever a run needs to be reproduced.

## Unity presentation behavior

`EgoVehicleReplayView` keeps the snapshot authoritative but smooths simulated
and live-source presentation with a short timestamped pose buffer. It renders
about 100 ms behind the newest packet and interpolates only between two received
poses; it does not invent forward motion. Paused or stale data therefore holds
or disappears instead of drifting. nuScenes replay continues to use its existing
replay interpolation unchanged.

`ReplayActorManager` grounds cars, trucks, buses, pedestrians, bicycles,
motorcycles, barriers, and generic road actors to the ego road plane. Imported
visuals are centered on a stable actor root before the root is placed at half
the actor height.

`StreamingRouteProvider` prefers the optional route-ahead section published by
the simulator and rebuilds the road only when its route revision changes. An
older adapter that does not publish a route remains compatible: the provider
falls back to building a breadcrumb route from accepted ego positions.

## Future sensor-gateway transition

The future gateway replaces the planner, sensor simulator, and Python driving
world. It must synchronize its real sensors, convert coordinates and units,
assign validity/freshness, track surrounding objects with stable IDs, and emit
the same `TwinSnapshot` contract. A map or navigation service may populate the
optional route section; if it is unavailable, Unity retains its ego-breadcrumb
fallback. Unity presentation code remains unchanged.

For a real vehicle, RoadWeave Live Twin visualizes the vehicle's authoritative
state; it must not send simulated throttle, brake, or steering commands to the
physical car. Experimental interventions remain isolated in Test Lab.

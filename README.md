# RoadWeave

## Overview

RoadWeave is a digital twin research prototype for autonomous vehicles. It connects vehicle and environmental data to a Unity simulation where users can observe the vehicle, surrounding traffic, pedestrians, weather, sensors, and driving decisions.

RoadWeave is designed to work with different data sources. It currently supports nuScenes replay data and a simulated live stream. In the future, these sources can be replaced by a sensor gateway connected to a real vehicle.

## Purpose

The purpose of RoadWeave is to study how a digital twin can represent an autonomous vehicle and evaluate its behaviour under normal and risky conditions.

RoadWeave can be used to:

1. Visualize a vehicle and its environment in real time

2. Display vehicle information such as speed, temperature, gear, position, and wheel state

3. Create driving scenarios involving pedestrians, cars, and trucks

4. Test the vehicle in rain, snow, fog, and dry weather

5. Simulate sensor faults and reduced sensor reliability

6. Compare rule controlled driving with hybrid machine learning driving

7. Measure safety, performance, communication latency, and system reliability

## Main Modes

### Live Twin

Live Twin represents information received from an external data source.

The current source can be either nuScenes replay data or the Python simulated live stream. The vehicle, roads, surrounding actors, and information panels are updated from the received data.

A future sensor gateway can replace these sources while keeping the Unity application largely unchanged.

### Test Lab

Test Lab is a separate simulation environment for controlled experiments.

It allows users to add pedestrians, stopped cars, slow cars, trucks, weather conditions, sensor faults, and risky driver behaviour. The vehicle then reacts using either the rule controller or the hybrid machine learning controller.

Test Lab does not send commands to a real vehicle. It is intended for safe simulation and evaluation.

## Data Flow

RoadWeave uses the following data flow:

```text
Data Source
    ↓
Source Adapter
    ↓
TwinSnapshot
    ↓
DigitalTwinStateManager
    ↓
Unity Vehicle, Actors, Roads, Interface, and Experiments
```

Each source adapter converts its original data into a common format called `TwinSnapshot`.

Because Unity reads the same `TwinSnapshot` format, the data source can be replaced without rebuilding the entire Unity application.

## Supported Data Sources

RoadWeave currently supports:

1. nuScenes replay data

2. Python simulated live streaming data

3. Unity Test Lab scenarios

The planned future source is a sensor gateway that receives data from GPS, IMU, CAN bus, camera, LiDAR, and radar systems.

## Machine Learning Models

RoadWeave contains four main machine learning components.

### Risk Model

The risk model estimates the current driving risk.

It predicts one of four levels:

1. Low

2. Moderate

3. High

4. Extreme

The model uses information such as vehicle speed, acceleration, nearby actors, distance, and time to collision.

It was trained using processed data from nuScenes and the K Risk dataset.

### Driving Policy Model

The driving policy model recommends the next driving action.

It can recommend:

1. Keep the current behaviour

2. Accelerate

3. Decelerate

4. Change to the left lane

5. Change to the right lane

The model was trained using a common data format created from nuScenes and K Risk data.

### Weather Model

The weather model estimates an appropriate observed driving speed tendency for different weather conditions.

It supports:

1. Dry weather

2. Rain

3. Snow

4. Fog

The model was trained using information extracted from the Extreme Driving Conditions Dataset.

This model represents driving tendencies found in the data. It does not claim to calculate the perfect safe speed for every road or vehicle.

### Sensor Reliability Models

RoadWeave contains separate reliability models for:

1. Camera

2. LiDAR

3. Radar

These models estimate how reliable each sensor is under different conditions. They consider message age, dropout, confidence, continuity, disagreement, noise, weather, and calibration problems.

The current models were trained using generated virtual sensor scenarios. They provide a research baseline and will require further validation using real sensor data.

## Hybrid Safety Controller

RoadWeave does not allow the machine learning models to control every safety decision alone.

The hybrid controller combines machine learning predictions with deterministic safety rules. The models recommend risk levels and driving actions, while the safety supervisor can brake, stop, or reject an unsafe decision.

This design keeps machine learning involved while maintaining a final safety layer.

## Experiments

RoadWeave includes experiments for:

1. Comparing nuScenes replay with simulated streaming

2. Testing different update rates

3. Testing packet loss, delay, missing data, and disconnections

4. Comparing the rule controller with the hybrid machine learning controller

5. Evaluating risk and driving policy models

6. Evaluating sensor reliability and weather models

The generated results include CSV files, summaries, tables, and research graphs.

## Technology

RoadWeave uses:

1. Unity and C Sharp for visualization and simulation

2. Python for data processing, simulated streaming, training, and inference

3. scikit learn for machine learning

4. pandas and NumPy for dataset processing

5. JSON and UDP for communication between Python and Unity

6. nuScenes, K Risk, and Extreme Driving Conditions data

## Future Development

The next major step is replacing the simulated stream with a real sensor gateway.

The gateway will collect information from vehicle sensors and convert it into the same `TwinSnapshot` structure already used by RoadWeave.

Future work can also include real vehicle validation, improved clock synchronization, stronger security, additional sensor data, and a React dashboard for experiment results.

## Current Status

RoadWeave is a working research prototype. It demonstrates a source independent digital twin architecture, live vehicle visualization, controlled autonomous driving experiments, machine learning assisted decisions, sensor reliability estimation, and repeatable evaluation.

It is not currently a production autonomous driving system and should not be used to control a real vehicle without further validation and safety testing.

using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// NuScenes/replay-specific interpolation. This is intentionally outside the
/// canonical state store so another source never needs to understand ReplayPackage.
/// </summary>
public static class ReplaySnapshotSampler
{
    public const int SupportedSchemaVersion = 1;

    public static bool ValidatePackage(
        ReplayPackage package,
        out TwinCoordinateFrame coordinateFrame,
        out string error)
    {
        coordinateFrame = null;
        if (package == null)
        {
            error = "The replay JSON is empty.";
            return false;
        }

        if (package.schemaVersion != SupportedSchemaVersion)
        {
            error = $"Unsupported replay schemaVersion {package.schemaVersion}. " +
                    $"This build supports schemaVersion {SupportedSchemaVersion}.";
            return false;
        }

        if (package.egoFrames == null || package.egoFrames.Length == 0)
        {
            error = "The replay contains no ego frames.";
            return false;
        }

        if (package.durationSeconds <= 0f)
        {
            error = "The replay duration must be greater than zero seconds.";
            return false;
        }

        if (!IsFinite(package.durationSeconds) || !IsFinite(package.sampleIntervalSeconds))
        {
            error = "Replay duration and sample interval must be finite numbers.";
            return false;
        }

        if (!ValidateEgoFrames(package.egoFrames, out error) ||
            !ValidateVehicleFrames(package.vehicleFrames, out error) ||
            !ValidateWheelFrames(package.wheelFrames, out error) ||
            !ValidateActorTracks(package.actors, out error))
            return false;

        return TwinCoordinateFrameService.TryCreateCanonicalFrame(
            package.coordinateSystem,
            out coordinateFrame,
            out error);
    }

    private static bool ValidateEgoFrames(EgoReplayFrame[] frames, out string error)
    {
        float previousTime = float.NegativeInfinity;
        for (int index = 0; index < frames.Length; index++)
        {
            EgoReplayFrame frame = frames[index];
            if (frame == null || frame.position == null)
            {
                error = $"egoFrames[{index}] or its position is null.";
                return false;
            }
            if (!ValidateOrderedTime(frame.time, previousTime, $"egoFrames[{index}]", out error) ||
                !IsFinite(frame.position) || !IsFinite(frame.yawDegrees) ||
                !IsFinite(frame.speedMetersPerSecond) || !IsFinite(frame.longitudinalAcceleration))
            {
                if (error == null) error = $"egoFrames[{index}] contains a non-finite value.";
                return false;
            }
            previousTime = frame.time;
        }
        error = null;
        return true;
    }

    private static bool ValidateVehicleFrames(VehicleReplayFrame[] frames, out string error)
    {
        if (frames == null || frames.Length == 0) { error = null; return true; }
        float previousTime = float.NegativeInfinity;
        for (int index = 0; index < frames.Length; index++)
        {
            VehicleReplayFrame frame = frames[index];
            if (frame == null)
            {
                error = $"vehicleFrames[{index}] is null.";
                return false;
            }
            if (!ValidateOrderedTime(frame.time, previousTime, $"vehicleFrames[{index}]", out error) ||
                !IsFinite(frame.speedKilometersPerHour) || !IsFinite(frame.batteryPercent) ||
                !IsFinite(frame.availableDistanceKilometers) || !IsFinite(frame.throttlePercent) ||
                !IsFinite(frame.brake) || !IsFinite(frame.steeringDegrees) ||
                !IsFinite(frame.steeringSpeed) || !IsFinite(frame.yawRate))
            {
                if (error == null) error = $"vehicleFrames[{index}] contains a non-finite value.";
                return false;
            }
            previousTime = frame.time;
        }
        error = null;
        return true;
    }

    private static bool ValidateWheelFrames(WheelReplayFrame[] frames, out string error)
    {
        if (frames == null || frames.Length == 0) { error = null; return true; }
        float previousTime = float.NegativeInfinity;
        for (int index = 0; index < frames.Length; index++)
        {
            WheelReplayFrame frame = frames[index];
            if (frame == null)
            {
                error = $"wheelFrames[{index}] is null.";
                return false;
            }
            if (!ValidateOrderedTime(frame.time, previousTime, $"wheelFrames[{index}]", out error) ||
                !IsFinite(frame.frontLeftRpm) || !IsFinite(frame.frontRightRpm) ||
                !IsFinite(frame.rearLeftRpm) || !IsFinite(frame.rearRightRpm))
            {
                if (error == null) error = $"wheelFrames[{index}] contains a non-finite value.";
                return false;
            }
            previousTime = frame.time;
        }
        error = null;
        return true;
    }

    private static bool ValidateActorTracks(ActorReplayTrack[] tracks, out string error)
    {
        if (tracks == null || tracks.Length == 0) { error = null; return true; }
        HashSet<string> ids = new HashSet<string>(StringComparer.Ordinal);
        for (int trackIndex = 0; trackIndex < tracks.Length; trackIndex++)
        {
            ActorReplayTrack track = tracks[trackIndex];
            if (track == null || string.IsNullOrWhiteSpace(track.id) || !ids.Add(track.id))
            {
                error = $"actors[{trackIndex}] is null or has a missing/duplicate stable ID.";
                return false;
            }
            if (track.size == null || !IsFinite(track.size) ||
                track.size.x <= 0f || track.size.y <= 0f || track.size.z <= 0f)
            {
                error = $"actors[{trackIndex}] requires finite, positive dimensions.";
                return false;
            }
            if (track.frames == null || track.frames.Length == 0)
            {
                error = $"actors[{trackIndex}] has no observation frames.";
                return false;
            }

            float previousTime = float.NegativeInfinity;
            for (int frameIndex = 0; frameIndex < track.frames.Length; frameIndex++)
            {
                ActorReplayFrame frame = track.frames[frameIndex];
                string label = $"actors[{trackIndex}].frames[{frameIndex}]";
                if (frame == null || frame.position == null)
                {
                    error = $"{label} or its position is null.";
                    return false;
                }
                if (!ValidateOrderedTime(frame.time, previousTime, label, out error) ||
                    !IsFinite(frame.position) || !IsFinite(frame.yawDegrees))
                {
                    if (error == null) error = $"{label} contains a non-finite value.";
                    return false;
                }
                previousTime = frame.time;
            }
        }
        error = null;
        return true;
    }

    private static bool ValidateOrderedTime(float time, float previousTime, string label, out string error)
    {
        if (!IsFinite(time))
        {
            error = $"{label}.time is not finite.";
            return false;
        }
        if (time < previousTime)
        {
            error = $"{label}.time is out of order.";
            return false;
        }
        error = null;
        return true;
    }

    private static bool IsFinite(ReplayVector3 value) =>
        value != null && IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
    private static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

    public static TwinSnapshot Sample(
        ReplayPackage package,
        TwinCoordinateFrame coordinateFrame,
        float replayTime,
        long sequenceNumber,
        TwinSessionInfo session)
    {
        float time = Mathf.Clamp(replayTime, 0f, package.durationSeconds);
        TwinSnapshot snapshot = new TwinSnapshot
        {
            metadata = new TwinSnapshotMetadata
            {
                sequenceNumber = sequenceNumber,
                sourceTimestampSeconds = time,
                timelineDurationSeconds = package.durationSeconds,
                staleAfterSeconds = Mathf.Max(0.1f, package.sampleIntervalSeconds * 3f),
                validity = TwinDataValidity.Valid,
                freshness = TwinDataFreshness.Fresh,
                coordinateFrame = coordinateFrame,
                session = CopySession(session)
            }
        };

        SampleEgo(package.egoFrames, time, snapshot.ego);
        SampleVehicle(package.vehicleFrames, time, snapshot.ego, snapshot.vehicle);
        SampleWheels(package.wheelFrames, time, snapshot.wheels);
        snapshot.actors = SampleActors(
            package.actors,
            time,
            package.sampleIntervalSeconds,
            sequenceNumber);

        bool telemetryPartial = snapshot.vehicle.validity != TwinDataValidity.Valid ||
                                snapshot.wheels.validity != TwinDataValidity.Valid;
        if (telemetryPartial)
        {
            snapshot.metadata.validity = TwinDataValidity.Partial;
            snapshot.metadata.validityMessage =
                "Ego pose is valid; one or more optional telemetry groups are unavailable.";
        }

        return snapshot;
    }

    private static void SampleEgo(EgoReplayFrame[] frames, float time, TwinEgoState result)
    {
        if (frames == null || frames.Length == 0)
        {
            result.validity = TwinDataValidity.Invalid;
            return;
        }

        FindPair(frames.Length, time, index => frames[index].time, out int fromIndex, out int toIndex, out float t);
        EgoReplayFrame from = frames[fromIndex];
        EgoReplayFrame to = frames[toIndex];
        result.position = Vector3.Lerp(SafeVector(from.position), SafeVector(to.position), t);
        result.yawDegrees = Mathf.LerpAngle(from.yawDegrees, to.yawDegrees, t);
        result.speedMetersPerSecond = Mathf.Lerp(from.speedMetersPerSecond, to.speedMetersPerSecond, t);
        result.longitudinalAcceleration = Mathf.Lerp(from.longitudinalAcceleration, to.longitudinalAcceleration, t);
        result.validity = TwinDataValidity.Valid;
    }

    private static void SampleVehicle(
        VehicleReplayFrame[] frames,
        float time,
        TwinEgoState ego,
        TwinVehicleState result)
    {
        if (frames == null || frames.Length == 0)
        {
            result.speedKilometersPerHour = ego.speedMetersPerSecond * 3.6f;
            result.validity = TwinDataValidity.Partial;
            return;
        }

        FindPair(frames.Length, time, index => frames[index].time, out int fromIndex, out int toIndex, out float t);
        VehicleReplayFrame from = frames[fromIndex];
        VehicleReplayFrame to = frames[toIndex];
        result.speedKilometersPerHour = Mathf.Lerp(from.speedKilometersPerHour, to.speedKilometersPerHour, t);
        result.batteryPercent = Mathf.Lerp(from.batteryPercent, to.batteryPercent, t);
        result.availableDistanceKilometers = Mathf.Lerp(from.availableDistanceKilometers, to.availableDistanceKilometers, t);
        result.gearPosition = Nearest(from.gearPosition, to.gearPosition, t);
        result.throttlePercent = Mathf.Lerp(from.throttlePercent, to.throttlePercent, t);
        result.brake = Mathf.Lerp(from.brake, to.brake, t);
        result.brakeSwitch = Nearest(from.brakeSwitch, to.brakeSwitch, t);
        result.steeringDegrees = Mathf.Lerp(from.steeringDegrees, to.steeringDegrees, t);
        result.steeringSpeed = Mathf.Lerp(from.steeringSpeed, to.steeringSpeed, t);
        result.yawRate = Mathf.Lerp(from.yawRate, to.yawRate, t);
        result.leftSignal = Nearest(from.leftSignal, to.leftSignal, t);
        result.rightSignal = Nearest(from.rightSignal, to.rightSignal, t);
        result.temperatureIsValid = packageTemperatureUnavailable;
        result.validity = TwinDataValidity.Valid;
    }

    // Kept as a constant to make the absence explicit rather than fabricating a value.
    private const bool packageTemperatureUnavailable = false;

    private static void SampleWheels(WheelReplayFrame[] frames, float time, TwinWheelState result)
    {
        if (frames == null || frames.Length == 0)
        {
            result.validity = TwinDataValidity.Invalid;
            return;
        }

        FindPair(frames.Length, time, index => frames[index].time, out int fromIndex, out int toIndex, out float t);
        WheelReplayFrame from = frames[fromIndex];
        WheelReplayFrame to = frames[toIndex];
        result.frontLeftRpm = Mathf.Lerp(from.frontLeftRpm, to.frontLeftRpm, t);
        result.frontRightRpm = Mathf.Lerp(from.frontRightRpm, to.frontRightRpm, t);
        result.rearLeftRpm = Mathf.Lerp(from.rearLeftRpm, to.rearLeftRpm, t);
        result.rearRightRpm = Mathf.Lerp(from.rearRightRpm, to.rearRightRpm, t);
        result.validity = TwinDataValidity.Valid;
    }

    private static TwinActorState[] SampleActors(
        ActorReplayTrack[] tracks,
        float time,
        float sampleInterval,
        long sequenceNumber)
    {
        if (tracks == null || tracks.Length == 0)
            return Array.Empty<TwinActorState>();

        float visibilityPadding = Mathf.Max(0.05f, sampleInterval * 0.55f);
        List<TwinActorState> actors = new List<TwinActorState>(tracks.Length);
        foreach (ActorReplayTrack track in tracks)
        {
            if (track == null || string.IsNullOrWhiteSpace(track.id) ||
                track.frames == null || track.frames.Length == 0)
                continue;

            ActorReplayFrame first = track.frames[0];
            ActorReplayFrame last = track.frames[track.frames.Length - 1];
            if (time < first.time - visibilityPadding || time > last.time + visibilityPadding)
                continue;

            FindPair(track.frames.Length, time, index => track.frames[index].time,
                out int fromIndex, out int toIndex, out float t);
            ActorReplayFrame from = track.frames[fromIndex];
            ActorReplayFrame to = track.frames[toIndex];
            Vector3 fromPosition = SafeVector(from.position);
            Vector3 toPosition = SafeVector(to.position);
            float seconds = Mathf.Max(0.0001f, to.time - from.time);
            Vector3 velocity = toIndex == fromIndex ? Vector3.zero : (toPosition - fromPosition) / seconds;
            float speed = velocity.magnitude;

            actors.Add(new TwinActorState
            {
                id = track.id,
                semanticClass = MapActorClass(track.category, track.prefabType),
                sourceClass = string.IsNullOrWhiteSpace(track.category) ? track.prefabType : track.category,
                motionState = speed < 0.3f
                    ? TwinActorMotionState.Stopped
                    : speed < 2.5f ? TwinActorMotionState.Slow : TwinActorMotionState.Moving,
                observationTimestampSeconds = time,
                observationSequenceNumber = sequenceNumber,
                freshness = TwinDataFreshness.Fresh,
                position = Vector3.Lerp(fromPosition, toPosition, t),
                yawDegrees = Mathf.LerpAngle(from.yawDegrees, to.yawDegrees, t),
                velocityMetersPerSecond = velocity,
                dimensionsMeters = SafeVector(track.size),
                confidence = 1f,
                visibility = t < 0.5f ? from.visibility : to.visibility,
                validity = TwinDataValidity.Valid
            });
        }

        return actors.ToArray();
    }

    private static TwinActorClass MapActorClass(string category, string prefabType)
    {
        string label = ((category ?? string.Empty) + " " + (prefabType ?? string.Empty)).ToLowerInvariant();
        if (label.Contains("pedestrian")) return TwinActorClass.Pedestrian;
        if (label.Contains("truck")) return TwinActorClass.Truck;
        if (label.Contains("bus")) return TwinActorClass.Bus;
        if (label.Contains("bicycle") || label.Contains("cycle")) return TwinActorClass.Bicycle;
        if (label.Contains("motorcycle")) return TwinActorClass.Motorcycle;
        if (label.Contains("barrier")) return TwinActorClass.Barrier;
        if (label.Contains("car") || label.Contains("vehicle")) return TwinActorClass.Car;
        return TwinActorClass.Other;
    }

    private static TwinSessionInfo CopySession(TwinSessionInfo session)
    {
        if (session == null)
            return new TwinSessionInfo();
        return new TwinSessionInfo
        {
            sourceId = session.sourceId,
            sessionId = session.sessionId,
            sourceKind = session.sourceKind,
            status = session.status,
            capabilities = session.capabilities,
            statusMessage = session.statusMessage
        };
    }

    private static Vector3 SafeVector(ReplayVector3 value)
    {
        return value == null ? Vector3.zero : value.ToVector3();
    }

    private static int Nearest(int from, int to, float t) => t < 0.5f ? from : to;

    private static void FindPair(
        int length,
        float time,
        Func<int, float> getTime,
        out int fromIndex,
        out int toIndex,
        out float interpolation)
    {
        fromIndex = FindLowerIndex(time, length, getTime);
        toIndex = Mathf.Min(fromIndex + 1, length - 1);
        float fromTime = getTime(fromIndex);
        float toTime = getTime(toIndex);
        interpolation = Mathf.Approximately(fromTime, toTime)
            ? 0f
            : Mathf.InverseLerp(fromTime, toTime, time);
    }

    private static int FindLowerIndex(float time, int length, Func<int, float> getTime)
    {
        if (length <= 1 || time <= getTime(0)) return 0;
        if (time >= getTime(length - 1)) return length - 1;

        int low = 0;
        int high = length - 1;
        while (low <= high)
        {
            int middle = (low + high) / 2;
            if (getTime(middle) <= time) low = middle + 1;
            else high = middle - 1;
        }
        return Mathf.Clamp(high, 0, length - 1);
    }
}

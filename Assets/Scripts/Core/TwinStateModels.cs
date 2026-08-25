using System;
using UnityEngine;

/// <summary>
/// The origin of a snapshot. Unity consumers must use the canonical snapshot and
/// must not branch on this value to obtain vehicle or actor data.
/// </summary>
public enum TwinSourceKind
{
    Unknown,
    Replay,
    SimulatedStream,
    LiveSensor
}

public enum TwinSessionStatus
{
    Disconnected,
    Connecting,
    Ready,
    Running,
    Paused,
    Finished,
    Error
}

public enum TwinDataValidity
{
    Unknown,
    Valid,
    Partial,
    Invalid
}

public enum TwinDataFreshness
{
    Unknown,
    Fresh,
    Stale
}

public enum TwinActorClass
{
    Unknown,
    Car,
    Truck,
    Bus,
    Pedestrian,
    Bicycle,
    Motorcycle,
    Barrier,
    TrafficLight,
    Other
}

public enum TwinActorMotionState
{
    Unknown,
    Stopped,
    Slow,
    Moving
}

[Flags]
public enum TwinSourceCapabilities
{
    None = 0,
    EgoPose = 1 << 0,
    VehicleTelemetry = 1 << 1,
    WheelTelemetry = 1 << 2,
    SurroundingActors = 1 << 3,
    Seek = 1 << 4,
    Pause = 1 << 5,
    LiveUpdates = 1 << 6,
    FutureTrajectory = 1 << 7
}

[Serializable]
public class TwinCoordinateFrame
{
    public const string CanonicalFrameId = "roadweave.unity.local";

    [Tooltip("Stable coordinate-frame identifier.")]
    public string frameId = CanonicalFrameId;
    [Tooltip("Parent frame, or empty when this frame is the local origin.")]
    public string parentFrameId = string.Empty;
    public string handedness = "left";
    public string horizontalUnit = "meter";
    public string angleUnit = "degree";
    public string xAxis = "right";
    public string yAxis = "up";
    public string zAxis = "forward";
    public string originDescription = "source-defined local origin";
    public string yawConvention = "rotation around +Y in Unity degrees";

    public static TwinCoordinateFrame UnityLocal(string originDescription)
    {
        return new TwinCoordinateFrame
        {
            originDescription = string.IsNullOrWhiteSpace(originDescription)
                ? "source-defined local origin"
                : originDescription
        };
    }
}

[Serializable]
public class TwinSessionInfo
{
    public string sourceId;
    public string sessionId;
    public TwinSourceKind sourceKind;
    public TwinSessionStatus status;
    public TwinSourceCapabilities capabilities;
    public string statusMessage;
}

[Serializable]
public class TwinSnapshotMetadata
{
    [Tooltip("Monotonically increasing within one source session.")]
    public long sequenceNumber;
    [Tooltip("Seconds on the source clock. For replay this is replay time.")]
    public double sourceTimestampSeconds;
    [Tooltip("Unity real-time seconds when the state manager accepted the snapshot.")]
    public double receiptTimestampSeconds;
    [Tooltip("Optional source timeline duration in seconds; zero means unknown/live.")]
    public double timelineDurationSeconds;
    [Tooltip("Maximum age after receipt for streaming sources. Replay snapshots remain fresh while paused.")]
    public float staleAfterSeconds = 1f;
    public TwinDataValidity validity = TwinDataValidity.Unknown;
    public TwinDataFreshness freshness = TwinDataFreshness.Unknown;
    public string validityMessage;
    public TwinCoordinateFrame coordinateFrame = new TwinCoordinateFrame();
    public TwinSessionInfo session = new TwinSessionInfo();
}

[Serializable]
public class TwinEgoState
{
    [Tooltip("Position in meters in metadata.coordinateFrame.")]
    public Vector3 position;
    [Tooltip("Rotation around +Y in degrees.")]
    public float yawDegrees;
    [Tooltip("Forward speed in meters per second.")]
    public float speedMetersPerSecond;
    [Tooltip("Forward acceleration in meters per second squared.")]
    public float longitudinalAcceleration;
    public TwinDataValidity validity = TwinDataValidity.Unknown;
}

[Serializable]
public class TwinVehicleState
{
    [Tooltip("Vehicle speed in kilometers per hour.")]
    public float speedKilometersPerHour;
    [Tooltip("Battery state of charge in percent [0, 100].")]
    public float batteryPercent;
    [Tooltip("Estimated remaining distance in kilometers.")]
    public float availableDistanceKilometers;
    public int gearPosition;
    [Tooltip("Throttle command in percent [0, 100].")]
    public float throttlePercent;
    [Tooltip("Brake command in source-defined normalized units; brakeIsValid indicates availability.")]
    public float brake;
    public int brakeSwitch;
    public float steeringDegrees;
    public float steeringSpeed;
    public float yawRate;
    public int leftSignal;
    public int rightSignal;
    public float temperatureCelsius;
    public bool temperatureIsValid;
    public TwinDataValidity validity = TwinDataValidity.Unknown;
}

[Serializable]
public class TwinWheelState
{
    [Tooltip("Wheel angular speeds in revolutions per minute.")]
    public float frontLeftRpm;
    public float frontRightRpm;
    public float rearLeftRpm;
    public float rearRightRpm;
    public TwinDataValidity validity = TwinDataValidity.Unknown;
}

[Serializable]
public class TwinRouteState
{
    [Tooltip("Stable route identifier within one source session.")]
    public string routeId;
    [Tooltip("Incremented only when the published route window changes.")]
    public long revision;
    [Tooltip("Distance between the two lane centerlines in meters.")]
    public float laneWidthMeters;
    [Tooltip("Ordered canonical positions for the ego/right-lane centerline.")]
    public Vector3[] points = Array.Empty<Vector3>();
    public TwinDataValidity validity = TwinDataValidity.Unknown;
}

[Serializable]
public class TwinActorState
{
    [Tooltip("Stable within a source session; never use a Unity instance ID here.")]
    public string id;
    public TwinActorClass semanticClass;
    [Tooltip("Original source label retained for diagnostics and future mappings.")]
    public string sourceClass;
    public TwinActorMotionState motionState;
    [Tooltip("Seconds on the source clock when this actor observation was made.")]
    public double observationTimestampSeconds = -1d;
    [Tooltip("Monotonic observation sequence within the source session, or -1 when unavailable.")]
    public long observationSequenceNumber = -1;
    [Tooltip("Freshness of this actor observation, independent from the enclosing snapshot.")]
    public TwinDataFreshness freshness = TwinDataFreshness.Unknown;
    [Tooltip("Position in meters in metadata.coordinateFrame.")]
    public Vector3 position;
    public float yawDegrees;
    [Tooltip("Velocity in meters per second in metadata.coordinateFrame.")]
    public Vector3 velocityMetersPerSecond;
    [Tooltip("Object dimensions in meters: x=width, y=height, z=length.")]
    public Vector3 dimensionsMeters;
    public float confidence;
    public string visibility;
    public TwinDataValidity validity = TwinDataValidity.Unknown;
}

/// <summary>
/// The single source-neutral contract consumed by RoadWeave presentation code.
/// Values use the units documented on each field and the coordinate frame in metadata.
/// </summary>
[Serializable]
public class TwinSnapshot
{
    public TwinSnapshotMetadata metadata = new TwinSnapshotMetadata();
    public TwinEgoState ego = new TwinEgoState();
    public TwinVehicleState vehicle = new TwinVehicleState();
    public TwinWheelState wheels = new TwinWheelState();
    [Tooltip("Optional route-ahead data. Older sources may omit it.")]
    public TwinRouteState route;
    public TwinActorState[] actors = Array.Empty<TwinActorState>();
}

public interface ITwinSnapshotSink
{
    bool PublishSnapshot(TwinSnapshot snapshot);
    void PublishSessionStatus(TwinSessionInfo session);
}

public interface ITwinStateSource
{
    TwinSessionInfo Session { get; }
    event Action<TwinSnapshot> SnapshotProduced;
    event Action<TwinSessionInfo> SessionChanged;
}

public interface ITwinSessionControl
{
    bool CanStart { get; }
    void StartSession();
    void PauseSession();
    void RestartSession(bool startRunning = false);
}

using System;
using UnityEngine;

[Serializable]
public class ReplayPackage
{
    public int schemaVersion;
    public string source;
    public string sceneId;
    public string description;
    public float durationSeconds;
    public float sampleIntervalSeconds;
    public string coordinateSystem;
    public bool hasTemperatureData;
    public EgoReplayFrame[] egoFrames;
    public VehicleReplayFrame[] vehicleFrames;
    public WheelReplayFrame[] wheelFrames;
    public ActorReplayTrack[] actors;
}

[Serializable]
public class ReplayVector3
{
    public float x;
    public float y;
    public float z;

    public Vector3 ToVector3()
    {
        return new Vector3(x, y, z);
    }
}

[Serializable]
public class EgoReplayFrame
{
    public float time;
    public ReplayVector3 position;
    public float yawDegrees;
    public float speedMetersPerSecond;
    public float longitudinalAcceleration;
}

[Serializable]
public class VehicleReplayFrame
{
    public float time;
    public float speedKilometersPerHour;
    public float batteryPercent;
    public float availableDistanceKilometers;
    public int gearPosition;
    public float throttlePercent;
    public float brake;
    public int brakeSwitch;
    public float steeringDegrees;
    public float steeringSpeed;
    public float yawRate;
    public int leftSignal;
    public int rightSignal;
}

[Serializable]
public class WheelReplayFrame
{
    public float time;
    public float frontLeftRpm;
    public float frontRightRpm;
    public float rearLeftRpm;
    public float rearRightRpm;
}

[Serializable]
public class ActorReplayTrack
{
    public string id;
    public string category;
    public string prefabType;
    public ReplayVector3 size;
    public ActorReplayFrame[] frames;
}

[Serializable]
public class ActorReplayFrame
{
    public float time;
    public ReplayVector3 position;
    public float yawDegrees;
    public string attribute;
    public string visibility;
}

public enum ReplayStatus
{
    Loading,
    Ready,
    Playing,
    Paused,
    Finished,
    Error
}

using System;
using UnityEngine;

public class DigitalTwinStateManager : MonoBehaviour
{
    public static DigitalTwinStateManager Instance { get; private set; }

    public ReplayPackage Package { get; private set; }
    public TwinEgoState Ego { get; } = new TwinEgoState();
    public TwinVehicleState Vehicle { get; } = new TwinVehicleState();
    public TwinWheelState Wheels { get; } = new TwinWheelState();
    public ReplayStatus Status { get; private set; } = ReplayStatus.Loading;
    public float CurrentTime { get; private set; }
    public bool IsReady => Package != null && Status != ReplayStatus.Error;
    public bool IsPlaying => Status == ReplayStatus.Playing;

    public event Action Initialized;
    public event Action StateUpdated;
    public event Action<ReplayStatus> StatusChanged;

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Debug.LogWarning("A second DigitalTwinStateManager was removed.");
            Destroy(this);
            return;
        }

        Instance = this;
    }

    private void OnDestroy()
    {
        if (Instance == this)
            Instance = null;
    }

    public void Initialize(ReplayPackage replayPackage)
    {
        Package = replayPackage;
        CurrentTime = 0f;
        UpdateState(0f);
        SetStatus(ReplayStatus.Ready);
        Initialized?.Invoke();
    }

    public void SetStatus(ReplayStatus newStatus)
    {
        if (Status == newStatus)
            return;

        Status = newStatus;
        StatusChanged?.Invoke(Status);
    }

    public void SetError()
    {
        SetStatus(ReplayStatus.Error);
    }

    public void UpdateState(float replayTime)
    {
        if (Package == null)
            return;

        CurrentTime = Mathf.Clamp(replayTime, 0f, Package.durationSeconds);
        SampleEgo(CurrentTime);
        SampleVehicle(CurrentTime);
        SampleWheels(CurrentTime);
        StateUpdated?.Invoke();
    }

    private void SampleEgo(float time)
    {
        EgoReplayFrame[] frames = Package.egoFrames;
        if (frames == null || frames.Length == 0)
            return;

        FindEgoPair(frames, time, out EgoReplayFrame from, out EgoReplayFrame to, out float t);
        Ego.position = Vector3.Lerp(from.position.ToVector3(), to.position.ToVector3(), t);
        Ego.yawDegrees = Mathf.LerpAngle(from.yawDegrees, to.yawDegrees, t);
        Ego.speedMetersPerSecond = Mathf.Lerp(from.speedMetersPerSecond, to.speedMetersPerSecond, t);
        Ego.longitudinalAcceleration = Mathf.Lerp(from.longitudinalAcceleration, to.longitudinalAcceleration, t);
    }

    private void SampleVehicle(float time)
    {
        VehicleReplayFrame[] frames = Package.vehicleFrames;
        if (frames == null || frames.Length == 0)
            return;

        FindVehiclePair(frames, time, out VehicleReplayFrame from, out VehicleReplayFrame to, out float t);
        Vehicle.speedKilometersPerHour = Mathf.Lerp(from.speedKilometersPerHour, to.speedKilometersPerHour, t);
        Vehicle.batteryPercent = Mathf.Lerp(from.batteryPercent, to.batteryPercent, t);
        Vehicle.availableDistanceKilometers = Mathf.Lerp(from.availableDistanceKilometers, to.availableDistanceKilometers, t);
        Vehicle.gearPosition = from.gearPosition;
        Vehicle.throttlePercent = Mathf.Lerp(from.throttlePercent, to.throttlePercent, t);
        Vehicle.brake = Mathf.Lerp(from.brake, to.brake, t);
        Vehicle.brakeSwitch = from.brakeSwitch;
        Vehicle.steeringDegrees = Mathf.Lerp(from.steeringDegrees, to.steeringDegrees, t);
        Vehicle.steeringSpeed = Mathf.Lerp(from.steeringSpeed, to.steeringSpeed, t);
        Vehicle.yawRate = Mathf.Lerp(from.yawRate, to.yawRate, t);
        Vehicle.leftSignal = from.leftSignal;
        Vehicle.rightSignal = from.rightSignal;
    }

    private void SampleWheels(float time)
    {
        WheelReplayFrame[] frames = Package.wheelFrames;
        if (frames == null || frames.Length == 0)
            return;

        FindWheelPair(frames, time, out WheelReplayFrame from, out WheelReplayFrame to, out float t);
        Wheels.frontLeftRpm = Mathf.Lerp(from.frontLeftRpm, to.frontLeftRpm, t);
        Wheels.frontRightRpm = Mathf.Lerp(from.frontRightRpm, to.frontRightRpm, t);
        Wheels.rearLeftRpm = Mathf.Lerp(from.rearLeftRpm, to.rearLeftRpm, t);
        Wheels.rearRightRpm = Mathf.Lerp(from.rearRightRpm, to.rearRightRpm, t);
    }

    private static int FindLowerIndex(float time, int length, Func<int, float> getTime)
    {
        if (length <= 1 || time <= getTime(0))
            return 0;

        if (time >= getTime(length - 1))
            return length - 1;

        int low = 0;
        int high = length - 1;

        while (low <= high)
        {
            int middle = (low + high) / 2;
            if (getTime(middle) <= time)
                low = middle + 1;
            else
                high = middle - 1;
        }

        return Mathf.Clamp(high, 0, length - 1);
    }

    private static float Interpolation(float time, float fromTime, float toTime)
    {
        if (Mathf.Approximately(fromTime, toTime))
            return 0f;
        return Mathf.InverseLerp(fromTime, toTime, time);
    }

    private static void FindEgoPair(EgoReplayFrame[] frames, float time, out EgoReplayFrame from, out EgoReplayFrame to, out float t)
    {
        int fromIndex = FindLowerIndex(time, frames.Length, index => frames[index].time);
        int toIndex = Mathf.Min(fromIndex + 1, frames.Length - 1);
        from = frames[fromIndex];
        to = frames[toIndex];
        t = Interpolation(time, from.time, to.time);
    }

    private static void FindVehiclePair(VehicleReplayFrame[] frames, float time, out VehicleReplayFrame from, out VehicleReplayFrame to, out float t)
    {
        int fromIndex = FindLowerIndex(time, frames.Length, index => frames[index].time);
        int toIndex = Mathf.Min(fromIndex + 1, frames.Length - 1);
        from = frames[fromIndex];
        to = frames[toIndex];
        t = Interpolation(time, from.time, to.time);
    }

    private static void FindWheelPair(WheelReplayFrame[] frames, float time, out WheelReplayFrame from, out WheelReplayFrame to, out float t)
    {
        int fromIndex = FindLowerIndex(time, frames.Length, index => frames[index].time);
        int toIndex = Mathf.Min(fromIndex + 1, frames.Length - 1);
        from = frames[fromIndex];
        to = frames[toIndex];
        t = Interpolation(time, from.time, to.time);
    }
}

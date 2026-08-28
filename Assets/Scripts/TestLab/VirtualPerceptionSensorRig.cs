using System;
using System.Collections.Generic;
using UnityEngine;

public enum VirtualPerceptionSensorType
{
    Camera,
    Lidar,
    Radar
}

public enum VirtualSensorFaultType
{
    None,
    Dropout,
    RangeNoise,
    VelocityNoise,
    Bias,
    Latency,
    FalsePositive,
    Calibration,
    CompleteFailure
}

[Serializable]
public sealed class VirtualSensorHealthFeatures
{
    public string sensorId;
    public string sensorType;
    public string weather;
    public float dropoutRate;
    public float messageAgeMean;
    public float messageAgeMax;
    public float detectionCountMean;
    public float detectionCountStd;
    public float confidenceMean;
    public float confidenceStd;
    public float trackContinuity;
    public float rangeVariance;
    public float velocityVariance;
    public float innovationMean;
    public float innovationStd;
    public float crossSensorDisagreement;
    public float egoSpeedMean;
    public float egoSpeedStd;
    public float yawRateMean;
    public float yawRateStd;
}

public sealed class VirtualPerceptionFrame
{
    public SimulatedSensorSnapshot fusedSnapshot = new SimulatedSensorSnapshot();
    public VirtualSensorHealthFeatures[] sensorHealth = Array.Empty<VirtualSensorHealthFeatures>();
}

/// <summary>
/// Object-level camera/LiDAR/radar simulation. The existing
/// SimulatedVehicleSensorSuite supplies ground truth. This component produces
/// noisy, delayed, missing and occasionally false observations for the learned
/// policy, while the controller retains the untouched truth only for its final
/// collision-safety envelope.
/// </summary>
[DisallowMultipleComponent]
public sealed class VirtualPerceptionSensorRig : MonoBehaviour
{
    [Header("Reliability Window")]
    [SerializeField, Min(1f)] private float historySeconds = 3f;
    [SerializeField, Min(5f)] private float faultProfileDurationSeconds = 25f;
    [SerializeField] private int reproducibleSeed = 2026;

    [Header("Fault Generation")]
    [SerializeField, Range(0f, 1f)] private float faultProbability = 0.65f;
    [SerializeField, Range(0f, 1f)] private float minimumFaultSeverity = 0.15f;
    [SerializeField, Range(0f, 1f)] private float maximumFaultSeverity = 0.90f;

    private sealed class ChannelProfile
    {
        public VirtualPerceptionSensorType type;
        public string id;
        public float updateRateHz;
        public float nominalDropout;
        public float nominalRangeNoise;
        public float nominalVelocityNoise;
        public float nominalLatency;
        public float baseDropout;
        public float rangeNoise;
        public float velocityNoise;
        public float latency;
        public float rangeFactor = 1f;
        public float falsePositiveRate;
        public float rangeBias;
        public float laneBias;
        public VirtualSensorFaultType fault;
        public float severity;
    }

    private sealed class Detection
    {
        public SimulatedActorObservation truth;
        public float gap;
        public float actorSpeed;
        public float closingSpeed;
        public float confidence;
    }

    private sealed class HealthSample
    {
        public float timestamp;
        public bool messageReceived;
        public float messageAge;
        public float detectionCount;
        public float confidence;
        public float meanRange;
        public float meanVelocity;
        public float innovation;
        public float egoSpeed;
        public float yawRate;
        public HashSet<string> tracks = new HashSet<string>();
    }

    private sealed class ChannelState
    {
        public ChannelProfile profile;
        public float nextUpdateTime;
        public float lastSuccessfulTime = float.NegativeInfinity;
        public Dictionary<string, Detection> detections = new Dictionary<string, Detection>();
        public readonly Queue<HealthSample> history = new Queue<HealthSample>();
        public HashSet<string> previousTracks = new HashSet<string>();
        public float previousMeanRange;
        public bool hasPreviousMeanRange;
    }

    private readonly Dictionary<VirtualPerceptionSensorType, ChannelState> channels =
        new Dictionary<VirtualPerceptionSensorType, ChannelState>();
    private System.Random random;
    private float nextFaultProfileTime;
    private float latestTimestamp;
    private string latestWeather = "Dry";

    public string FaultSummary { get; private set; } = "Virtual sensors not initialized";

    private void Awake()
    {
        Initialize(reproducibleSeed);
    }

    public void Initialize(int seed)
    {
        reproducibleSeed = seed;
        random = new System.Random(seed);
        channels.Clear();
        AddChannel(VirtualPerceptionSensorType.Camera, "front_camera", 15f, 0.02f, 0.45f, 0.80f, 0.08f);
        AddChannel(VirtualPerceptionSensorType.Lidar, "roof_lidar", 10f, 0.01f, 0.12f, 0.35f, 0.10f);
        AddChannel(VirtualPerceptionSensorType.Radar, "front_radar", 20f, 0.01f, 0.30f, 0.12f, 0.05f);
        latestTimestamp = 0f;
        nextFaultProfileTime = 0f;
        RandomizeFaultProfiles();
    }

    public VirtualPerceptionFrame Observe(
        SimulatedSensorSnapshot groundTruth,
        string weather,
        float timestamp,
        float egoSpeedMps,
        float yawRateDegreesPerSecond)
    {
        if (groundTruth == null)
            groundTruth = new SimulatedSensorSnapshot();
        if (random == null || channels.Count == 0)
            Initialize(reproducibleSeed);

        latestTimestamp = Mathf.Max(latestTimestamp, timestamp);
        latestWeather = string.IsNullOrWhiteSpace(weather) ? "Dry" : weather;
        if (latestTimestamp >= nextFaultProfileTime)
            RandomizeFaultProfiles();

        Dictionary<string, SimulatedActorObservation> truthSlots = BuildTruthSlots(groundTruth);
        foreach (ChannelState channel in channels.Values)
        {
            if (latestTimestamp + 0.0001f < channel.nextUpdateTime)
                continue;
            channel.nextUpdateTime = latestTimestamp + 1f / Mathf.Max(1f, channel.profile.updateRateHz);
            UpdateChannel(channel, truthSlots, egoSpeedMps, yawRateDegreesPerSecond);
        }

        VirtualPerceptionFrame frame = new VirtualPerceptionFrame();
        frame.fusedSnapshot = Fuse(truthSlots);
        frame.sensorHealth = BuildHealthFeatures(egoSpeedMps, yawRateDegreesPerSecond);
        return frame;
    }

    private void AddChannel(
        VirtualPerceptionSensorType type,
        string id,
        float updateRateHz,
        float dropout,
        float rangeNoise,
        float velocityNoise,
        float latency)
    {
        channels[type] = new ChannelState
        {
            profile = new ChannelProfile
            {
                type = type,
                id = id,
                updateRateHz = updateRateHz,
                nominalDropout = dropout,
                nominalRangeNoise = rangeNoise,
                nominalVelocityNoise = velocityNoise,
                nominalLatency = latency,
                baseDropout = dropout,
                rangeNoise = rangeNoise,
                velocityNoise = velocityNoise,
                latency = latency
            }
        };
    }

    private void RandomizeFaultProfiles()
    {
        nextFaultProfileTime = latestTimestamp + Mathf.Max(5f, faultProfileDurationSeconds);
        List<string> descriptions = new List<string>();
        foreach (ChannelState channel in channels.Values)
        {
            ResetBaseProfile(channel.profile);
            if (NextFloat() < faultProbability)
            {
                Array values = Enum.GetValues(typeof(VirtualSensorFaultType));
                channel.profile.fault = (VirtualSensorFaultType)values.GetValue(random.Next(1, values.Length));
                channel.profile.severity = Mathf.Lerp(
                    minimumFaultSeverity,
                    maximumFaultSeverity,
                    NextFloat()
                );
                ApplyFault(channel.profile);
            }
            descriptions.Add($"{channel.profile.type}:{channel.profile.fault} {channel.profile.severity:P0}");
        }
        FaultSummary = string.Join(", ", descriptions);
    }

    private static void ResetBaseProfile(ChannelProfile profile)
    {
        profile.fault = VirtualSensorFaultType.None;
        profile.severity = 0f;
        profile.baseDropout = profile.nominalDropout;
        profile.rangeNoise = profile.nominalRangeNoise;
        profile.velocityNoise = profile.nominalVelocityNoise;
        profile.latency = profile.nominalLatency;
        profile.rangeFactor = 1f;
        profile.falsePositiveRate = 0f;
        profile.rangeBias = 0f;
        profile.laneBias = 0f;
    }

    private static void ApplyFault(ChannelProfile profile)
    {
        float severity = profile.severity;
        switch (profile.fault)
        {
            case VirtualSensorFaultType.Dropout:
                profile.baseDropout = Mathf.Clamp01(profile.baseDropout + 0.75f * severity);
                break;
            case VirtualSensorFaultType.RangeNoise:
                profile.rangeNoise *= 1f + 8f * severity;
                break;
            case VirtualSensorFaultType.VelocityNoise:
                profile.velocityNoise *= 1f + 10f * severity;
                break;
            case VirtualSensorFaultType.Bias:
                profile.rangeBias = Mathf.Lerp(0.5f, 6f, severity);
                break;
            case VirtualSensorFaultType.Latency:
                profile.latency += Mathf.Lerp(0.1f, 1.2f, severity);
                break;
            case VirtualSensorFaultType.FalsePositive:
                profile.falsePositiveRate = Mathf.Lerp(0.05f, 0.65f, severity);
                break;
            case VirtualSensorFaultType.Calibration:
                profile.laneBias = Mathf.Lerp(0.3f, 3f, severity);
                profile.rangeBias = Mathf.Lerp(0.2f, 2f, severity);
                break;
            case VirtualSensorFaultType.CompleteFailure:
                profile.baseDropout = 1f;
                profile.rangeFactor = 0f;
                break;
        }
    }

    private void UpdateChannel(
        ChannelState channel,
        Dictionary<string, SimulatedActorObservation> truthSlots,
        float egoSpeed,
        float yawRate)
    {
        ChannelProfile active = WeatherAdjusted(channel.profile, latestWeather);
        HealthSample sample = new HealthSample
        {
            timestamp = latestTimestamp,
            egoSpeed = egoSpeed,
            yawRate = yawRate
        };

        bool messageReceived = NextFloat() >= active.baseDropout;
        sample.messageReceived = messageReceived;
        sample.messageAge = messageReceived
            ? Mathf.Max(0f, active.latency + Gaussian(0f, 0.015f))
            : float.IsNegativeInfinity(channel.lastSuccessfulTime)
                ? historySeconds
                : Mathf.Max(0f, latestTimestamp - channel.lastSuccessfulTime);

        if (messageReceived)
        {
            channel.lastSuccessfulTime = latestTimestamp;
            channel.detections.Clear();
            float confidenceSum = 0f;
            float rangeSum = 0f;
            float velocitySum = 0f;
            int detected = 0;
            foreach (KeyValuePair<string, SimulatedActorObservation> item in truthSlots)
            {
                SimulatedActorObservation truth = item.Value;
                if (truth == null)
                    continue;
                float effectiveRange = SensorRange(active.type) * active.rangeFactor;
                if (truth.gapDistance > effectiveRange)
                    continue;
                float weatherMiss = WeatherDetectionPenalty(active.type, latestWeather);
                float missProbability = Mathf.Clamp01(active.baseDropout * 0.35f + weatherMiss);
                if (NextFloat() < missProbability)
                    continue;

                float gap = Mathf.Max(0f, truth.gapDistance + active.rangeBias + Gaussian(0f, active.rangeNoise));
                float actorSpeed = truth.actorLongitudinalSpeed + Gaussian(0f, active.velocityNoise);
                float closingSpeed = Mathf.Max(0f, egoSpeed - actorSpeed);
                float normalizedError = Mathf.Abs(gap - truth.gapDistance) / Mathf.Max(1f, truth.gapDistance);
                float confidence = Mathf.Clamp01(0.98f - normalizedError - active.latency * 0.15f - weatherMiss * 0.5f);
                Detection detection = new Detection
                {
                    truth = truth,
                    gap = gap,
                    actorSpeed = actorSpeed,
                    closingSpeed = closingSpeed,
                    confidence = confidence
                };
                channel.detections[item.Key] = detection;
                sample.tracks.Add(item.Key);
                confidenceSum += confidence;
                rangeSum += gap;
                velocitySum += actorSpeed;
                detected++;
            }

            int falsePositives = NextFloat() < active.falsePositiveRate ? 1 : 0;
            sample.detectionCount = detected + falsePositives;
            // A valid frame with no nearby actors is healthy, not a
            // zero-confidence detection. Dropout and stale-message features
            // represent sensor failure independently of road occupancy.
            sample.confidence = detected > 0
                ? confidenceSum / detected
                : falsePositives > 0 ? 0.25f : 1f;
            sample.meanRange = detected > 0 ? rangeSum / detected : 0f;
            sample.meanVelocity = detected > 0 ? velocitySum / detected : 0f;
            if (channel.hasPreviousMeanRange && detected > 0)
                sample.innovation = Mathf.Abs(sample.meanRange - channel.previousMeanRange);
            channel.previousMeanRange = sample.meanRange;
            channel.hasPreviousMeanRange = detected > 0;
        }
        else
        {
            channel.detections.Clear();
        }

        channel.history.Enqueue(sample);
        while (channel.history.Count > 1 &&
               channel.history.Peek().timestamp < latestTimestamp - historySeconds)
            channel.history.Dequeue();
    }

    private ChannelProfile WeatherAdjusted(ChannelProfile source, string weather)
    {
        ChannelProfile result = new ChannelProfile
        {
            type = source.type,
            id = source.id,
            updateRateHz = source.updateRateHz,
            nominalDropout = source.nominalDropout,
            nominalRangeNoise = source.nominalRangeNoise,
            nominalVelocityNoise = source.nominalVelocityNoise,
            nominalLatency = source.nominalLatency,
            baseDropout = source.baseDropout,
            rangeNoise = source.rangeNoise,
            velocityNoise = source.velocityNoise,
            latency = source.latency,
            rangeFactor = source.rangeFactor,
            falsePositiveRate = source.falsePositiveRate,
            rangeBias = source.rangeBias,
            laneBias = source.laneBias,
            fault = source.fault,
            severity = source.severity
        };
        string normalized = (weather ?? "Dry").Trim().ToUpperInvariant();
        if (normalized == "RAIN")
        {
            if (result.type == VirtualPerceptionSensorType.Camera) { result.baseDropout += 0.08f; result.rangeNoise *= 1.6f; result.rangeFactor *= 0.82f; }
            else if (result.type == VirtualPerceptionSensorType.Lidar) { result.baseDropout += 0.05f; result.rangeNoise *= 1.8f; result.rangeFactor *= 0.86f; }
            else { result.baseDropout += 0.015f; result.rangeNoise *= 1.15f; }
        }
        else if (normalized == "SNOW")
        {
            if (result.type == VirtualPerceptionSensorType.Camera) { result.baseDropout += 0.16f; result.rangeNoise *= 2.2f; result.rangeFactor *= 0.68f; }
            else if (result.type == VirtualPerceptionSensorType.Lidar) { result.baseDropout += 0.13f; result.rangeNoise *= 2.8f; result.rangeFactor *= 0.64f; }
            else { result.baseDropout += 0.04f; result.rangeNoise *= 1.35f; }
        }
        else if (normalized == "FOG")
        {
            if (result.type == VirtualPerceptionSensorType.Camera) { result.baseDropout += 0.28f; result.rangeNoise *= 2.5f; result.rangeFactor *= 0.45f; }
            else if (result.type == VirtualPerceptionSensorType.Lidar) { result.baseDropout += 0.14f; result.rangeNoise *= 2.1f; result.rangeFactor *= 0.70f; }
            else { result.baseDropout += 0.02f; result.rangeNoise *= 1.1f; }
        }
        result.baseDropout = Mathf.Clamp01(result.baseDropout);
        return result;
    }

    private static float WeatherDetectionPenalty(VirtualPerceptionSensorType type, string weather)
    {
        string normalized = (weather ?? "Dry").Trim().ToUpperInvariant();
        if (normalized == "FOG")
            return type == VirtualPerceptionSensorType.Camera ? 0.28f : type == VirtualPerceptionSensorType.Lidar ? 0.12f : 0.015f;
        if (normalized == "SNOW")
            return type == VirtualPerceptionSensorType.Camera ? 0.16f : type == VirtualPerceptionSensorType.Lidar ? 0.13f : 0.035f;
        if (normalized == "RAIN")
            return type == VirtualPerceptionSensorType.Camera ? 0.08f : type == VirtualPerceptionSensorType.Lidar ? 0.05f : 0.01f;
        return 0f;
    }

    private static float SensorRange(VirtualPerceptionSensorType type)
    {
        switch (type)
        {
            case VirtualPerceptionSensorType.Camera: return 55f;
            case VirtualPerceptionSensorType.Lidar: return 70f;
            default: return 100f;
        }
    }

    private static Dictionary<string, SimulatedActorObservation> BuildTruthSlots(SimulatedSensorSnapshot truth)
    {
        return new Dictionary<string, SimulatedActorObservation>
        {
            { "right_front", truth.rightLaneFront },
            { "right_rear", truth.rightLaneRear },
            { "left_front", truth.leftLaneFront },
            { "left_rear", truth.leftLaneRear },
            { "pedestrian", truth.pedestrianHazard }
        };
    }

    private SimulatedSensorSnapshot Fuse(Dictionary<string, SimulatedActorObservation> truthSlots)
    {
        SimulatedSensorSnapshot result = new SimulatedSensorSnapshot();
        result.rightLaneFront = FuseSlot("right_front", truthSlots["right_front"]);
        result.rightLaneRear = FuseSlot("right_rear", truthSlots["right_rear"]);
        result.leftLaneFront = FuseSlot("left_front", truthSlots["left_front"]);
        result.leftLaneRear = FuseSlot("left_rear", truthSlots["left_rear"]);
        result.pedestrianHazard = FuseSlot("pedestrian", truthSlots["pedestrian"]);
        return result;
    }

    private SimulatedActorObservation FuseSlot(string slot, SimulatedActorObservation truth)
    {
        if (truth == null)
            return null;
        float gap = 0f;
        float speed = 0f;
        float closing = 0f;
        float totalWeight = 0f;
        foreach (ChannelState channel in channels.Values)
        {
            if (latestTimestamp - channel.lastSuccessfulTime > 0.75f)
                continue;
            if (!channel.detections.TryGetValue(slot, out Detection detection))
                continue;
            float weight = Mathf.Max(0.05f, detection.confidence);
            gap += detection.gap * weight;
            speed += detection.actorSpeed * weight;
            closing += detection.closingSpeed * weight;
            totalWeight += weight;
        }
        if (totalWeight <= 0f)
            return null;

        float fusedGap = gap / totalWeight;
        float fusedClosing = closing / totalWeight;
        return new SimulatedActorObservation
        {
            actor = truth.actor,
            collider = truth.collider,
            combinedBounds = truth.combinedBounds,
            longitudinalDistance = truth.longitudinalDistance,
            laneOffset = truth.laneOffset,
            gapDistance = fusedGap,
            actorLongitudinalSpeed = speed / totalWeight,
            relativeClosingSpeed = fusedClosing,
            timeToCollision = fusedClosing > 0.05f ? fusedGap / fusedClosing : float.PositiveInfinity
        };
    }

    private VirtualSensorHealthFeatures[] BuildHealthFeatures(float egoSpeed, float yawRate)
    {
        Dictionary<VirtualPerceptionSensorType, float> meanRanges = new Dictionary<VirtualPerceptionSensorType, float>();
        foreach (ChannelState channel in channels.Values)
            meanRanges[channel.profile.type] = LatestSuccessfulMeanRange(channel);

        List<VirtualSensorHealthFeatures> features = new List<VirtualSensorHealthFeatures>();
        foreach (ChannelState channel in channels.Values)
        {
            List<HealthSample> samples = new List<HealthSample>(channel.history);
            VirtualSensorHealthFeatures item = new VirtualSensorHealthFeatures
            {
                sensorId = channel.profile.id,
                sensorType = channel.profile.type.ToString(),
                weather = latestWeather,
                dropoutRate = 1f - Mean(samples, sample => sample.messageReceived ? 1f : 0f),
                messageAgeMean = Mean(samples, sample => sample.messageAge),
                messageAgeMax = Maximum(samples, sample => sample.messageAge),
                detectionCountMean = Mean(samples, sample => sample.detectionCount),
                detectionCountStd = StandardDeviation(samples, sample => sample.detectionCount),
                confidenceMean = Mean(samples, sample => sample.confidence),
                confidenceStd = StandardDeviation(samples, sample => sample.confidence),
                trackContinuity = TrackContinuity(samples),
                rangeVariance = Variance(samples, sample => sample.meanRange),
                velocityVariance = Variance(samples, sample => sample.meanVelocity),
                innovationMean = Mean(samples, sample => sample.innovation),
                innovationStd = StandardDeviation(samples, sample => sample.innovation),
                crossSensorDisagreement = CrossSensorDisagreement(channel.profile.type, meanRanges),
                egoSpeedMean = Mean(samples, sample => sample.egoSpeed, egoSpeed),
                egoSpeedStd = StandardDeviation(samples, sample => sample.egoSpeed),
                yawRateMean = Mean(samples, sample => sample.yawRate, yawRate),
                yawRateStd = StandardDeviation(samples, sample => sample.yawRate)
            };
            features.Add(item);
        }
        return features.ToArray();
    }

    private static float LatestSuccessfulMeanRange(ChannelState channel)
    {
        HealthSample[] samples = channel.history.ToArray();
        for (int index = samples.Length - 1; index >= 0; index--)
            if (samples[index].messageReceived && samples[index].detectionCount > 0f)
                return samples[index].meanRange;
        return float.NaN;
    }

    private static float CrossSensorDisagreement(
        VirtualPerceptionSensorType current,
        Dictionary<VirtualPerceptionSensorType, float> values)
    {
        if (!values.TryGetValue(current, out float own) || float.IsNaN(own))
            return 0f;
        float total = 0f;
        int count = 0;
        foreach (KeyValuePair<VirtualPerceptionSensorType, float> item in values)
        {
            if (item.Key == current || float.IsNaN(item.Value))
                continue;
            total += Mathf.Abs(own - item.Value);
            count++;
        }
        return count > 0 ? total / count : 0f;
    }

    private static float TrackContinuity(List<HealthSample> samples)
    {
        HashSet<string> previous = null;
        float total = 0f;
        int comparisons = 0;
        foreach (HealthSample sample in samples)
        {
            if (!sample.messageReceived)
                continue;
            if (previous != null)
            {
                HashSet<string> union = new HashSet<string>(previous);
                union.UnionWith(sample.tracks);
                if (union.Count > 0)
                {
                    HashSet<string> intersection = new HashSet<string>(previous);
                    intersection.IntersectWith(sample.tracks);
                    total += (float)intersection.Count / union.Count;
                    comparisons++;
                }
            }
            previous = sample.tracks;
        }
        return comparisons > 0 ? total / comparisons : 1f;
    }

    private static float Mean(List<HealthSample> samples, Func<HealthSample, float> selector, float fallback = 0f)
    {
        if (samples.Count == 0)
            return fallback;
        float total = 0f;
        foreach (HealthSample sample in samples)
            total += selector(sample);
        return total / samples.Count;
    }

    private static float Maximum(List<HealthSample> samples, Func<HealthSample, float> selector)
    {
        float maximum = 0f;
        foreach (HealthSample sample in samples)
            maximum = Mathf.Max(maximum, selector(sample));
        return maximum;
    }

    private static float Variance(List<HealthSample> samples, Func<HealthSample, float> selector)
    {
        if (samples.Count < 2)
            return 0f;
        float mean = Mean(samples, selector);
        float total = 0f;
        foreach (HealthSample sample in samples)
        {
            float difference = selector(sample) - mean;
            total += difference * difference;
        }
        return total / samples.Count;
    }

    private static float StandardDeviation(List<HealthSample> samples, Func<HealthSample, float> selector)
    {
        return Mathf.Sqrt(Variance(samples, selector));
    }

    private float NextFloat()
    {
        return (float)random.NextDouble();
    }

    private float Gaussian(float mean, float standardDeviation)
    {
        double first = Math.Max(1e-12, random.NextDouble());
        double second = random.NextDouble();
        double normal = Math.Sqrt(-2.0 * Math.Log(first)) * Math.Cos(2.0 * Math.PI * second);
        return mean + standardDeviation * (float)normal;
    }
}

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

/// <summary>
/// Source-neutral instrumentation for Experiment 1. It observes only snapshots
/// accepted by DigitalTwinStateManager, so replay and streaming sources are
/// measured through the same interface without changing either adapter.
/// </summary>
public sealed class SourceExperimentRecorder : MonoBehaviour
{
    [SerializeField] private DigitalTwinStateManager stateManager;
    [SerializeField, Min(1f)] private float measurementDurationSeconds = 15f;
    [SerializeField] private string outputFolderName = "ExperimentResults";
    [SerializeField] private bool recordAutomatically = true;

    public bool IsRecording { get; private set; }
    public string LastSamplesPath { get; private set; }
    public string LastSummaryPath { get; private set; }

    private readonly List<SnapshotSample> samples = new List<SnapshotSample>();
    private readonly List<double> interArrivalMilliseconds = new List<double>();
    private readonly List<double> frameTimeMilliseconds = new List<double>();

    private double applicationStartedAt;
    private double readyAt = -1d;
    private string readySessionId;
    private double runStartedAt;
    private double firstSnapshotAt = -1d;
    private double previousReceiptAt = -1d;
    private long previousSequence;
    private bool hasPreviousSequence;
    private string activeSessionId;
    private string completedSessionId;
    private TwinSourceKind activeSourceKind;
    private string activeSourceId;
    private int runNumber;
    private int sequenceGapCount;
    private int nonMonotonicAcceptedCount;
    private int partialSnapshotCount;
    private int staleSnapshotCount;
    private int freshFrameCount;
    private int observedFrameCount;
    private int rejectedSnapshotLogCount;
    private int rejectedOutOfOrderLogCount;
    private int warningCount;
    private int errorCount;
    private long transportPacketsReceivedAtStart = -1;
    private long transportPacketsPublishedAtStart = -1;
    private SimulatedJsonUdpTransport simulatedTransport;

    private sealed class SnapshotSample
    {
        public double elapsedSeconds;
        public double unityTimeSeconds;
        public double sourceTimestampSeconds;
        public double receiptTimestampSeconds;
        public long sequenceNumber;
        public double interArrivalMilliseconds = double.NaN;
        public string sessionStatus;
        public string validity;
        public string freshness;
        public Vector3 egoPosition;
        public float speedKilometersPerHour;
        public int actorCount;
        public double frameTimeMilliseconds;
        public double framesPerSecond;
    }

    private void Awake()
    {
        applicationStartedAt = Time.realtimeSinceStartup;
        ResolveStateManager();
    }

    private void OnEnable()
    {
        ResolveStateManager();
        if (stateManager != null)
        {
            stateManager.SnapshotUpdated += HandleSnapshotUpdated;
            stateManager.SessionChanged += HandleSessionChanged;
            HandleSessionChanged(stateManager.Session);
        }
        Application.logMessageReceived += HandleLogMessage;
    }

    private void Update()
    {
        if (!IsRecording)
            return;

        double frameMilliseconds = Time.unscaledDeltaTime * 1000d;
        if (frameMilliseconds > 0d)
            frameTimeMilliseconds.Add(frameMilliseconds);

        observedFrameCount++;
        if (stateManager != null && stateManager.IsFresh)
            freshFrameCount++;

        if (Time.realtimeSinceStartup - runStartedAt >= measurementDurationSeconds)
            CompleteRun("measurement_window_complete", true);
    }

    private void OnDisable()
    {
        if (IsRecording)
            CompleteRun("recorder_disabled_before_window_completed", false);
        if (stateManager != null)
        {
            stateManager.SnapshotUpdated -= HandleSnapshotUpdated;
            stateManager.SessionChanged -= HandleSessionChanged;
        }
        Application.logMessageReceived -= HandleLogMessage;
    }

    private void ResolveStateManager()
    {
        if (stateManager == null)
            stateManager = FindAnyObjectByType<DigitalTwinStateManager>();
    }

    private void HandleSessionChanged(TwinSessionInfo session)
    {
        if (session == null)
            return;

        double now = Time.realtimeSinceStartup;
        string sessionId = session.sessionId ?? string.Empty;

        if (session.status == TwinSessionStatus.Ready)
        {
            readyAt = now;
            readySessionId = sessionId;
        }

        if (session.status == TwinSessionStatus.Running)
        {
            if (recordAutomatically && !IsRecording && sessionId != completedSessionId)
                BeginRun(session, now);
            return;
        }

        if (!IsRecording)
            return;

        if (session.status == TwinSessionStatus.Finished)
            CompleteRun("source_session_finished", true);
        else if (session.status == TwinSessionStatus.Error)
            CompleteRun("source_session_error", false);
        else if (session.status == TwinSessionStatus.Disconnected)
            CompleteRun("source_disconnected", false);
    }

    private void BeginRun(TwinSessionInfo session, double now)
    {
        samples.Clear();
        interArrivalMilliseconds.Clear();
        frameTimeMilliseconds.Clear();
        firstSnapshotAt = -1d;
        previousReceiptAt = -1d;
        previousSequence = 0;
        hasPreviousSequence = false;
        sequenceGapCount = 0;
        nonMonotonicAcceptedCount = 0;
        partialSnapshotCount = 0;
        staleSnapshotCount = 0;
        freshFrameCount = 0;
        observedFrameCount = 0;
        rejectedSnapshotLogCount = 0;
        rejectedOutOfOrderLogCount = 0;
        warningCount = 0;
        errorCount = 0;

        activeSessionId = session.sessionId ?? string.Empty;
        activeSourceKind = session.sourceKind;
        activeSourceId = string.IsNullOrWhiteSpace(session.sourceId)
            ? session.sourceKind.ToString()
            : session.sourceId;
        runStartedAt = now;
        runNumber = FindNextRunNumber(SourceSlug(activeSourceKind));

        simulatedTransport = activeSourceKind == TwinSourceKind.SimulatedStream
            ? FindAnyObjectByType<SimulatedJsonUdpTransport>()
            : null;
        transportPacketsReceivedAtStart = simulatedTransport == null
            ? -1
            : simulatedTransport.PacketsReceived;
        transportPacketsPublishedAtStart = simulatedTransport == null
            ? -1
            : simulatedTransport.PacketsPublished;

        IsRecording = true;
        Debug.Log(
            $"RoadWeave Experiment 1 recording started: {activeSourceKind}, " +
            $"run {runNumber:D3}, {measurementDurationSeconds:F1} seconds.");
    }

    private void HandleSnapshotUpdated(TwinSnapshot snapshot)
    {
        if (!IsRecording || snapshot?.metadata?.session == null)
            return;
        if (!string.Equals(snapshot.metadata.session.sessionId, activeSessionId, StringComparison.Ordinal))
            return;

        long sequence = snapshot.metadata.sequenceNumber;
        if (hasPreviousSequence && sequence == previousSequence)
        {
            // Freshness expiry republishes the current snapshot. Update its
            // recorded state without counting it as another source observation.
            if (samples.Count > 0)
                samples[samples.Count - 1].freshness = snapshot.metadata.freshness.ToString();
            return;
        }

        if (hasPreviousSequence)
        {
            if (sequence > previousSequence + 1)
            {
                long missing = sequence - previousSequence - 1;
                sequenceGapCount += missing > int.MaxValue ? int.MaxValue : (int)missing;
            }
            else if (sequence < previousSequence)
            {
                nonMonotonicAcceptedCount++;
            }
        }

        double receipt = snapshot.metadata.receiptTimestampSeconds;
        double interArrival = double.NaN;
        if (previousReceiptAt >= 0d && receipt >= previousReceiptAt)
        {
            interArrival = (receipt - previousReceiptAt) * 1000d;
            interArrivalMilliseconds.Add(interArrival);
        }

        double now = Time.realtimeSinceStartup;
        if (firstSnapshotAt < 0d)
            firstSnapshotAt = now;

        if (snapshot.metadata.validity == TwinDataValidity.Partial)
            partialSnapshotCount++;
        if (snapshot.metadata.freshness == TwinDataFreshness.Stale)
            staleSnapshotCount++;

        float speed = snapshot.vehicle != null && snapshot.vehicle.validity != TwinDataValidity.Invalid
            ? snapshot.vehicle.speedKilometersPerHour
            : (snapshot.ego?.speedMetersPerSecond ?? 0f) * 3.6f;
        double frameMilliseconds = Time.unscaledDeltaTime * 1000d;

        samples.Add(new SnapshotSample
        {
            elapsedSeconds = now - runStartedAt,
            unityTimeSeconds = now,
            sourceTimestampSeconds = snapshot.metadata.sourceTimestampSeconds,
            receiptTimestampSeconds = receipt,
            sequenceNumber = sequence,
            interArrivalMilliseconds = interArrival,
            sessionStatus = snapshot.metadata.session.status.ToString(),
            validity = snapshot.metadata.validity.ToString(),
            freshness = snapshot.metadata.freshness.ToString(),
            egoPosition = snapshot.ego?.position ?? Vector3.zero,
            speedKilometersPerHour = speed,
            actorCount = snapshot.actors?.Length ?? 0,
            frameTimeMilliseconds = frameMilliseconds,
            framesPerSecond = frameMilliseconds > 0d ? 1000d / frameMilliseconds : 0d
        });

        previousReceiptAt = receipt;
        previousSequence = sequence;
        hasPreviousSequence = true;
    }

    private void CompleteRun(string completionReason, bool nominalCompletion)
    {
        if (!IsRecording)
            return;
        IsRecording = false;

        double finishedAt = Time.realtimeSinceStartup;
        double actualDuration = Math.Max(0d, finishedAt - runStartedAt);
        TwinSessionStatus finalStatus = stateManager?.Session?.status ?? TwinSessionStatus.Disconnected;
        long transportReceived = simulatedTransport == null
            ? -1
            : Math.Max(0L, simulatedTransport.PacketsReceived - transportPacketsReceivedAtStart);
        long transportPublished = simulatedTransport == null
            ? -1
            : Math.Max(0L, simulatedTransport.PacketsPublished - transportPacketsPublishedAtStart);

        string outputDirectory = ResolveOutputDirectory();
        Directory.CreateDirectory(outputDirectory);
        string baseName = $"experiment1_{SourceSlug(activeSourceKind)}_run_{runNumber:D3}";
        LastSamplesPath = Path.Combine(outputDirectory, baseName + "_samples.csv");
        LastSummaryPath = Path.Combine(outputDirectory, baseName + "_summary.csv");

        try
        {
            File.WriteAllText(LastSamplesPath, BuildSamplesCsv(), new UTF8Encoding(false));
            File.WriteAllText(
                LastSummaryPath,
                BuildSummaryCsv(
                    completionReason,
                    finalStatus,
                    nominalCompletion,
                    actualDuration,
                    transportReceived,
                    transportPublished),
                new UTF8Encoding(false));
            Debug.Log(
                "RoadWeave Experiment 1 CSV files saved automatically:\n" +
                LastSamplesPath + "\n" + LastSummaryPath);
        }
        catch (Exception exception)
        {
            Debug.LogError("Could not save RoadWeave Experiment 1 CSV files: " + exception.Message);
        }

        completedSessionId = activeSessionId;
    }

    private string BuildSamplesCsv()
    {
        StringBuilder csv = new StringBuilder();
        csv.AppendLine(
            "run_id,source_kind,source_id,session_id,experiment_elapsed_s,unity_time_s," +
            "source_timestamp_s,receipt_timestamp_s,sequence_number,inter_arrival_ms," +
            "session_status,validity,freshness,ego_x_m,ego_y_m,ego_z_m,speed_kph," +
            "actor_count,unity_frame_time_ms,fps");

        string runId = $"{SourceSlug(activeSourceKind)}_{runNumber:D3}";
        foreach (SnapshotSample sample in samples)
        {
            csv.Append(Csv(runId)).Append(',')
                .Append(Csv(activeSourceKind.ToString())).Append(',')
                .Append(Csv(activeSourceId)).Append(',')
                .Append(Csv(activeSessionId)).Append(',')
                .Append(Number(sample.elapsedSeconds)).Append(',')
                .Append(Number(sample.unityTimeSeconds)).Append(',')
                .Append(Number(sample.sourceTimestampSeconds)).Append(',')
                .Append(Number(sample.receiptTimestampSeconds)).Append(',')
                .Append(sample.sequenceNumber.ToString(CultureInfo.InvariantCulture)).Append(',')
                .Append(OptionalNumber(sample.interArrivalMilliseconds)).Append(',')
                .Append(Csv(sample.sessionStatus)).Append(',')
                .Append(Csv(sample.validity)).Append(',')
                .Append(Csv(sample.freshness)).Append(',')
                .Append(Number(sample.egoPosition.x)).Append(',')
                .Append(Number(sample.egoPosition.y)).Append(',')
                .Append(Number(sample.egoPosition.z)).Append(',')
                .Append(Number(sample.speedKilometersPerHour)).Append(',')
                .Append(sample.actorCount.ToString(CultureInfo.InvariantCulture)).Append(',')
                .Append(Number(sample.frameTimeMilliseconds)).Append(',')
                .Append(Number(sample.framesPerSecond)).AppendLine();
        }
        return csv.ToString();
    }

    private string BuildSummaryCsv(
        string completionReason,
        TwinSessionStatus finalStatus,
        bool nominalCompletion,
        double actualDuration,
        long transportReceived,
        long transportPublished)
    {
        double startupMilliseconds = readyAt >= 0d && readySessionId == activeSessionId
            ? (readyAt - applicationStartedAt) * 1000d
            : double.NaN;
        double driveResponseMilliseconds = firstSnapshotAt >= 0d
            ? (firstSnapshotAt - runStartedAt) * 1000d
            : double.NaN;
        double effectiveRate = actualDuration > 0d ? samples.Count / actualDuration : 0d;
        double freshSnapshotPercentage = samples.Count == 0
            ? 0d
            : 100d * CountFreshSnapshots() / samples.Count;
        double freshFramePercentage = observedFrameCount == 0
            ? 0d
            : 100d * freshFrameCount / observedFrameCount;
        double meanFrameMilliseconds = Mean(frameTimeMilliseconds);
        double minimumFramesPerSecond = frameTimeMilliseconds.Count == 0
            ? 0d
            : 1000d / Maximum(frameTimeMilliseconds);
        double meanActors = MeanActorCount();
        int maximumActors = MaximumActorCount();
        double transportPublishPercentage = transportReceived <= 0
            ? double.NaN
            : 100d * transportPublished / transportReceived;
        bool completedWithoutError = nominalCompletion && samples.Count > 0 && errorCount == 0;

        StringBuilder csv = new StringBuilder();
        csv.AppendLine(
            "run_id,source_kind,source_id,session_id,completion_reason,final_session_status," +
            "completed_without_error,startup_time_ms,drive_response_time_ms,target_duration_s," +
            "actual_duration_s,accepted_snapshots,effective_update_rate_hz,mean_inter_arrival_ms," +
            "median_inter_arrival_ms,p95_inter_arrival_ms,sequence_gap_count," +
            "non_monotonic_accepted_count,partial_snapshot_count,stale_snapshot_count," +
            "fresh_snapshot_percentage,fresh_frame_percentage,mean_frame_time_ms," +
            "p95_frame_time_ms,mean_fps,minimum_fps,mean_actor_count,maximum_actor_count," +
            "rejected_snapshot_log_count,rejected_out_of_order_log_count,warning_count,error_count," +
            "transport_packets_received,transport_packets_published,transport_publish_percentage");
        csv.Append(Csv($"{SourceSlug(activeSourceKind)}_{runNumber:D3}")).Append(',')
            .Append(Csv(activeSourceKind.ToString())).Append(',')
            .Append(Csv(activeSourceId)).Append(',')
            .Append(Csv(activeSessionId)).Append(',')
            .Append(Csv(completionReason)).Append(',')
            .Append(Csv(finalStatus.ToString())).Append(',')
            .Append(completedWithoutError ? "true" : "false").Append(',')
            .Append(OptionalNumber(startupMilliseconds)).Append(',')
            .Append(OptionalNumber(driveResponseMilliseconds)).Append(',')
            .Append(Number(measurementDurationSeconds)).Append(',')
            .Append(Number(actualDuration)).Append(',')
            .Append(samples.Count.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(Number(effectiveRate)).Append(',')
            .Append(OptionalNumber(Mean(interArrivalMilliseconds))).Append(',')
            .Append(OptionalNumber(Percentile(interArrivalMilliseconds, 50d))).Append(',')
            .Append(OptionalNumber(Percentile(interArrivalMilliseconds, 95d))).Append(',')
            .Append(sequenceGapCount.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(nonMonotonicAcceptedCount.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(partialSnapshotCount.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(staleSnapshotCount.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(Number(freshSnapshotPercentage)).Append(',')
            .Append(Number(freshFramePercentage)).Append(',')
            .Append(OptionalNumber(meanFrameMilliseconds)).Append(',')
            .Append(OptionalNumber(Percentile(frameTimeMilliseconds, 95d))).Append(',')
            .Append(meanFrameMilliseconds > 0d ? Number(1000d / meanFrameMilliseconds) : string.Empty).Append(',')
            .Append(Number(minimumFramesPerSecond)).Append(',')
            .Append(Number(meanActors)).Append(',')
            .Append(maximumActors.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(rejectedSnapshotLogCount.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(rejectedOutOfOrderLogCount.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(warningCount.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(errorCount.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(transportReceived >= 0 ? transportReceived.ToString(CultureInfo.InvariantCulture) : string.Empty).Append(',')
            .Append(transportPublished >= 0 ? transportPublished.ToString(CultureInfo.InvariantCulture) : string.Empty).Append(',')
            .Append(OptionalNumber(transportPublishPercentage)).AppendLine();
        return csv.ToString();
    }

    private void HandleLogMessage(string condition, string stackTrace, LogType type)
    {
        if (!IsRecording)
            return;
        if (type == LogType.Warning)
            warningCount++;
        else if (type == LogType.Error || type == LogType.Exception || type == LogType.Assert)
            errorCount++;

        if (condition.IndexOf("Rejected out-of-order snapshot", StringComparison.OrdinalIgnoreCase) >= 0)
            rejectedOutOfOrderLogCount++;
        else if (condition.IndexOf("Rejected digital-twin snapshot", StringComparison.OrdinalIgnoreCase) >= 0 ||
                 condition.IndexOf("Rejected SimulatedStream snapshot", StringComparison.OrdinalIgnoreCase) >= 0 ||
                 condition.IndexOf("Could not parse snapshot JSON", StringComparison.OrdinalIgnoreCase) >= 0)
            rejectedSnapshotLogCount++;
    }

    private string ResolveOutputDirectory()
    {
        string folder = string.IsNullOrWhiteSpace(outputFolderName)
            ? "ExperimentResults"
            : outputFolderName.Trim();
#if UNITY_EDITOR
        return Path.GetFullPath(Path.Combine(Application.dataPath, "..", folder));
#else
        return Path.Combine(Application.persistentDataPath, folder);
#endif
    }

    private int FindNextRunNumber(string sourceSlug)
    {
        string directory = ResolveOutputDirectory();
        if (!Directory.Exists(directory))
            return 1;

        int maximum = 0;
        string prefix = $"experiment1_{sourceSlug}_run_";
        foreach (string path in Directory.GetFiles(directory, prefix + "*_summary.csv"))
        {
            string name = Path.GetFileNameWithoutExtension(path);
            int start = prefix.Length;
            int end = name.IndexOf("_summary", start, StringComparison.Ordinal);
            if (end > start && int.TryParse(name.Substring(start, end - start), out int value))
                maximum = Math.Max(maximum, value);
        }
        return maximum + 1;
    }

    private int CountFreshSnapshots()
    {
        int count = 0;
        foreach (SnapshotSample sample in samples)
            if (sample.freshness == TwinDataFreshness.Fresh.ToString())
                count++;
        return count;
    }

    private double MeanActorCount()
    {
        if (samples.Count == 0)
            return 0d;
        double total = 0d;
        foreach (SnapshotSample sample in samples)
            total += sample.actorCount;
        return total / samples.Count;
    }

    private int MaximumActorCount()
    {
        int maximum = 0;
        foreach (SnapshotSample sample in samples)
            maximum = Math.Max(maximum, sample.actorCount);
        return maximum;
    }

    private static string SourceSlug(TwinSourceKind sourceKind)
    {
        switch (sourceKind)
        {
            case TwinSourceKind.Replay: return "nuscenes";
            case TwinSourceKind.SimulatedStream: return "simulated";
            case TwinSourceKind.LiveSensor: return "live_sensor";
            default: return "unknown";
        }
    }

    private static double Mean(List<double> values)
    {
        if (values.Count == 0)
            return double.NaN;
        double total = 0d;
        foreach (double value in values)
            total += value;
        return total / values.Count;
    }

    private static double Maximum(List<double> values)
    {
        if (values.Count == 0)
            return double.NaN;
        double maximum = values[0];
        for (int index = 1; index < values.Count; index++)
            maximum = Math.Max(maximum, values[index]);
        return maximum;
    }

    private static double Percentile(List<double> values, double percentile)
    {
        if (values.Count == 0)
            return double.NaN;
        List<double> sorted = new List<double>(values);
        sorted.Sort();
        double rank = percentile / 100d * (sorted.Count - 1);
        int lower = (int)Math.Floor(rank);
        int upper = (int)Math.Ceiling(rank);
        if (lower == upper)
            return sorted[lower];
        double fraction = rank - lower;
        return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
    }

    private static string Number(double value) =>
        value.ToString("0.######", CultureInfo.InvariantCulture);

    private static string OptionalNumber(double value) =>
        double.IsNaN(value) || double.IsInfinity(value) ? string.Empty : Number(value);

    private static string Csv(string value)
    {
        string safe = value ?? string.Empty;
        if (safe.IndexOfAny(new[] { ',', '"', '\n', '\r' }) < 0)
            return safe;
        return "\"" + safe.Replace("\"", "\"\"") + "\"";
    }
}

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Profiling;

/// <summary>
/// Records RoadWeave Experiment 3 without changing the normal twin pipeline.
/// It measures accepted snapshots, displayed-state freshness, rejection policy,
/// Unity responsiveness, outage recovery and the first post-recovery position jump.
/// </summary>
[DisallowMultipleComponent]
public sealed class FaultInjectionExperimentRecorder : MonoBehaviour
{
    [Header("RoadWeave References")]
    [SerializeField] private DigitalTwinStateManager stateManager;
    [SerializeField] private SimulatedJsonUdpTransport transport;
    [SerializeField] private SimulatedStreamTwinSource simulatedSource;
    [SerializeField] private Transform displayedEgoVehicle;

    [Header("Experiment 3 Protocol")]
    [SerializeField, Min(0f)] private float warmupDurationSeconds = 30f;
    [SerializeField, Min(1f)] private float measurementDurationSeconds = 300f;
    [SerializeField, Min(2f)] private float maximumExpectedSilenceSeconds = 15f;
    [SerializeField] private string outputFolderName = "ExperimentResults/experiment3/unity";
    [SerializeField] private bool recordAutomatically = true;
    [SerializeField] private bool requireExperimentRunId = true;

    public bool IsWarmingUp { get; private set; }
    public bool IsRecording { get; private set; }
    public string LastSummaryPath { get; private set; }

    private sealed class SnapshotRow
    {
        public double elapsed;
        public long sequence;
        public double sourceUtc;
        public double acceptedUtc;
        public double latencyMs;
        public long sequenceGap;
        public string validity;
        public string freshness;
        public Vector3 position;
    }

    private sealed class FrameRow
    {
        public double elapsed;
        public double frameUtc;
        public long sequence;
        public double sourceUtc;
        public double stateAgeMs;
        public double frameTimeMs;
        public double fps;
        public double allocatedMb;
        public Vector3 position;
        public double positionStep;
        public string freshness;
    }

    private sealed class EventRow
    {
        public double elapsed;
        public string eventType;
        public long sequence;
        public string detail;
    }

    private readonly List<SnapshotRow> snapshots = new List<SnapshotRow>();
    private readonly List<FrameRow> frames = new List<FrameRow>();
    private readonly List<EventRow> events = new List<EventRow>();
    private readonly List<double> latencies = new List<double>();
    private readonly List<double> stateAges = new List<double>();
    private readonly List<double> frameTimes = new List<double>();

    private double warmupStarted;
    private float activeWarmupDuration;
    private float activeMeasurementDuration;
    private double measurementStarted;
    private double measurementStartedUtc;
    private double lastAcceptedRealtime = -1d;
    private string activeSessionId;
    private string completedSessionId;
    private string runId = "unidentified";
    private string profileId = "baseline";
    private TwinSnapshot latest;
    private TwinTransportDiagnostics diagnostics;
    private long lastRecordedSequence = -1;
    private long previousAcceptedSequence = -1;
    private Vector3 previousDisplayedPosition;
    private bool hasPreviousPosition;
    private int staleFrames;
    private bool previouslyStale;
    private double staleStartedElapsed = double.NaN;
    private double recoveryElapsed = double.NaN;
    private double recoveryTimeMs = double.NaN;
    private double recoveryJumpWindowEnd = -1d;
    private double maximumRecoveryJump;
    private bool outageStartLogged;
    private bool outageEndLogged;
    private int warningCount;
    private int errorCount;
    private long receivedStart;
    private long publishedStart;
    private long rejectedInvalidStart;
    private long rejectedDuplicateStart;
    private long rejectedOutOfOrderStart;
    private long rejectedContractStart;
    private long rejectedMalformedJsonStart;

    private void Awake() => ResolveReferences();

    private void OnEnable()
    {
        ResolveReferences();
        Subscribe();
        Application.logMessageReceived += HandleLog;
        if (stateManager != null) HandleSession(stateManager.Session);
    }

    private void OnDisable()
    {
        if (IsRecording) Finish("recorder_disabled", false);
        IsWarmingUp = false;
        Unsubscribe();
        Application.logMessageReceived -= HandleLog;
    }

    private void Update()
    {
        double now = Time.realtimeSinceStartup;
        if (IsWarmingUp && now - warmupStarted >= activeWarmupDuration)
            BeginMeasurement(now);
        if (!IsRecording) return;

        RecordFrame(now);
        RecordScheduledOutageEvents(now - measurementStarted);
        if (lastAcceptedRealtime >= 0d && now - lastAcceptedRealtime > maximumExpectedSilenceSeconds)
        {
            Finish("source_silence_exceeded_limit", false);
            return;
        }
        if (now - measurementStarted >= activeMeasurementDuration)
            Finish("measurement_window_complete", true);
    }

    private void ResolveReferences()
    {
        if (stateManager == null) stateManager = FindAnyObjectByType<DigitalTwinStateManager>();
        if (transport == null) transport = FindAnyObjectByType<SimulatedJsonUdpTransport>();
        if (simulatedSource == null) simulatedSource = FindAnyObjectByType<SimulatedStreamTwinSource>();
        if (displayedEgoVehicle == null)
        {
            GameObject ego = GameObject.Find("EgoVehicleRoot");
            if (ego != null) displayedEgoVehicle = ego.transform;
        }
    }

    private void Subscribe()
    {
        if (stateManager == null) return;
        stateManager.SnapshotUpdated -= HandleSnapshot;
        stateManager.SnapshotUpdated += HandleSnapshot;
        stateManager.SessionChanged -= HandleSession;
        stateManager.SessionChanged += HandleSession;
        stateManager.SnapshotRejected -= HandleRejected;
        stateManager.SnapshotRejected += HandleRejected;
    }

    private void Unsubscribe()
    {
        if (stateManager == null) return;
        stateManager.SnapshotUpdated -= HandleSnapshot;
        stateManager.SessionChanged -= HandleSession;
        stateManager.SnapshotRejected -= HandleRejected;
    }

    private void HandleSession(TwinSessionInfo session)
    {
        if (session == null || session.sourceKind != TwinSourceKind.SimulatedStream) return;
        string sessionId = session.sessionId ?? string.Empty;
        if (session.status == TwinSessionStatus.Running)
        {
            if (recordAutomatically && !requireExperimentRunId && !IsWarmingUp && !IsRecording &&
                !string.Equals(sessionId, completedSessionId, StringComparison.Ordinal))
                BeginWarmup(sessionId);
        }
        else if (IsRecording)
        {
            Finish("source_stopped_before_window_completed", false);
        }
        else if (IsWarmingUp)
        {
            IsWarmingUp = false;
        }
    }

    private void BeginWarmup(string sessionId)
    {
        ResetRun();
        activeSessionId = sessionId;
        warmupStarted = Time.realtimeSinceStartup;
        activeWarmupDuration = warmupDurationSeconds;
        activeMeasurementDuration = measurementDurationSeconds;
        IsWarmingUp = true;
        Debug.Log($"RoadWeave Experiment 3 warm-up started. Measurement begins in {activeWarmupDuration:F0} seconds.");
        if (activeWarmupDuration <= 0f) BeginMeasurement(warmupStarted);
    }

    private void BeginMeasurement(double now)
    {
        if (!IsWarmingUp) return;
        IsWarmingUp = false;
        IsRecording = true;
        measurementStarted = now;
        measurementStartedUtc = UtcNow();
        lastAcceptedRealtime = now;
        receivedStart = transport == null ? -1 : transport.PacketsReceived;
        publishedStart = transport == null ? -1 : transport.PacketsPublished;
        rejectedInvalidStart = stateManager == null ? 0 : stateManager.RejectedInvalidSnapshotCount;
        rejectedDuplicateStart = stateManager == null ? 0 : stateManager.RejectedDuplicateSnapshotCount;
        rejectedOutOfOrderStart = stateManager == null ? 0 : stateManager.RejectedOutOfOrderSnapshotCount;
        rejectedContractStart = simulatedSource == null ? 0 : simulatedSource.RejectedContractSnapshotCount;
        rejectedMalformedJsonStart = simulatedSource == null ? 0 : simulatedSource.RejectedMalformedJsonCount;
        if (displayedEgoVehicle != null)
        {
            previousDisplayedPosition = displayedEgoVehicle.localPosition;
            hasPreviousPosition = true;
        }
        events.Add(new EventRow { elapsed = 0d, eventType = "measurement_started", sequence = lastRecordedSequence, detail = profileId });
        Debug.Log($"RoadWeave Experiment 3 measurement started for run '{runId}' ({activeMeasurementDuration:F0} seconds).");
    }

    private void HandleSnapshot(TwinSnapshot snapshot)
    {
        if (snapshot?.metadata?.session == null ||
            snapshot.metadata.session.sourceKind != TwinSourceKind.SimulatedStream) return;

        string snapshotSessionId = snapshot.metadata.session.sessionId ?? string.Empty;
        if (!string.IsNullOrEmpty(activeSessionId) &&
            !string.Equals(snapshotSessionId, activeSessionId, StringComparison.Ordinal)) return;

        TwinTransportDiagnostics incomingDiagnostics = snapshot.metadata.transportDiagnostics;
        if (recordAutomatically && !IsWarmingUp && !IsRecording &&
            snapshot.metadata.session.status == TwinSessionStatus.Running &&
            !string.Equals(snapshotSessionId, completedSessionId, StringComparison.Ordinal) &&
            (!requireExperimentRunId || !string.IsNullOrWhiteSpace(incomingDiagnostics?.runId)))
        {
            BeginWarmup(snapshotSessionId);
        }

        latest = snapshot;
        if (incomingDiagnostics != null)
        {
            diagnostics = incomingDiagnostics;
            if (!string.IsNullOrWhiteSpace(diagnostics.runId)) runId = diagnostics.runId;
            if (!string.IsNullOrWhiteSpace(diagnostics.profileId)) profileId = diagnostics.profileId;
            if (IsWarmingUp)
            {
                if (diagnostics.experimentWarmupSeconds >= 0f)
                    activeWarmupDuration = diagnostics.experimentWarmupSeconds;
                if (diagnostics.experimentDurationSeconds > 0f)
                    activeMeasurementDuration = diagnostics.experimentDurationSeconds;
            }
        }
        if (!IsRecording) return;

        bool stale = snapshot.metadata.freshness == TwinDataFreshness.Stale;
        if (stale && !previouslyStale)
        {
            staleStartedElapsed = Time.realtimeSinceStartup - measurementStarted;
            events.Add(new EventRow { elapsed = staleStartedElapsed, eventType = "state_became_stale", sequence = snapshot.metadata.sequenceNumber, detail = "holding_last_accepted_state" });
        }
        else if (!stale && previouslyStale)
        {
            MarkRecovery(snapshot.metadata.sequenceNumber, "fresh_state_recovered");
        }
        previouslyStale = stale;

        if (!stale && double.IsNaN(recoveryElapsed) && diagnostics != null &&
            diagnostics.disconnectAtMeasurementSeconds >= 0f && diagnostics.disconnectDurationSeconds > 0f)
        {
            double elapsed = Time.realtimeSinceStartup - measurementStarted;
            double expectedEnd = diagnostics.disconnectAtMeasurementSeconds + diagnostics.disconnectDurationSeconds;
            if (elapsed >= expectedEnd)
                MarkRecovery(snapshot.metadata.sequenceNumber, "first_snapshot_after_outage");
        }

        long sequence = snapshot.metadata.sequenceNumber;
        if (stale || sequence == lastRecordedSequence) return;
        long gap = previousAcceptedSequence < 0 ? 0 : Math.Max(0, sequence - previousAcceptedSequence - 1);
        previousAcceptedSequence = sequence;
        lastRecordedSequence = sequence;
        lastAcceptedRealtime = Time.realtimeSinceStartup;
        double acceptedUtc = UtcNow();
        double latency = snapshot.metadata.sentTimestampUtcSeconds > 0d
            ? (acceptedUtc - snapshot.metadata.sentTimestampUtcSeconds) * 1000d
            : double.NaN;
        if (latency >= 0d && latency < 60000d) latencies.Add(latency); else latency = double.NaN;
        snapshots.Add(new SnapshotRow
        {
            elapsed = lastAcceptedRealtime - measurementStarted,
            sequence = sequence,
            sourceUtc = snapshot.metadata.sentTimestampUtcSeconds,
            acceptedUtc = acceptedUtc,
            latencyMs = latency,
            sequenceGap = gap,
            validity = snapshot.metadata.validity.ToString(),
            freshness = snapshot.metadata.freshness.ToString(),
            position = snapshot.ego?.position ?? Vector3.zero
        });
    }

    private void HandleRejected(TwinSnapshot snapshot, string reason)
    {
        if (!IsRecording) return;
        events.Add(new EventRow
        {
            elapsed = Time.realtimeSinceStartup - measurementStarted,
            eventType = "snapshot_rejected",
            sequence = snapshot?.metadata == null ? -1 : snapshot.metadata.sequenceNumber,
            detail = reason
        });
    }

    private void RecordScheduledOutageEvents(double elapsed)
    {
        if (diagnostics == null || diagnostics.disconnectAtMeasurementSeconds < 0f ||
            diagnostics.disconnectDurationSeconds <= 0f) return;
        double start = diagnostics.disconnectAtMeasurementSeconds;
        double end = start + diagnostics.disconnectDurationSeconds;
        if (!outageStartLogged && elapsed >= start)
        {
            outageStartLogged = true;
            events.Add(new EventRow { elapsed = elapsed, eventType = "scheduled_outage_started", sequence = lastRecordedSequence, detail = $"expected_start_s={start:F3}" });
        }
        if (!outageEndLogged && elapsed >= end)
        {
            outageEndLogged = true;
            events.Add(new EventRow { elapsed = elapsed, eventType = "scheduled_outage_ended", sequence = lastRecordedSequence, detail = $"expected_end_s={end:F3}" });
        }
    }

    private void MarkRecovery(long sequence, string eventType)
    {
        if (!double.IsNaN(recoveryElapsed)) return;
        recoveryElapsed = Time.realtimeSinceStartup - measurementStarted;
        double expectedEnd = diagnostics != null && diagnostics.disconnectAtMeasurementSeconds >= 0f
            ? diagnostics.disconnectAtMeasurementSeconds + diagnostics.disconnectDurationSeconds
            : staleStartedElapsed;
        recoveryTimeMs = Math.Max(0d, (recoveryElapsed - expectedEnd) * 1000d);
        recoveryJumpWindowEnd = Time.realtimeSinceStartup + 1d;
        events.Add(new EventRow { elapsed = recoveryElapsed, eventType = eventType, sequence = sequence, detail = $"recovery_ms={recoveryTimeMs:F3}" });
    }

    private void RecordFrame(double now)
    {
        double frameTime = Time.unscaledDeltaTime * 1000d;
        if (frameTime <= 0d) return;
        double frameUtc = UtcNow();
        double stateAge = latest?.metadata != null && latest.metadata.sentTimestampUtcSeconds > 0d
            ? (frameUtc - latest.metadata.sentTimestampUtcSeconds) * 1000d
            : double.NaN;
        if (stateAge >= 0d && stateAge < 60000d) stateAges.Add(stateAge); else stateAge = double.NaN;
        frameTimes.Add(frameTime);
        string freshness = latest?.metadata == null ? "Unknown" : latest.metadata.freshness.ToString();
        if (freshness == TwinDataFreshness.Stale.ToString()) staleFrames++;
        // EgoVehicleReplayView writes canonical snapshot coordinates to the
        // vehicle root's local transform. World position includes ReplayRoot's
        // city placement and rotation and therefore cannot be compared to the
        // Python source truth directly.
        Vector3 position = displayedEgoVehicle == null ? (latest?.ego?.position ?? Vector3.zero) : displayedEgoVehicle.localPosition;
        double step = hasPreviousPosition ? Vector3.Distance(previousDisplayedPosition, position) : double.NaN;
        if (now <= recoveryJumpWindowEnd && !double.IsNaN(step))
            maximumRecoveryJump = double.IsNaN(maximumRecoveryJump) ? step : Math.Max(maximumRecoveryJump, step);
        previousDisplayedPosition = position;
        hasPreviousPosition = true;
        frames.Add(new FrameRow
        {
            elapsed = now - measurementStarted,
            frameUtc = frameUtc,
            sequence = latest?.metadata == null ? -1 : latest.metadata.sequenceNumber,
            sourceUtc = latest?.metadata == null ? 0d : latest.metadata.sentTimestampUtcSeconds,
            stateAgeMs = stateAge,
            frameTimeMs = frameTime,
            fps = 1000d / frameTime,
            allocatedMb = Profiler.GetTotalAllocatedMemoryLong() / (1024d * 1024d),
            position = position,
            positionStep = step,
            freshness = freshness
        });
    }

    private void Finish(string reason, bool nominal)
    {
        if (!IsRecording) return;
        IsRecording = false;
        IsWarmingUp = false;
        double duration = Math.Max(0d, Time.realtimeSinceStartup - measurementStarted);
        bool valid = nominal && duration >= activeMeasurementDuration - 0.25d && snapshots.Count > 0 && errorCount == 0;
        string directory = ResolveOutputDirectory();
        Directory.CreateDirectory(directory);
        string safeRunId = SafeName(runId);
        string baseName = FindAvailableBase(directory, $"experiment3_{safeRunId}");
        string snapshotPath = Path.Combine(directory, baseName + "_snapshots.csv");
        string framePath = Path.Combine(directory, baseName + "_frames.csv");
        string eventPath = Path.Combine(directory, baseName + "_events.csv");
        LastSummaryPath = Path.Combine(directory, baseName + "_summary.csv");
        try
        {
            File.WriteAllText(snapshotPath, SnapshotsCsv(), new UTF8Encoding(false));
            File.WriteAllText(framePath, FramesCsv(), new UTF8Encoding(false));
            File.WriteAllText(eventPath, EventsCsv(), new UTF8Encoding(false));
            File.WriteAllText(LastSummaryPath, SummaryCsv(reason, duration, valid), new UTF8Encoding(false));
            Debug.Log($"RoadWeave Experiment 3 CSV files saved. Valid for analysis: {valid}.\n{LastSummaryPath}");
        }
        catch (Exception exception)
        {
            Debug.LogError("Could not save Experiment 3 CSV files: " + exception.Message);
        }
        completedSessionId = activeSessionId;
    }

    private string SnapshotsCsv()
    {
        StringBuilder csv = new StringBuilder("run_id,profile_id,experiment_elapsed_s,sequence_number,source_utc_s,accepted_utc_s,latency_ms,sequence_gap,validity,freshness,ego_x_m,ego_y_m,ego_z_m\n");
        foreach (SnapshotRow row in snapshots)
            csv.Append(Csv(runId)).Append(',').Append(Csv(profileId)).Append(',').Append(N(row.elapsed)).Append(',').Append(row.sequence).Append(',').Append(N(row.sourceUtc)).Append(',').Append(N(row.acceptedUtc)).Append(',').Append(O(row.latencyMs)).Append(',').Append(row.sequenceGap).Append(',').Append(Csv(row.validity)).Append(',').Append(Csv(row.freshness)).Append(',').Append(N(row.position.x)).Append(',').Append(N(row.position.y)).Append(',').Append(N(row.position.z)).Append('\n');
        return csv.ToString();
    }

    private string FramesCsv()
    {
        StringBuilder csv = new StringBuilder("run_id,profile_id,experiment_elapsed_s,frame_utc_s,latest_sequence,latest_source_utc_s,state_age_ms,frame_time_ms,fps,allocated_memory_mb,displayed_x_m,displayed_y_m,displayed_z_m,position_step_m,freshness\n");
        foreach (FrameRow row in frames)
            csv.Append(Csv(runId)).Append(',').Append(Csv(profileId)).Append(',').Append(N(row.elapsed)).Append(',').Append(N(row.frameUtc)).Append(',').Append(row.sequence).Append(',').Append(O(row.sourceUtc)).Append(',').Append(O(row.stateAgeMs)).Append(',').Append(N(row.frameTimeMs)).Append(',').Append(N(row.fps)).Append(',').Append(N(row.allocatedMb)).Append(',').Append(N(row.position.x)).Append(',').Append(N(row.position.y)).Append(',').Append(N(row.position.z)).Append(',').Append(O(row.positionStep)).Append(',').Append(Csv(row.freshness)).Append('\n');
        return csv.ToString();
    }

    private string EventsCsv()
    {
        StringBuilder csv = new StringBuilder("run_id,profile_id,experiment_elapsed_s,event_type,sequence_number,detail\n");
        foreach (EventRow row in events)
            csv.Append(Csv(runId)).Append(',').Append(Csv(profileId)).Append(',').Append(N(row.elapsed)).Append(',').Append(Csv(row.eventType)).Append(',').Append(row.sequence).Append(',').Append(Csv(row.detail)).Append('\n');
        return csv.ToString();
    }

    private string SummaryCsv(string reason, double duration, bool valid)
    {
        long received = transport == null || receivedStart < 0 ? -1 : Math.Max(0, transport.PacketsReceived - receivedStart);
        long published = transport == null || publishedStart < 0 ? -1 : Math.Max(0, transport.PacketsPublished - publishedStart);
        long invalid = stateManager == null ? 0 : stateManager.RejectedInvalidSnapshotCount - rejectedInvalidStart;
        long duplicates = stateManager == null ? 0 : stateManager.RejectedDuplicateSnapshotCount - rejectedDuplicateStart;
        long outOfOrder = stateManager == null ? 0 : stateManager.RejectedOutOfOrderSnapshotCount - rejectedOutOfOrderStart;
        long contractRejected = simulatedSource == null ? 0 : simulatedSource.RejectedContractSnapshotCount - rejectedContractStart;
        long malformedJson = simulatedSource == null ? 0 : simulatedSource.RejectedMalformedJsonCount - rejectedMalformedJsonStart;
        double stalePercent = frames.Count == 0 ? double.NaN : 100d * staleFrames / frames.Count;
        double meanFrame = Mean(frameTimes);
        TwinTransportDiagnostics d = diagnostics ?? new TwinTransportDiagnostics();
        string header = "run_id,profile_id,completion_reason,valid_for_analysis,target_duration_s,actual_duration_s,warmup_duration_s,packet_loss_percent,delay_ms,disconnect_at_s,disconnect_duration_s,duplicate_percent,out_of_order_percent,missing_vehicle_percent,invalid_vehicle_percent,missing_actors_percent,received_messages,transport_published_messages,applied_snapshots,rejected_contract,rejected_malformed_json,rejected_invalid,rejected_duplicate,rejected_out_of_order,stale_frame_percentage,mean_latency_ms,p95_latency_ms,mean_state_age_ms,p95_state_age_ms,mean_frame_time_ms,p95_frame_time_ms,mean_fps,recovery_elapsed_s,recovery_time_ms,maximum_recovery_jump_m,warning_count,error_count,snapshot_rows,frame_rows\n";
        string row = Csv(runId) + "," + Csv(profileId) + "," + Csv(reason) + "," + (valid ? "true" : "false") + "," + N(activeMeasurementDuration) + "," + N(duration) + "," + N(activeWarmupDuration) + "," + N(d.packetLossPercent) + "," + N(d.injectedDelayMilliseconds) + "," + N(d.disconnectAtMeasurementSeconds) + "," + N(d.disconnectDurationSeconds) + "," + N(d.duplicatePercent) + "," + N(d.outOfOrderPercent) + "," + N(d.missingVehiclePercent) + "," + N(d.invalidVehiclePercent) + "," + N(d.missingActorsPercent) + "," + received + "," + published + "," + snapshots.Count + "," + contractRejected + "," + malformedJson + "," + invalid + "," + duplicates + "," + outOfOrder + "," + O(stalePercent) + "," + O(Mean(latencies)) + "," + O(Percentile(latencies, 95d)) + "," + O(Mean(stateAges)) + "," + O(Percentile(stateAges, 95d)) + "," + O(meanFrame) + "," + O(Percentile(frameTimes, 95d)) + "," + O(meanFrame > 0d ? 1000d / meanFrame : double.NaN) + "," + O(recoveryElapsed) + "," + O(recoveryTimeMs) + "," + O(maximumRecoveryJump) + "," + warningCount + "," + errorCount + "," + snapshots.Count + "," + frames.Count + "\n";
        return header + row;
    }

    private void ResetRun()
    {
        snapshots.Clear(); frames.Clear(); events.Clear(); latencies.Clear(); stateAges.Clear(); frameTimes.Clear();
        latest = null; diagnostics = null; runId = "unidentified"; profileId = "baseline";
        lastRecordedSequence = -1; previousAcceptedSequence = -1; hasPreviousPosition = false;
        staleFrames = 0; previouslyStale = false; staleStartedElapsed = double.NaN; recoveryElapsed = double.NaN;
        recoveryTimeMs = double.NaN; recoveryJumpWindowEnd = -1d; maximumRecoveryJump = double.NaN;
        outageStartLogged = false; outageEndLogged = false;
        warningCount = 0; errorCount = 0; LastSummaryPath = null;
    }

    private void HandleLog(string condition, string trace, LogType type)
    {
        if (!IsRecording) return;
        if (type == LogType.Warning) warningCount++;
        else if (type == LogType.Error || type == LogType.Exception || type == LogType.Assert) errorCount++;
    }

    private string ResolveOutputDirectory()
    {
#if UNITY_EDITOR
        return Path.GetFullPath(Path.Combine(Application.dataPath, "..", outputFolderName));
#else
        return Path.Combine(Application.persistentDataPath, outputFolderName);
#endif
    }

    private static string SafeName(string value)
    {
        if (string.IsNullOrWhiteSpace(value)) return "unidentified";
        StringBuilder result = new StringBuilder();
        foreach (char c in value) result.Append(char.IsLetterOrDigit(c) || c == '-' || c == '_' ? c : '_');
        return result.ToString();
    }

    private static string FindAvailableBase(string directory, string preferred)
    {
        if (!File.Exists(Path.Combine(directory, preferred + "_summary.csv"))) return preferred;
        for (int index = 2; index < 1000; index++)
        {
            string candidate = preferred + "_repeat_" + index.ToString("D2", CultureInfo.InvariantCulture);
            if (!File.Exists(Path.Combine(directory, candidate + "_summary.csv"))) return candidate;
        }
        return preferred + "_" + DateTime.UtcNow.ToString("yyyyMMddHHmmss", CultureInfo.InvariantCulture);
    }

    private static double UtcNow() => (DateTime.UtcNow.Ticks - 621355968000000000L) / (double)TimeSpan.TicksPerSecond;
    private static string N(double value) => value.ToString("0.######", CultureInfo.InvariantCulture);
    private static string O(double value) => double.IsNaN(value) || double.IsInfinity(value) ? string.Empty : N(value);
    private static string Csv(string value) => "\"" + (value ?? string.Empty).Replace("\"", "\"\"") + "\"";
    private static double Mean(List<double> values)
    {
        if (values.Count == 0) return double.NaN;
        double total = 0d; foreach (double value in values) total += value; return total / values.Count;
    }
    private static double Percentile(List<double> values, double percentile)
    {
        if (values.Count == 0) return double.NaN;
        List<double> ordered = new List<double>(values); ordered.Sort();
        double index = (ordered.Count - 1) * percentile / 100d;
        int lower = (int)Math.Floor(index); int upper = Math.Min(ordered.Count - 1, lower + 1);
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower);
    }
}

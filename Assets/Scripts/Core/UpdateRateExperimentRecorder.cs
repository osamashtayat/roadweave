using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Profiling;

/// <summary>
/// Source-specific instrumentation for RoadWeave Experiment 2.
///
/// The experiment varies only the simulated TwinSnapshot publication rate.
/// Snapshot rows measure sender-to-Unity latency. Frame rows measure the age of
/// the state actually available for display, Unity frame cost, memory, and the
/// rendered ego vehicle's frame-to-frame position step.
///
/// Keep SourceExperimentRecorder disabled while this recorder is enabled so
/// Experiment 1 remains a separate, unchanged study.
/// </summary>
[DisallowMultipleComponent]
public sealed class UpdateRateExperimentRecorder : MonoBehaviour
{
    [Header("RoadWeave References")]
    [SerializeField] private DigitalTwinStateManager stateManager;
    [SerializeField] private SimulatedJsonUdpTransport transport;
    [SerializeField] private Transform displayedEgoVehicle;

    [Header("Experiment 2 Protocol")]
    [SerializeField, Min(0f)] private float warmupDurationSeconds = 30f;
    [SerializeField, Min(1f)] private float measurementDurationSeconds = 300f;
    [SerializeField, Min(1f)] private float sourceDataTimeoutSeconds = 5f;
    [SerializeField] private string outputFolderName = "ExperimentResults/experiment2";
    [SerializeField] private bool recordAutomatically = true;

    [Header("Invalid-run Detection")]
    [SerializeField] private bool detectCanonicalActorOverlap = true;
    [SerializeField] private bool stopRunOnCollision = true;
    [SerializeField] private Vector3 egoCollisionDimensionsMeters = new Vector3(1.9f, 1.5f, 4.6f);
    [SerializeField, Range(0.5f, 1f)] private float overlapEnvelopeScale = 0.9f;

    public bool IsWarmingUp { get; private set; }
    public bool IsRecording { get; private set; }
    public bool CollisionDetected { get; private set; }
    public string LastSnapshotsPath { get; private set; }
    public string LastFramesPath { get; private set; }
    public string LastSummaryPath { get; private set; }

    private readonly List<SnapshotSample> snapshotSamples = new List<SnapshotSample>();
    private readonly List<FrameSample> frameSamples = new List<FrameSample>();
    private readonly List<double> latencyMilliseconds = new List<double>();
    private readonly List<double> interArrivalMilliseconds = new List<double>();
    private readonly List<double> stateAgeMilliseconds = new List<double>();
    private readonly List<double> frameTimeMilliseconds = new List<double>();
    private readonly List<double> allocatedMemoryMegabytes = new List<double>();
    private readonly List<double> reservedMemoryMegabytes = new List<double>();
    private readonly List<double> positionStepsMeters = new List<double>();

    private double warmupStartedAt;
    private double measurementStartedAt;
    private double firstAcceptedAt = -1d;
    private double previousReceiptAt = -1d;
    private double lastSnapshotAt = -1d;
    private long firstSequence;
    private long lastSequence;
    private bool hasSequence;
    private int sequenceGapCount;
    private int nonMonotonicSequenceCount;
    private int warningCount;
    private int errorCount;
    private long transportReceivedAtStart = -1;
    private long transportPublishedAtStart = -1;
    private string activeSessionId;
    private string completedSessionId;
    private float configuredRateHz;
    private TwinSnapshot latestSnapshot;
    private Vector3 previousDisplayedPosition;
    private bool hasPreviousDisplayedPosition;
    private bool pendingCollisionStop;
    private double collisionElapsedSeconds = double.NaN;
    private string collisionActorId = string.Empty;

    private sealed class SnapshotSample
    {
        public double elapsedSeconds;
        public float configuredRateHz;
        public long sequenceNumber;
        public double sentUtcSeconds;
        public double acceptedUtcSeconds;
        public double latencyMilliseconds = double.NaN;
        public double interArrivalMilliseconds = double.NaN;
        public double sourceTimestampSeconds;
        public double receiptTimestampSeconds;
        public string freshness;
        public int actorCount;
        public Vector3 egoPosition;
        public float speedKilometersPerHour;
    }

    private sealed class FrameSample
    {
        public double elapsedSeconds;
        public float configuredRateHz;
        public double stateAgeMilliseconds = double.NaN;
        public double frameTimeMilliseconds;
        public double framesPerSecond;
        public double allocatedMemoryMegabytes;
        public double reservedMemoryMegabytes;
        public Vector3 displayedPosition;
        public double positionStepMeters = double.NaN;
        public string freshness;
    }

    private void Awake()
    {
        ResolveReferences();
    }

    private void OnEnable()
    {
        ResolveReferences();
        Subscribe();
        Application.logMessageReceived += HandleLogMessage;

        if (stateManager != null)
            HandleSessionChanged(stateManager.Session);
    }

    private void Update()
    {
        double now = Time.realtimeSinceStartup;

        if (IsWarmingUp && now - warmupStartedAt >= warmupDurationSeconds)
            BeginMeasurement(now);

        if (!IsRecording)
            return;

        RecordFrame(now);

        if (lastSnapshotAt >= 0d && now - lastSnapshotAt > sourceDataTimeoutSeconds)
        {
            CompleteRun("source_data_timeout", false);
            return;
        }

        if (pendingCollisionStop)
        {
            pendingCollisionStop = false;
            CompleteRun("collision_detected", false);
            return;
        }

        if (now - measurementStartedAt >= measurementDurationSeconds)
            CompleteRun("measurement_window_complete", true);
    }

    private void OnDisable()
    {
        if (IsRecording)
            CompleteRun("recorder_disabled_before_window_completed", false);
        else
            IsWarmingUp = false;

        Unsubscribe();
        Application.logMessageReceived -= HandleLogMessage;
    }

    private void OnDestroy()
    {
        Unsubscribe();
        Application.logMessageReceived -= HandleLogMessage;
    }

    private void ResolveReferences()
    {
        if (stateManager == null)
            stateManager = FindAnyObjectByType<DigitalTwinStateManager>();
        if (transport == null)
            transport = FindAnyObjectByType<SimulatedJsonUdpTransport>();
        if (displayedEgoVehicle == null)
        {
            GameObject ego = GameObject.Find("EgoVehicleRoot");
            if (ego != null)
                displayedEgoVehicle = ego.transform;
        }
    }

    private void Subscribe()
    {
        if (stateManager == null)
            return;

        stateManager.SnapshotUpdated -= HandleSnapshotUpdated;
        stateManager.SnapshotUpdated += HandleSnapshotUpdated;
        stateManager.SessionChanged -= HandleSessionChanged;
        stateManager.SessionChanged += HandleSessionChanged;
    }

    private void Unsubscribe()
    {
        if (stateManager == null)
            return;

        stateManager.SnapshotUpdated -= HandleSnapshotUpdated;
        stateManager.SessionChanged -= HandleSessionChanged;
    }

    private void HandleSessionChanged(TwinSessionInfo session)
    {
        if (session == null || session.sourceKind != TwinSourceKind.SimulatedStream)
            return;

        string sessionId = session.sessionId ?? string.Empty;
        if (session.status == TwinSessionStatus.Running)
        {
            if (recordAutomatically &&
                !IsWarmingUp &&
                !IsRecording &&
                !string.Equals(sessionId, completedSessionId, StringComparison.Ordinal))
            {
                BeginWarmup(sessionId);
            }
            return;
        }

        if (IsRecording)
        {
            string reason = session.status == TwinSessionStatus.Error
                ? "source_session_error"
                : session.status == TwinSessionStatus.Disconnected
                    ? "source_disconnected"
                    : "source_stopped_before_window_completed";
            CompleteRun(reason, false);
        }
        else if (IsWarmingUp)
        {
            IsWarmingUp = false;
            Debug.LogWarning("RoadWeave Experiment 2 warm-up cancelled because the simulated source stopped running.");
        }
    }

    private void BeginWarmup(string sessionId)
    {
        ResetRunState();
        activeSessionId = sessionId;
        warmupStartedAt = Time.realtimeSinceStartup;
        IsWarmingUp = true;

        Debug.Log(
            $"RoadWeave Experiment 2 warm-up started for session {activeSessionId}. " +
            $"Measurement begins automatically in {warmupDurationSeconds:F0} seconds.");

        if (warmupDurationSeconds <= 0f)
            BeginMeasurement(warmupStartedAt);
    }

    private void BeginMeasurement(double now)
    {
        if (!IsWarmingUp)
            return;

        IsWarmingUp = false;
        IsRecording = true;
        measurementStartedAt = now;
        lastSnapshotAt = now;
        transportReceivedAtStart = transport == null ? -1 : transport.PacketsReceived;
        transportPublishedAtStart = transport == null ? -1 : transport.PacketsPublished;

        if (displayedEgoVehicle != null)
        {
            previousDisplayedPosition = displayedEgoVehicle.position;
            hasPreviousDisplayedPosition = true;
        }

        string rateText = configuredRateHz > 0f ? $"{configuredRateHz:F1} Hz" : "rate pending";
        Debug.Log(
            $"RoadWeave Experiment 2 measurement started: {rateText}, " +
            $"{measurementDurationSeconds:F0} seconds. Do not interact with the application.");
    }

    private void ResetRunState()
    {
        snapshotSamples.Clear();
        frameSamples.Clear();
        latencyMilliseconds.Clear();
        interArrivalMilliseconds.Clear();
        stateAgeMilliseconds.Clear();
        frameTimeMilliseconds.Clear();
        allocatedMemoryMegabytes.Clear();
        reservedMemoryMegabytes.Clear();
        positionStepsMeters.Clear();

        firstAcceptedAt = -1d;
        previousReceiptAt = -1d;
        lastSnapshotAt = -1d;
        firstSequence = 0;
        lastSequence = 0;
        hasSequence = false;
        sequenceGapCount = 0;
        nonMonotonicSequenceCount = 0;
        warningCount = 0;
        errorCount = 0;
        configuredRateHz = 0f;
        latestSnapshot = null;
        hasPreviousDisplayedPosition = false;
        CollisionDetected = false;
        pendingCollisionStop = false;
        collisionElapsedSeconds = double.NaN;
        collisionActorId = string.Empty;
        LastSnapshotsPath = null;
        LastFramesPath = null;
        LastSummaryPath = null;
    }

    private void HandleSnapshotUpdated(TwinSnapshot snapshot)
    {
        if (snapshot?.metadata?.session == null ||
            snapshot.metadata.session.sourceKind != TwinSourceKind.SimulatedStream ||
            !string.Equals(snapshot.metadata.session.sessionId, activeSessionId, StringComparison.Ordinal))
        {
            return;
        }

        latestSnapshot = snapshot;
        if (snapshot.metadata.nominalUpdateRateHz > 0f)
            configuredRateHz = snapshot.metadata.nominalUpdateRateHz;

        if (!IsRecording)
            return;

        double acceptedUtcSeconds = UtcNowSeconds();
        double latency = double.NaN;
        if (snapshot.metadata.sentTimestampUtcSeconds > 0d)
        {
            latency = (acceptedUtcSeconds - snapshot.metadata.sentTimestampUtcSeconds) * 1000d;
            if (latency >= 0d && latency < 60000d)
                latencyMilliseconds.Add(latency);
            else
                latency = double.NaN;
        }

        double interArrival = double.NaN;
        double receipt = snapshot.metadata.receiptTimestampSeconds;
        if (previousReceiptAt >= 0d && receipt >= previousReceiptAt)
        {
            interArrival = (receipt - previousReceiptAt) * 1000d;
            interArrivalMilliseconds.Add(interArrival);
        }

        long sequence = snapshot.metadata.sequenceNumber;
        if (!hasSequence)
        {
            firstSequence = sequence;
            hasSequence = true;
        }
        else if (sequence > lastSequence + 1)
        {
            long missing = sequence - lastSequence - 1;
            sequenceGapCount += missing > int.MaxValue ? int.MaxValue : (int)missing;
        }
        else if (sequence <= lastSequence)
        {
            nonMonotonicSequenceCount++;
        }
        lastSequence = sequence;

        double now = Time.realtimeSinceStartup;
        lastSnapshotAt = now;
        if (firstAcceptedAt < 0d)
            firstAcceptedAt = now;

        float speedKilometersPerHour = snapshot.vehicle != null &&
                                        snapshot.vehicle.validity != TwinDataValidity.Invalid
            ? snapshot.vehicle.speedKilometersPerHour
            : (snapshot.ego?.speedMetersPerSecond ?? 0f) * 3.6f;

        snapshotSamples.Add(new SnapshotSample
        {
            elapsedSeconds = now - measurementStartedAt,
            configuredRateHz = configuredRateHz,
            sequenceNumber = sequence,
            sentUtcSeconds = snapshot.metadata.sentTimestampUtcSeconds,
            acceptedUtcSeconds = acceptedUtcSeconds,
            latencyMilliseconds = latency,
            interArrivalMilliseconds = interArrival,
            sourceTimestampSeconds = snapshot.metadata.sourceTimestampSeconds,
            receiptTimestampSeconds = receipt,
            freshness = snapshot.metadata.freshness.ToString(),
            actorCount = snapshot.actors?.Length ?? 0,
            egoPosition = snapshot.ego?.position ?? Vector3.zero,
            speedKilometersPerHour = speedKilometersPerHour
        });

        previousReceiptAt = receipt;

        if (!CollisionDetected &&
            detectCanonicalActorOverlap &&
            TryFindOverlappingActor(snapshot, out string actorId))
        {
            CollisionDetected = true;
            collisionElapsedSeconds = now - measurementStartedAt;
            collisionActorId = actorId;
            Debug.LogError(
                $"RoadWeave Experiment 2 detected an ego/actor overlap with '{actorId}' " +
                $"at {collisionElapsedSeconds:F3} seconds. This run is invalid.");
            if (stopRunOnCollision)
                pendingCollisionStop = true;
        }
    }

    private void RecordFrame(double now)
    {
        double frameMilliseconds = Time.unscaledDeltaTime * 1000d;
        if (frameMilliseconds <= 0d)
            return;

        double stateAge = double.NaN;
        if (latestSnapshot?.metadata != null && latestSnapshot.metadata.sentTimestampUtcSeconds > 0d)
        {
            stateAge = (UtcNowSeconds() - latestSnapshot.metadata.sentTimestampUtcSeconds) * 1000d;
            if (stateAge >= 0d && stateAge < 60000d)
                stateAgeMilliseconds.Add(stateAge);
            else
                stateAge = double.NaN;
        }

        double allocatedMemory = Profiler.GetTotalAllocatedMemoryLong() / (1024d * 1024d);
        double reservedMemory = Profiler.GetTotalReservedMemoryLong() / (1024d * 1024d);
        Vector3 displayedPosition = displayedEgoVehicle == null
            ? (latestSnapshot?.ego?.position ?? Vector3.zero)
            : displayedEgoVehicle.position;
        double positionStep = double.NaN;
        if (hasPreviousDisplayedPosition)
        {
            positionStep = Vector3.Distance(previousDisplayedPosition, displayedPosition);
            positionStepsMeters.Add(positionStep);
        }
        previousDisplayedPosition = displayedPosition;
        hasPreviousDisplayedPosition = true;

        frameTimeMilliseconds.Add(frameMilliseconds);
        allocatedMemoryMegabytes.Add(allocatedMemory);
        reservedMemoryMegabytes.Add(reservedMemory);

        frameSamples.Add(new FrameSample
        {
            elapsedSeconds = now - measurementStartedAt,
            configuredRateHz = configuredRateHz,
            stateAgeMilliseconds = stateAge,
            frameTimeMilliseconds = frameMilliseconds,
            framesPerSecond = 1000d / frameMilliseconds,
            allocatedMemoryMegabytes = allocatedMemory,
            reservedMemoryMegabytes = reservedMemory,
            displayedPosition = displayedPosition,
            positionStepMeters = positionStep,
            freshness = latestSnapshot?.metadata == null
                ? TwinDataFreshness.Unknown.ToString()
                : latestSnapshot.metadata.freshness.ToString()
        });
    }

    private bool TryFindOverlappingActor(TwinSnapshot snapshot, out string actorId)
    {
        actorId = string.Empty;
        if (snapshot?.ego == null || snapshot.actors == null)
            return false;

        Quaternion worldToEgo = Quaternion.Euler(0f, -snapshot.ego.yawDegrees, 0f);
        float egoHalfWidth = Mathf.Max(0.5f, egoCollisionDimensionsMeters.x * 0.5f);
        float egoHalfLength = Mathf.Max(1f, egoCollisionDimensionsMeters.z * 0.5f);
        float scale = Mathf.Clamp(overlapEnvelopeScale, 0.5f, 1f);

        foreach (TwinActorState actor in snapshot.actors)
        {
            if (actor == null || actor.validity == TwinDataValidity.Invalid)
                continue;
            if (actor.semanticClass == TwinActorClass.TrafficLight)
                continue;

            Vector3 localDelta = worldToEgo * (actor.position - snapshot.ego.position);
            if (Mathf.Abs(localDelta.y) > 3f)
                continue;

            Vector3 dimensions = actor.dimensionsMeters;
            float actorHalfWidth = dimensions.x > 0.1f ? dimensions.x * 0.5f : 0.5f;
            float actorHalfLength = dimensions.z > 0.1f ? dimensions.z * 0.5f : 0.5f;
            float combinedHalfWidth = (egoHalfWidth + actorHalfWidth) * scale;
            float combinedHalfLength = (egoHalfLength + actorHalfLength) * scale;

            if (Mathf.Abs(localDelta.x) <= combinedHalfWidth &&
                Mathf.Abs(localDelta.z) <= combinedHalfLength)
            {
                actorId = string.IsNullOrWhiteSpace(actor.id)
                    ? actor.semanticClass.ToString()
                    : actor.id;
                return true;
            }
        }

        return false;
    }

    private void CompleteRun(string completionReason, bool nominalCompletion)
    {
        if (!IsRecording)
            return;

        IsRecording = false;
        IsWarmingUp = false;

        double finishedAt = Time.realtimeSinceStartup;
        double actualDuration = Math.Max(0d, finishedAt - measurementStartedAt);
        long receivedMessages = transport == null || transportReceivedAtStart < 0
            ? -1
            : Math.Max(0L, transport.PacketsReceived - transportReceivedAtStart);
        long publishedMessages = transport == null || transportPublishedAtStart < 0
            ? -1
            : Math.Max(0L, transport.PacketsPublished - transportPublishedAtStart);
        long scheduledMessages = hasSequence ? Math.Max(1L, lastSequence - firstSequence + 1L) : 0L;
        long appliedSnapshots = snapshotSamples.Count;
        long droppedMessages = Math.Max(0L, scheduledMessages - appliedSnapshots);
        bool completedWithoutError = nominalCompletion &&
                                     appliedSnapshots > 0 &&
                                     errorCount == 0 &&
                                     !CollisionDetected;
        bool validForAnalysis = completedWithoutError &&
                                actualDuration >= measurementDurationSeconds - 0.25d;

        string outputDirectory = ResolveOutputDirectory();
        Directory.CreateDirectory(outputDirectory);
        string rateSlug = RateSlug(configuredRateHz);
        int runNumber = FindNextRunNumber(outputDirectory, rateSlug);
        string runId = $"{rateSlug}_run_{runNumber:D3}";
        string baseName = $"experiment2_{runId}";
        LastSnapshotsPath = Path.Combine(outputDirectory, baseName + "_snapshots.csv");
        LastFramesPath = Path.Combine(outputDirectory, baseName + "_frames.csv");
        LastSummaryPath = Path.Combine(outputDirectory, baseName + "_summary.csv");

        try
        {
            File.WriteAllText(LastSnapshotsPath, BuildSnapshotsCsv(runId), new UTF8Encoding(false));
            File.WriteAllText(LastFramesPath, BuildFramesCsv(runId), new UTF8Encoding(false));
            File.WriteAllText(
                LastSummaryPath,
                BuildSummaryCsv(
                    runId,
                    completionReason,
                    actualDuration,
                    scheduledMessages,
                    receivedMessages,
                    publishedMessages,
                    appliedSnapshots,
                    droppedMessages,
                    completedWithoutError,
                    validForAnalysis),
                new UTF8Encoding(false));

            Debug.Log(
                $"RoadWeave Experiment 2 CSV files saved. Valid for analysis: {validForAnalysis}.\n" +
                LastSnapshotsPath + "\n" + LastFramesPath + "\n" + LastSummaryPath);
        }
        catch (Exception exception)
        {
            Debug.LogError("Could not save RoadWeave Experiment 2 CSV files: " + exception.Message);
        }

        completedSessionId = activeSessionId;
    }

    private string BuildSnapshotsCsv(string runId)
    {
        StringBuilder csv = new StringBuilder();
        csv.AppendLine(
            "run_id,configured_rate_hz,experiment_elapsed_s,sequence_number,sent_utc_s," +
            "accepted_utc_s,latency_ms,inter_arrival_ms,source_timestamp_s,receipt_timestamp_s," +
            "freshness,actor_count,ego_x_m,ego_y_m,ego_z_m,speed_kph");

        foreach (SnapshotSample sample in snapshotSamples)
        {
            csv.Append(Csv(runId)).Append(',')
                .Append(Number(sample.configuredRateHz)).Append(',')
                .Append(Number(sample.elapsedSeconds)).Append(',')
                .Append(sample.sequenceNumber.ToString(CultureInfo.InvariantCulture)).Append(',')
                .Append(OptionalNumber(sample.sentUtcSeconds)).Append(',')
                .Append(OptionalNumber(sample.acceptedUtcSeconds)).Append(',')
                .Append(OptionalNumber(sample.latencyMilliseconds)).Append(',')
                .Append(OptionalNumber(sample.interArrivalMilliseconds)).Append(',')
                .Append(Number(sample.sourceTimestampSeconds)).Append(',')
                .Append(Number(sample.receiptTimestampSeconds)).Append(',')
                .Append(Csv(sample.freshness)).Append(',')
                .Append(sample.actorCount.ToString(CultureInfo.InvariantCulture)).Append(',')
                .Append(Number(sample.egoPosition.x)).Append(',')
                .Append(Number(sample.egoPosition.y)).Append(',')
                .Append(Number(sample.egoPosition.z)).Append(',')
                .Append(Number(sample.speedKilometersPerHour)).AppendLine();
        }

        return csv.ToString();
    }

    private string BuildFramesCsv(string runId)
    {
        StringBuilder csv = new StringBuilder();
        csv.AppendLine(
            "run_id,configured_rate_hz,experiment_elapsed_s,state_age_ms,frame_time_ms,fps," +
            "allocated_memory_mb,reserved_memory_mb,displayed_x_m,displayed_y_m,displayed_z_m," +
            "frame_position_step_m,freshness");

        foreach (FrameSample sample in frameSamples)
        {
            csv.Append(Csv(runId)).Append(',')
                .Append(Number(sample.configuredRateHz)).Append(',')
                .Append(Number(sample.elapsedSeconds)).Append(',')
                .Append(OptionalNumber(sample.stateAgeMilliseconds)).Append(',')
                .Append(Number(sample.frameTimeMilliseconds)).Append(',')
                .Append(Number(sample.framesPerSecond)).Append(',')
                .Append(Number(sample.allocatedMemoryMegabytes)).Append(',')
                .Append(Number(sample.reservedMemoryMegabytes)).Append(',')
                .Append(Number(sample.displayedPosition.x)).Append(',')
                .Append(Number(sample.displayedPosition.y)).Append(',')
                .Append(Number(sample.displayedPosition.z)).Append(',')
                .Append(OptionalNumber(sample.positionStepMeters)).Append(',')
                .Append(Csv(sample.freshness)).AppendLine();
        }

        return csv.ToString();
    }

    private string BuildSummaryCsv(
        string runId,
        string completionReason,
        double actualDuration,
        long scheduledMessages,
        long receivedMessages,
        long publishedMessages,
        long appliedSnapshots,
        long droppedMessages,
        bool completedWithoutError,
        bool validForAnalysis)
    {
        double deliveryPercentage = scheduledMessages <= 0 || receivedMessages < 0
            ? double.NaN
            : 100d * receivedMessages / scheduledMessages;
        double applicationPercentage = receivedMessages <= 0
            ? double.NaN
            : 100d * appliedSnapshots / receivedMessages;
        double effectiveAppliedRate = actualDuration <= 0d
            ? 0d
            : appliedSnapshots / actualDuration;
        double meanFrameTime = Mean(frameTimeMilliseconds);
        double meanFps = meanFrameTime > 0d ? 1000d / meanFrameTime : double.NaN;
        double positionStepMean = Mean(positionStepsMeters);
        double positionStepSd = StandardDeviation(positionStepsMeters);
        double positionStepCv = positionStepMean > 0d
            ? positionStepSd / positionStepMean
            : double.NaN;

        StringBuilder csv = new StringBuilder();
        csv.AppendLine(
            "run_id,configured_rate_hz,completion_reason,completed_without_error,valid_for_analysis," +
            "target_duration_s,actual_duration_s,warmup_duration_s,scheduled_messages,received_messages," +
            "transport_published_messages,applied_snapshots,total_dropped,delivery_percentage," +
            "application_percentage,effective_applied_rate_hz,sequence_gap_count," +
            "non_monotonic_sequence_count,mean_latency_ms,median_latency_ms,p95_latency_ms," +
            "p99_latency_ms,maximum_latency_ms,mean_state_age_ms,median_state_age_ms,p95_state_age_ms," +
            "p99_state_age_ms,maximum_state_age_ms,mean_frame_time_ms,p95_frame_time_ms,mean_fps," +
            "allocated_memory_mean_mb,allocated_memory_max_mb,reserved_memory_mean_mb," +
            "reserved_memory_max_mb,position_step_mean_m,position_step_sd_m,position_step_cv," +
            "p95_position_step_m,maximum_position_step_m,collision_detected,collision_elapsed_s," +
            "collision_actor_id,warning_count,error_count,snapshot_rows,frame_rows");
        csv.Append(Csv(runId)).Append(',')
            .Append(Number(configuredRateHz)).Append(',')
            .Append(Csv(completionReason)).Append(',')
            .Append(completedWithoutError ? "true" : "false").Append(',')
            .Append(validForAnalysis ? "true" : "false").Append(',')
            .Append(Number(measurementDurationSeconds)).Append(',')
            .Append(Number(actualDuration)).Append(',')
            .Append(Number(warmupDurationSeconds)).Append(',')
            .Append(scheduledMessages.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(receivedMessages >= 0 ? receivedMessages.ToString(CultureInfo.InvariantCulture) : string.Empty).Append(',')
            .Append(publishedMessages >= 0 ? publishedMessages.ToString(CultureInfo.InvariantCulture) : string.Empty).Append(',')
            .Append(appliedSnapshots.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(droppedMessages.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(OptionalNumber(deliveryPercentage)).Append(',')
            .Append(OptionalNumber(applicationPercentage)).Append(',')
            .Append(Number(effectiveAppliedRate)).Append(',')
            .Append(sequenceGapCount.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(nonMonotonicSequenceCount.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(OptionalNumber(Mean(latencyMilliseconds))).Append(',')
            .Append(OptionalNumber(Percentile(latencyMilliseconds, 50d))).Append(',')
            .Append(OptionalNumber(Percentile(latencyMilliseconds, 95d))).Append(',')
            .Append(OptionalNumber(Percentile(latencyMilliseconds, 99d))).Append(',')
            .Append(OptionalNumber(Maximum(latencyMilliseconds))).Append(',')
            .Append(OptionalNumber(Mean(stateAgeMilliseconds))).Append(',')
            .Append(OptionalNumber(Percentile(stateAgeMilliseconds, 50d))).Append(',')
            .Append(OptionalNumber(Percentile(stateAgeMilliseconds, 95d))).Append(',')
            .Append(OptionalNumber(Percentile(stateAgeMilliseconds, 99d))).Append(',')
            .Append(OptionalNumber(Maximum(stateAgeMilliseconds))).Append(',')
            .Append(OptionalNumber(meanFrameTime)).Append(',')
            .Append(OptionalNumber(Percentile(frameTimeMilliseconds, 95d))).Append(',')
            .Append(OptionalNumber(meanFps)).Append(',')
            .Append(OptionalNumber(Mean(allocatedMemoryMegabytes))).Append(',')
            .Append(OptionalNumber(Maximum(allocatedMemoryMegabytes))).Append(',')
            .Append(OptionalNumber(Mean(reservedMemoryMegabytes))).Append(',')
            .Append(OptionalNumber(Maximum(reservedMemoryMegabytes))).Append(',')
            .Append(OptionalNumber(positionStepMean)).Append(',')
            .Append(OptionalNumber(positionStepSd)).Append(',')
            .Append(OptionalNumber(positionStepCv)).Append(',')
            .Append(OptionalNumber(Percentile(positionStepsMeters, 95d))).Append(',')
            .Append(OptionalNumber(Maximum(positionStepsMeters))).Append(',')
            .Append(CollisionDetected ? "true" : "false").Append(',')
            .Append(OptionalNumber(collisionElapsedSeconds)).Append(',')
            .Append(Csv(collisionActorId)).Append(',')
            .Append(warningCount.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(errorCount.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(snapshotSamples.Count.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(frameSamples.Count.ToString(CultureInfo.InvariantCulture)).AppendLine();

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
    }

    private string ResolveOutputDirectory()
    {
        string folder = string.IsNullOrWhiteSpace(outputFolderName)
            ? "ExperimentResults/experiment2"
            : outputFolderName.Trim();
#if UNITY_EDITOR
        return Path.GetFullPath(Path.Combine(Application.dataPath, "..", folder));
#else
        return Path.Combine(Application.persistentDataPath, folder);
#endif
    }

    private static string RateSlug(float rateHz)
    {
        int rounded = Mathf.Max(0, Mathf.RoundToInt(rateHz));
        return rounded > 0 ? $"{rounded:D3}hz" : "unknownhz";
    }

    private static int FindNextRunNumber(string directory, string rateSlug)
    {
        if (!Directory.Exists(directory))
            return 1;

        string prefix = $"experiment2_{rateSlug}_run_";
        int maximum = 0;
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

    private static double UtcNowSeconds()
    {
        const long unixEpochTicks = 621355968000000000L;
        return (DateTime.UtcNow.Ticks - unixEpochTicks) / (double)TimeSpan.TicksPerSecond;
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

    private static double StandardDeviation(List<double> values)
    {
        if (values.Count < 2)
            return double.NaN;
        double mean = Mean(values);
        double sum = 0d;
        foreach (double value in values)
        {
            double difference = value - mean;
            sum += difference * difference;
        }
        return Math.Sqrt(sum / (values.Count - 1));
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

    private static string Number(double value)
    {
        return value.ToString("0.######", CultureInfo.InvariantCulture);
    }

    private static string OptionalNumber(double value)
    {
        return double.IsNaN(value) || double.IsInfinity(value) ? string.Empty : Number(value);
    }

    private static string Csv(string value)
    {
        string safe = value ?? string.Empty;
        if (safe.IndexOfAny(new[] { ',', '"', '\n', '\r' }) < 0)
            return safe;
        return "\"" + safe.Replace("\"", "\"\"") + "\"";
    }
}

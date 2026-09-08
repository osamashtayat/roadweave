using System;
using UnityEngine;

/// <summary>
/// Source-neutral current-state store. Every Unity presentation component reads
/// the same accepted TwinSnapshot regardless of its source.
/// </summary>
public class DigitalTwinStateManager : MonoBehaviour, ITwinSnapshotSink
{
    public static DigitalTwinStateManager Instance { get; private set; }

    [Header("Streaming Safety")]
    [SerializeField, Min(0.05f)] private float defaultStaleAfterSeconds = 1f;
    [SerializeField] private bool rejectOutOfOrderSnapshots = true;

    // Transitional replay-only access for components not yet migrated.
    public ReplayPackage Package { get; private set; }

    public TwinSnapshot CurrentSnapshot { get; private set; }
    public TwinEgoState Ego => CurrentSnapshot?.ego ?? emptyEgo;
    public TwinVehicleState Vehicle => CurrentSnapshot?.vehicle ?? emptyVehicle;
    public TwinWheelState Wheels => CurrentSnapshot?.wheels ?? emptyWheels;
    public TwinActorState[] Actors => CurrentSnapshot?.actors ?? emptyActors;
    public TwinSnapshotMetadata Metadata => CurrentSnapshot?.metadata;
    public TwinSessionInfo Session { get; private set; } = new TwinSessionInfo
    {
        status = TwinSessionStatus.Disconnected,
        sourceKind = TwinSourceKind.Unknown
    };

    public ReplayStatus Status { get; private set; } = ReplayStatus.Loading;
    public float CurrentTime => CurrentSnapshot == null ? 0f : (float)CurrentSnapshot.metadata.sourceTimestampSeconds;
    /// <summary>
    /// Source time without the precision loss of the legacy float CurrentTime API.
    /// Use this for integration/delta calculations, especially with Unix timestamps.
    /// </summary>
    public double CurrentSourceTimestampSeconds =>
        CurrentSnapshot?.metadata == null ? 0d : CurrentSnapshot.metadata.sourceTimestampSeconds;
    public bool IsReady => CurrentSnapshot != null &&
                           CurrentSnapshot.metadata.validity != TwinDataValidity.Invalid &&
                           Session.status != TwinSessionStatus.Error;
    public bool IsPlaying => Session.status == TwinSessionStatus.Running;
    public bool IsFresh => CurrentSnapshot != null && CurrentSnapshot.metadata.freshness == TwinDataFreshness.Fresh;
    public ITwinSessionControl SessionControl => connectedSource as ITwinSessionControl;
    public ITwinStateSource ConnectedSource => connectedSource;
    public long RejectedInvalidSnapshotCount { get; private set; }
    public long RejectedDuplicateSnapshotCount { get; private set; }
    public long RejectedOutOfOrderSnapshotCount { get; private set; }

    public event Action Initialized;
    public event Action StateUpdated;
    public event Action<TwinSnapshot> SnapshotUpdated;
    public event Action<ReplayStatus> StatusChanged;
    public event Action<TwinSessionInfo> SessionChanged;
    public event Action<ITwinStateSource> SourceChanged;
    public event Action<TwinSnapshot, string> SnapshotRejected;

    private static readonly TwinEgoState emptyEgo = new TwinEgoState();
    private static readonly TwinVehicleState emptyVehicle = new TwinVehicleState();
    private static readonly TwinWheelState emptyWheels = new TwinWheelState();
    private static readonly TwinActorState[] emptyActors = Array.Empty<TwinActorState>();
    private ITwinStateSource connectedSource;
    private string acceptedSessionId;
    private long acceptedSequence = -1;
    private bool initializedRaised;

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

    private void Update() => RefreshFreshness();

    private void OnDestroy()
    {
        DisconnectSource();
        if (Instance == this) Instance = null;
    }

    public void ConnectSource(ITwinStateSource source)
    {
        if (ReferenceEquals(connectedSource, source)) return;
        DetachConnectedSource(false);
        connectedSource = source;
        ResetAcceptedSourceState(true);
        if (connectedSource == null)
        {
            PublishDisconnectedSession();
            SourceChanged?.Invoke(null);
            return;
        }
        connectedSource.SnapshotProduced += OnSourceSnapshot;
        connectedSource.SessionChanged += PublishSessionStatus;
        if (connectedSource.Session != null) PublishSessionStatus(connectedSource.Session);
        SourceChanged?.Invoke(connectedSource);
    }

    public void DisconnectSource()
    {
        DetachConnectedSource(true);
    }

    public void DisconnectSource(ITwinStateSource source)
    {
        if (ReferenceEquals(connectedSource, source))
            DisconnectSource();
    }

    private void OnSourceSnapshot(TwinSnapshot snapshot) => PublishSnapshot(snapshot);

    public bool PublishSnapshot(TwinSnapshot snapshot)
    {
        if (!ValidateSnapshot(snapshot, out string validationError))
        {
            RejectedInvalidSnapshotCount++;
            SnapshotRejected?.Invoke(snapshot, "invalid:" + validationError);
            Debug.LogWarning($"Rejected digital-twin snapshot: {validationError}");
            return false;
        }

        string sessionId = snapshot.metadata.session.sessionId ?? string.Empty;
        if (!string.Equals(sessionId, acceptedSessionId, StringComparison.Ordinal))
        {
            acceptedSessionId = sessionId;
            acceptedSequence = -1;
            initializedRaised = false;
        }

        if (rejectOutOfOrderSnapshots && snapshot.metadata.sequenceNumber <= acceptedSequence)
        {
            bool duplicate = snapshot.metadata.sequenceNumber == acceptedSequence;
            if (duplicate)
                RejectedDuplicateSnapshotCount++;
            else
                RejectedOutOfOrderSnapshotCount++;
            SnapshotRejected?.Invoke(snapshot, duplicate ? "duplicate_sequence" : "out_of_order_sequence");
            Debug.LogWarning($"Rejected out-of-order snapshot {snapshot.metadata.sequenceNumber}; last accepted is {acceptedSequence}.");
            return false;
        }

        snapshot.metadata.receiptTimestampSeconds = Time.realtimeSinceStartup;
        if (snapshot.metadata.staleAfterSeconds <= 0f) snapshot.metadata.staleAfterSeconds = defaultStaleAfterSeconds;
        // Receipt of a message can establish freshness only when the source did
        // not already declare it stale. Never relabel upstream-stale data Fresh.
        if (snapshot.metadata.freshness == TwinDataFreshness.Unknown)
            snapshot.metadata.freshness = TwinDataFreshness.Fresh;
        snapshot.actors = snapshot.actors ?? emptyActors;
        acceptedSequence = snapshot.metadata.sequenceNumber;
        CurrentSnapshot = snapshot;
        PublishSessionStatus(snapshot.metadata.session);

        if (!initializedRaised)
        {
            initializedRaised = true;
            Initialized?.Invoke();
        }
        StateUpdated?.Invoke();
        SnapshotUpdated?.Invoke(CurrentSnapshot);
        return true;
    }

    public void PublishSessionStatus(TwinSessionInfo session)
    {
        if (session == null) return;
        Session = session;
        ReplayStatus compatibilityStatus = ToReplayStatus(session.status);
        bool changed = Status != compatibilityStatus;
        Status = compatibilityStatus;
        SessionChanged?.Invoke(Session);
        if (changed) StatusChanged?.Invoke(Status);
    }

    /// <summary>
    /// Attaches replay-only compatibility data only when its owner is the active
    /// replay source. This prevents a replay finishing an asynchronous load from
    /// contaminating an active simulated/live session.
    /// </summary>
    public bool AttachReplayPackage(ReplayPackage replayPackage, ITwinStateSource owner)
    {
        if (replayPackage == null || owner == null ||
            !ReferenceEquals(connectedSource, owner) ||
            owner.Session?.sourceKind != TwinSourceKind.Replay)
            return false;
        Package = replayPackage;
        return true;
    }

    // Compatibility entry point for existing scenes. New replay code samples in its adapter.
    public void Initialize(ReplayPackage replayPackage)
    {
        if (connectedSource != null && connectedSource.Session?.sourceKind != TwinSourceKind.Replay)
        {
            Debug.LogWarning("Ignored legacy replay initialization while a non-replay source is active.");
            return;
        }
        if (!ReplaySnapshotSampler.ValidatePackage(replayPackage, out TwinCoordinateFrame frame, out string error))
        {
            Debug.LogError(error);
            SetError();
            return;
        }
        // Explicit legacy entry point: there is no source object to own the
        // package, so keep this compatibility path local to the state manager.
        Package = replayPackage;
        TwinSessionInfo replaySession = CreateCompatibilityReplaySession(replayPackage, TwinSessionStatus.Ready);
        acceptedSessionId = null;
        acceptedSequence = -1;
        PublishSnapshot(ReplaySnapshotSampler.Sample(replayPackage, frame, 0f, 0, replaySession));
    }

    // Compatibility sampling for older callers.
    public void UpdateState(float replayTime)
    {
        string error = Package == null ? "No compatibility replay package is attached." : null;
        TwinCoordinateFrame frame = null;
        if (Package == null || !ReplaySnapshotSampler.ValidatePackage(Package, out frame, out error))
        {
            if (!string.IsNullOrEmpty(error)) Debug.LogWarning(error);
            return;
        }
        TwinSessionInfo session = Session.sourceKind == TwinSourceKind.Replay
            ? Session
            : CreateCompatibilityReplaySession(Package, TwinSessionStatus.Ready);
        PublishSnapshot(ReplaySnapshotSampler.Sample(Package, frame, replayTime, acceptedSequence + 1, session));
    }

    public void SetStatus(ReplayStatus newStatus)
    {
        TwinSessionInfo next = CopySession(Session);
        next.status = FromReplayStatus(newStatus);
        PublishSessionStatus(next);
        if (CurrentSnapshot?.metadata != null) CurrentSnapshot.metadata.session = CopySession(next);
    }

    public void SetError()
    {
        TwinSessionInfo next = CopySession(Session);
        next.status = TwinSessionStatus.Error;
        if (string.IsNullOrEmpty(next.statusMessage)) next.statusMessage = "The active source reported an error.";
        PublishSessionStatus(next);
        if (CurrentSnapshot != null)
        {
            CurrentSnapshot.metadata.validity = TwinDataValidity.Invalid;
            CurrentSnapshot.metadata.validityMessage = next.statusMessage;
        }
    }

    private void RefreshFreshness()
    {
        if (CurrentSnapshot?.metadata == null || CurrentSnapshot.metadata.freshness == TwinDataFreshness.Stale) return;
        // Replay remains deterministic when paused. Streaming data must age out,
        // including after a live/simulated source is paused or stopped.
        if (Session.sourceKind == TwinSourceKind.Replay) return;
        double age = Time.realtimeSinceStartup - CurrentSnapshot.metadata.receiptTimestampSeconds;
        if (age <= CurrentSnapshot.metadata.staleAfterSeconds) return;
        CurrentSnapshot.metadata.freshness = TwinDataFreshness.Stale;
        foreach (TwinActorState actor in CurrentSnapshot.actors ?? emptyActors)
        {
            if (actor != null)
                actor.freshness = TwinDataFreshness.Stale;
        }
        StateUpdated?.Invoke();
        SnapshotUpdated?.Invoke(CurrentSnapshot);
    }

    private void DetachConnectedSource(bool publishDisconnected)
    {
        if (connectedSource != null)
        {
            connectedSource.SnapshotProduced -= OnSourceSnapshot;
            connectedSource.SessionChanged -= PublishSessionStatus;
            connectedSource = null;
        }
        if (!publishDisconnected)
            return;
        ResetAcceptedSourceState(true);
        PublishDisconnectedSession();
        SourceChanged?.Invoke(null);
    }

    private void ResetAcceptedSourceState(bool publishUnavailable)
    {
        acceptedSessionId = null;
        acceptedSequence = -1;
        initializedRaised = false;
        CurrentSnapshot = null;
        Package = null;
        if (publishUnavailable)
        {
            StateUpdated?.Invoke();
            SnapshotUpdated?.Invoke(null);
        }
    }

    private void PublishDisconnectedSession()
    {
        PublishSessionStatus(new TwinSessionInfo
        {
            sourceKind = TwinSourceKind.Unknown,
            status = TwinSessionStatus.Disconnected,
            statusMessage = "No digital-twin source is connected."
        });
    }

    private static bool ValidateSnapshot(TwinSnapshot snapshot, out string error)
    {
        if (snapshot == null) { error = "Snapshot is null."; return false; }
        if (snapshot.metadata == null) { error = "Metadata is missing."; return false; }
        if (snapshot.metadata.session == null) { error = "Session metadata is missing."; return false; }
        if (string.IsNullOrWhiteSpace(snapshot.metadata.session.sessionId)) { error = "Session ID is required."; return false; }
        if (!TwinCoordinateFrameService.IsCanonical(snapshot.metadata.coordinateFrame)) { error = "Snapshot is not in the canonical RoadWeave coordinate frame."; return false; }
        if (snapshot.ego == null || snapshot.vehicle == null || snapshot.wheels == null) { error = "Ego, vehicle, and wheel groups must be present; mark unavailable groups Invalid."; return false; }
        if (snapshot.metadata.validity == TwinDataValidity.Unknown) { error = "Snapshot validity must be explicitly Valid or Partial."; return false; }
        if (snapshot.metadata.validity == TwinDataValidity.Invalid) { error = snapshot.metadata.validityMessage ?? "Source marked the snapshot invalid."; return false; }
        if (!HasExplicitValidity(snapshot.ego.validity) ||
            !HasExplicitValidity(snapshot.vehicle.validity) ||
            !HasExplicitValidity(snapshot.wheels.validity))
        {
            error = "Ego, vehicle, and wheel validity must each be explicitly Valid, Partial, or Invalid.";
            return false;
        }
        if (snapshot.metadata.validity == TwinDataValidity.Valid &&
            (snapshot.ego.validity != TwinDataValidity.Valid ||
             snapshot.vehicle.validity != TwinDataValidity.Valid ||
             snapshot.wheels.validity != TwinDataValidity.Valid))
        {
            error = "A Valid snapshot cannot contain a Partial or Invalid required data group; mark the snapshot Partial.";
            return false;
        }
        error = null;
        return true;
    }

    private static bool HasExplicitValidity(TwinDataValidity validity) =>
        validity == TwinDataValidity.Valid ||
        validity == TwinDataValidity.Partial ||
        validity == TwinDataValidity.Invalid;

    private static TwinSessionInfo CreateCompatibilityReplaySession(ReplayPackage package, TwinSessionStatus status)
    {
        return new TwinSessionInfo
        {
            sourceId = string.IsNullOrWhiteSpace(package.source) ? "replay" : package.source,
            sessionId = string.IsNullOrWhiteSpace(package.sceneId) ? "legacy-replay" : package.sceneId,
            sourceKind = TwinSourceKind.Replay,
            status = status,
            capabilities = TwinSourceCapabilities.EgoPose | TwinSourceCapabilities.VehicleTelemetry |
                           TwinSourceCapabilities.WheelTelemetry | TwinSourceCapabilities.SurroundingActors |
                           TwinSourceCapabilities.Seek | TwinSourceCapabilities.Pause |
                           TwinSourceCapabilities.FutureTrajectory
        };
    }

    private static TwinSessionInfo CopySession(TwinSessionInfo session)
    {
        return new TwinSessionInfo
        {
            sourceId = session?.sourceId,
            sessionId = session?.sessionId,
            sourceKind = session?.sourceKind ?? TwinSourceKind.Unknown,
            status = session?.status ?? TwinSessionStatus.Disconnected,
            capabilities = session?.capabilities ?? TwinSourceCapabilities.None,
            statusMessage = session?.statusMessage
        };
    }

    private static ReplayStatus ToReplayStatus(TwinSessionStatus status)
    {
        switch (status)
        {
            case TwinSessionStatus.Ready: return ReplayStatus.Ready;
            case TwinSessionStatus.Running: return ReplayStatus.Playing;
            case TwinSessionStatus.Paused: return ReplayStatus.Paused;
            case TwinSessionStatus.Finished: return ReplayStatus.Finished;
            case TwinSessionStatus.Error: return ReplayStatus.Error;
            default: return ReplayStatus.Loading;
        }
    }

    private static TwinSessionStatus FromReplayStatus(ReplayStatus status)
    {
        switch (status)
        {
            case ReplayStatus.Ready: return TwinSessionStatus.Ready;
            case ReplayStatus.Playing: return TwinSessionStatus.Running;
            case ReplayStatus.Paused: return TwinSessionStatus.Paused;
            case ReplayStatus.Finished: return TwinSessionStatus.Finished;
            case ReplayStatus.Error: return TwinSessionStatus.Error;
            default: return TwinSessionStatus.Connecting;
        }
    }
}

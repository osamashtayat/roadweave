using System;
using UnityEngine;

/// <summary>
/// Common ingress point for adapters that receive complete canonical snapshots.
/// Transport code (socket, ROS, MQTT, HTTP, etc.) should parse its message and
/// call AcceptSnapshot; it must not update Unity objects directly.
/// </summary>
public abstract class TwinPushSourceBase : MonoBehaviour, ITwinStateSource, ITwinSessionControl
{
    [SerializeField] protected DigitalTwinStateManager stateManager;
    [SerializeField] private string sourceId;
    [SerializeField, Min(0.05f)] private float staleAfterSeconds = 1f;
    [SerializeField] private bool connectOnEnable = true;

    public TwinSessionInfo Session { get; private set; }
    public bool CanStart => Session != null && Session.status != TwinSessionStatus.Error;
    public event Action<TwinSnapshot> SnapshotProduced;
    public event Action<TwinSessionInfo> SessionChanged;

    protected abstract TwinSourceKind SourceKind { get; }
    protected virtual TwinSourceCapabilities Capabilities =>
        TwinSourceCapabilities.EgoPose |
        TwinSourceCapabilities.VehicleTelemetry |
        TwinSourceCapabilities.WheelTelemetry |
        TwinSourceCapabilities.SurroundingActors |
        TwinSourceCapabilities.LiveUpdates |
        TwinSourceCapabilities.Pause;

    protected virtual void Awake()
    {
        if (stateManager == null) stateManager = FindAnyObjectByType<DigitalTwinStateManager>();
        if (string.IsNullOrWhiteSpace(sourceId)) sourceId = SourceKind.ToString();
        BeginNewSession(TwinSessionStatus.Ready);
    }

    protected virtual void OnEnable()
    {
        if (connectOnEnable && stateManager != null) stateManager.ConnectSource(this);
    }

    protected virtual void OnDisable()
    {
        if (stateManager != null) stateManager.DisconnectSource(this);
    }

    public void BeginNewSession(TwinSessionStatus initialStatus = TwinSessionStatus.Ready)
    {
        Session = new TwinSessionInfo
        {
            sourceId = sourceId,
            sessionId = $"{sourceId}-{Guid.NewGuid():N}",
            sourceKind = SourceKind,
            status = initialStatus,
            capabilities = Capabilities
        };
        SessionChanged?.Invoke(Session);
    }

    public bool AcceptSnapshot(TwinSnapshot snapshot)
    {
        if (snapshot == null) return false;
        // Pause is part of this source's advertised contract. Transports may
        // continue receiving, but paused observations must not reach consumers.
        if (Session?.status == TwinSessionStatus.Paused)
            return false;
        if (!ValidateRequiredDomains(snapshot, out string validationError))
        {
            Debug.LogWarning($"Rejected {SourceKind} snapshot: {validationError}");
            return false;
        }
        snapshot.metadata.session = CopySession(Session);
        if (snapshot.metadata.staleAfterSeconds <= 0f)
            snapshot.metadata.staleAfterSeconds = staleAfterSeconds;
        NormalizeActorObservationMetadata(snapshot);
        SnapshotProduced?.Invoke(snapshot);
        return true;
    }

    /// <summary>Convenience ingress for tests or a future transport adapter.</summary>
    public bool AcceptJsonMessage(string json)
    {
        if (string.IsNullOrWhiteSpace(json)) return false;
        TwinSnapshot snapshot;
        try { snapshot = JsonUtility.FromJson<TwinSnapshot>(json); }
        catch (Exception exception)
        {
            // A malformed datagram is a message-level failure, not a permanent
            // source-session failure. The next valid live message may continue.
            Debug.LogWarning($"Could not parse snapshot JSON: {exception.Message}");
            return false;
        }
        if (snapshot == null)
        {
            Debug.LogWarning("Snapshot JSON produced no digital-twin state.");
            return false;
        }
        return AcceptSnapshot(snapshot);
    }

    public void StartSession()
    {
        MakeActiveSource();
        ChangeStatus(TwinSessionStatus.Running);
    }
    public void PauseSession() => ChangeStatus(TwinSessionStatus.Paused);

    public void RestartSession(bool startRunning = false)
    {
        MakeActiveSource();
        BeginNewSession(startRunning ? TwinSessionStatus.Running : TwinSessionStatus.Ready);
    }

    public void MakeActiveSource()
    {
        if (stateManager == null) stateManager = FindAnyObjectByType<DigitalTwinStateManager>();
        stateManager?.ConnectSource(this);
    }

    public void ReportError(string message)
    {
        if (Session == null) BeginNewSession(TwinSessionStatus.Error);
        Session.status = TwinSessionStatus.Error;
        Session.statusMessage = message;
        SessionChanged?.Invoke(Session);
    }

    private void ChangeStatus(TwinSessionStatus status)
    {
        if (Session == null) BeginNewSession(status);
        else
        {
            Session.status = status;
            Session.statusMessage = null;
            SessionChanged?.Invoke(Session);
        }
    }

    private static TwinSessionInfo CopySession(TwinSessionInfo session)
    {
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

    private static void NormalizeActorObservationMetadata(TwinSnapshot snapshot)
    {
        if (snapshot.actors == null)
            return;
        foreach (TwinActorState actor in snapshot.actors)
        {
            if (actor == null)
                continue;
            if (actor.observationTimestampSeconds < 0d ||
                double.IsNaN(actor.observationTimestampSeconds) ||
                double.IsInfinity(actor.observationTimestampSeconds))
                actor.observationTimestampSeconds = snapshot.metadata.sourceTimestampSeconds;
            if (actor.observationSequenceNumber < 0)
                actor.observationSequenceNumber = snapshot.metadata.sequenceNumber;
            if (actor.freshness == TwinDataFreshness.Unknown)
                actor.freshness = TwinDataFreshness.Fresh;
        }
    }

    private static bool ValidateRequiredDomains(TwinSnapshot snapshot, out string error)
    {
        if (snapshot.metadata == null)
        {
            error = "metadata is missing.";
            return false;
        }
        if (snapshot.ego == null || snapshot.vehicle == null || snapshot.wheels == null)
        {
            error = "ego, vehicle, and wheels must be present; unavailable groups must be marked Invalid.";
            return false;
        }
        if (snapshot.metadata.validity != TwinDataValidity.Valid &&
            snapshot.metadata.validity != TwinDataValidity.Partial)
        {
            error = "metadata.validity must be explicitly Valid or Partial.";
            return false;
        }
        if (!HasExplicitValidity(snapshot.ego.validity) ||
            !HasExplicitValidity(snapshot.vehicle.validity) ||
            !HasExplicitValidity(snapshot.wheels.validity))
        {
            error = "ego, vehicle, and wheels require explicit validity values.";
            return false;
        }
        if (!TwinCoordinateFrameService.IsCanonical(snapshot.metadata.coordinateFrame))
        {
            error = "metadata.coordinateFrame must explicitly use canonical Unity-local meters.";
            return false;
        }
        if (snapshot.metadata.validity == TwinDataValidity.Valid &&
            (snapshot.ego.validity != TwinDataValidity.Valid ||
             snapshot.vehicle.validity != TwinDataValidity.Valid ||
             snapshot.wheels.validity != TwinDataValidity.Valid))
        {
            error = "metadata.validity must be Partial when a required group is Partial or Invalid.";
            return false;
        }
        error = null;
        return true;
    }

    private static bool HasExplicitValidity(TwinDataValidity validity) =>
        validity == TwinDataValidity.Valid ||
        validity == TwinDataValidity.Partial ||
        validity == TwinDataValidity.Invalid;
}

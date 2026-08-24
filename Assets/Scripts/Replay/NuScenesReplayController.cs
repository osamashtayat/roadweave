using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Replay source adapter. It owns replay parsing, validation, time, and interpolation,
/// then publishes the same TwinSnapshot contract used by future streaming sources.
/// </summary>
public class NuScenesReplayController : MonoBehaviour, ITwinStateSource, ITwinSessionControl
{
    [Header("Replay File")]
    [SerializeField] private string replayFileName = "Replays/scene-0001-replay.json";

    [Header("Playback")]
    [SerializeField] private DigitalTwinStateManager stateManager;
    [SerializeField, Min(0.01f)] private float playbackSpeed = 1f;
    [SerializeField] private bool playAutomatically;
    [SerializeField] private bool loop;

    public ReplayPackage Package { get; private set; }
    public bool IsLoaded => Package != null;
    public bool CanStart => IsLoaded && Session.status != TwinSessionStatus.Error;
    public TwinSessionInfo Session { get; private set; }
    public event Action<TwinSnapshot> SnapshotProduced;
    public event Action<TwinSessionInfo> SessionChanged;

    private TwinCoordinateFrame coordinateFrame;
    private float replayTime;
    private long sequenceNumber;

    private void Awake()
{
    if (stateManager == null)
        stateManager =
            GetComponent<DigitalTwinStateManager>();

    if (stateManager == null)
        stateManager =
            FindAnyObjectByType<DigitalTwinStateManager>();

    Session = CreateSession(
        TwinSessionStatus.Connecting);
}

private void OnEnable()
{
    if (stateManager == null)
        stateManager =
            FindAnyObjectByType<DigitalTwinStateManager>();

    if (stateManager != null)
        stateManager.ConnectSource(this);
}

private void OnDisable()
{
    if (stateManager != null)
        stateManager.DisconnectSource(this);
}

    private IEnumerator Start()
    {
        yield return LoadReplay();
        if (playAutomatically && IsLoaded) StartDrive();
    }

    private void OnDestroy()
    {
        if (stateManager != null) stateManager.DisconnectSource(this);
    }

    private void Update()
    {
        if (!IsLoaded || Session.status != TwinSessionStatus.Running ||
            stateManager == null || !ReferenceEquals(stateManager.ConnectedSource, this))
            return;
        replayTime += Time.deltaTime * playbackSpeed;

        if (replayTime >= Package.durationSeconds)
        {
            if (loop)
            {
                replayTime = 0f;
                PublishCurrentSnapshot();
            }
            else
            {
                replayTime = Package.durationSeconds;
                PublishCurrentSnapshot();
                ChangeStatus(TwinSessionStatus.Finished);
            }
            return;
        }
        PublishCurrentSnapshot();
    }

    private IEnumerator LoadReplay()
    {
        if (stateManager == null)
        {
            Debug.LogError("NuScenesReplayController needs a DigitalTwinStateManager.");
            ChangeStatus(TwinSessionStatus.Error, "DigitalTwinStateManager is missing.");
            yield break;
        }

        ChangeStatus(TwinSessionStatus.Connecting);
        string path = Path.Combine(Application.streamingAssetsPath, replayFileName);
        string json = null;
        if (path.Contains("://"))
        {
            using (UnityWebRequest request = UnityWebRequest.Get(path))
            {
                yield return request.SendWebRequest();
                if (request.result != UnityWebRequest.Result.Success)
                {
                    FailLoading(path, request.error);
                    yield break;
                }
                json = request.downloadHandler.text;
            }
        }
        else
        {
            if (!File.Exists(path))
            {
                FailLoading(path, "File does not exist");
                yield break;
            }
            json = File.ReadAllText(path);
        }

        ReplayPackage loaded;
        try { loaded = JsonUtility.FromJson<ReplayPackage>(json); }
        catch (Exception exception)
        {
            FailLoading(path, $"Invalid JSON: {exception.Message}");
            yield break;
        }

        if (!ReplaySnapshotSampler.ValidatePackage(loaded, out coordinateFrame, out string validationError))
        {
            FailLoading(path, validationError);
            yield break;
        }

        Package = loaded;
        replayTime = 0f;
        sequenceNumber = 0;
        Session = CreateSession(TwinSessionStatus.Ready);
        // The replay may finish loading after the user selected a live source.
        // Only the active replay owns the compatibility package/state bridge.
        if (ReferenceEquals(stateManager.ConnectedSource, this))
            stateManager.AttachReplayPackage(Package, this);
        SessionChanged?.Invoke(Session);
        PublishCurrentSnapshot();
        Debug.Log($"Loaded {Package.sceneId}: {Package.durationSeconds:F1} seconds, {Package.actors?.Length ?? 0} actors.");
    }

    private void PublishCurrentSnapshot()
    {
        if (!IsLoaded || coordinateFrame == null) return;
        TwinSnapshot snapshot = ReplaySnapshotSampler.Sample(
            Package,
            coordinateFrame,
            replayTime,
            sequenceNumber++,
            Session);
        SnapshotProduced?.Invoke(snapshot);
    }

    private void FailLoading(string path, string reason)
    {
        string message = $"Could not load replay at {path}. {reason}";
        Debug.LogError(message);
        ChangeStatus(TwinSessionStatus.Error, message);
    }

    public void HandleDriveButton()
    {
        if (!CanStart) return;
        if (stateManager == null || !ReferenceEquals(stateManager.ConnectedSource, this))
        {
            StartDrive();
            return;
        }
        if (Session.status == TwinSessionStatus.Running) PauseReplay();
        else if (Session.status == TwinSessionStatus.Finished) RestartReplay(true);
        else StartDrive();
    }

    public void StartDrive()
    {
        if (!CanStart)
        {
            Debug.LogWarning("The Drive button was pressed before the replay finished loading.");
            return;
        }
        ActivateReplaySource();
        if (Session.status == TwinSessionStatus.Finished) replayTime = 0f;
        ChangeStatus(TwinSessionStatus.Running);
        PublishCurrentSnapshot();
    }

    public void PauseReplay()
    {
        if (Session.status == TwinSessionStatus.Running) ChangeStatus(TwinSessionStatus.Paused);
    }

    public void RestartReplay(bool startPlaying = false)
    {
        if (!IsLoaded) return;
        ActivateReplaySource();
        replayTime = 0f;
        // New session permits sequence numbers to restart without being treated as out of order.
        sequenceNumber = 0;
        Session = CreateSession(startPlaying ? TwinSessionStatus.Running : TwinSessionStatus.Ready);
        SessionChanged?.Invoke(Session);
        PublishCurrentSnapshot();
    }

    public void StartSession() => StartDrive();
    public void PauseSession() => PauseReplay();
    public void RestartSession(bool startRunning = false) => RestartReplay(startRunning);

    private void ActivateReplaySource()
    {
        if (stateManager == null)
            stateManager = FindAnyObjectByType<DigitalTwinStateManager>();
        if (stateManager == null)
            return;
        stateManager.ConnectSource(this);
        stateManager.AttachReplayPackage(Package, this);
    }

    private TwinSessionInfo CreateSession(TwinSessionStatus status)
    {
        string source = Package == null || string.IsNullOrWhiteSpace(Package.source) ? "nuScenes-replay" : Package.source;
        string scene = Package == null || string.IsNullOrWhiteSpace(Package.sceneId) ? "loading" : Package.sceneId;
        return new TwinSessionInfo
        {
            sourceId = source,
            sessionId = $"{scene}-{Guid.NewGuid():N}",
            sourceKind = TwinSourceKind.Replay,
            status = status,
            capabilities = TwinSourceCapabilities.EgoPose |
                           TwinSourceCapabilities.VehicleTelemetry |
                           TwinSourceCapabilities.WheelTelemetry |
                           TwinSourceCapabilities.SurroundingActors |
                           TwinSourceCapabilities.Seek |
                           TwinSourceCapabilities.Pause |
                           TwinSourceCapabilities.FutureTrajectory
        };
    }

    private void ChangeStatus(TwinSessionStatus status, string message = null)
    {
        if (Session == null) Session = CreateSession(status);
        Session.status = status;
        Session.statusMessage = message;
        SessionChanged?.Invoke(Session);
    }
}

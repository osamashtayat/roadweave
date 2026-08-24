using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Publishes canonical snapshot templates on a live cadence through a
/// SimulatedStreamTwinSource. It models a scheduled stream with a fresh session,
/// sequence, and clock; it does not expose replay seek or future-trajectory behavior.
/// </summary>
public sealed class SimulatedLiveTwinPlayer : MonoBehaviour
{
    [SerializeField] private SimulatedStreamTwinSource simulatedSource;
    [SerializeField] private TextAsset[] snapshotJsonTemplates;
    [SerializeField, Min(0.01f)] private float publishIntervalSeconds = 0.1f;
    [SerializeField, Min(1)] private int maximumPublishesPerFrame = 4;
    [SerializeField] private bool loop = true;
    [SerializeField] private bool startAutomatically;
    [SerializeField] private bool useUnscaledTime = true;

    private readonly List<TwinSnapshot> templates = new List<TwinSnapshot>();
    private int templateIndex;
    private long sequenceNumber;
    private double simulatedTimeSeconds;
    private float timeUntilNextPublish;
    private SimulatedStreamTwinSource subscribedSource;

    public bool IsRunning { get; private set; }
    public int TemplateCount => templates.Count;
    public long PublishedCount { get; private set; }

    private void Awake()
    {
        ResolveSource();
        LoadSerializedTemplates();
    }

    private void OnEnable()
    {
        ResolveSource();
        SubscribeToResolvedSource();
    }

    private void OnDisable()
    {
        if (subscribedSource != null)
            subscribedSource.SessionChanged -= HandleSourceSessionChanged;
        subscribedSource = null;
    }

    private void Start()
    {
        if (startAutomatically)
            StartPlayer();
    }

    private void Update()
    {
        Tick(useUnscaledTime ? Time.unscaledDeltaTime : Time.deltaTime);
    }

    public bool StartPlayer()
    {
        ResolveSource();
        if (simulatedSource == null)
        {
            Debug.LogError("SimulatedLiveTwinPlayer requires a SimulatedStreamTwinSource.");
            return false;
        }
        if (templates.Count == 0)
        {
            Debug.LogWarning("SimulatedLiveTwinPlayer has no snapshot templates to publish.");
            return false;
        }
        simulatedSource.StartSession();
        IsRunning = true;
        return true;
    }

    public void PausePlayer()
    {
        IsRunning = false;
        simulatedSource?.PauseSession();
    }

    public bool RestartPlayer(bool startRunning = true)
    {
        ResolveSource();
        if (simulatedSource == null)
            return false;
        templateIndex = 0;
        sequenceNumber = 0;
        simulatedTimeSeconds = 0d;
        timeUntilNextPublish = 0f;
        PublishedCount = 0;
        simulatedSource.RestartSession(startRunning);
        IsRunning = startRunning && templates.Count > 0;
        return IsRunning || !startRunning;
    }

    /// <summary>
    /// Advances the scheduler and returns the number of snapshots published.
    /// This explicit entry point also makes deterministic simulator integration easy.
    /// </summary>
    public int Tick(float deltaSeconds)
    {
        if (!IsRunning || templates.Count == 0)
            return 0;
        float delta = Mathf.Max(0f, deltaSeconds);
        timeUntilNextPublish -= delta;
        simulatedTimeSeconds += delta;
        int publishedThisTick = 0;
        int maximum = Mathf.Max(1, maximumPublishesPerFrame);
        float interval = Mathf.Max(0.01f, publishIntervalSeconds);
        while (timeUntilNextPublish <= 0f && publishedThisTick < maximum && IsRunning)
        {
            if (!PublishNextTemplate())
                break;
            timeUntilNextPublish += interval;
            publishedThisTick++;
        }
        return publishedThisTick;
    }

    public void SetTemplates(IEnumerable<TwinSnapshot> snapshotTemplates)
    {
        bool sessionAlreadyPublished = sequenceNumber > 0 || PublishedCount > 0;
        templates.Clear();
        if (snapshotTemplates != null)
        {
            foreach (TwinSnapshot snapshot in snapshotTemplates)
            {
                if (snapshot != null)
                    templates.Add(snapshot);
            }
        }
        templateIndex = 0;
        timeUntilNextPublish = 0f;
        // Replacing content is not a session restart. Preserve the monotonic
        // sequence/source clock so the state manager accepts the next snapshot.
        if (!sessionAlreadyPublished)
        {
            sequenceNumber = 0;
            simulatedTimeSeconds = 0d;
            PublishedCount = 0;
        }
        if (simulatedSource?.Session?.status == TwinSessionStatus.Running)
            IsRunning = templates.Count > 0;
    }

    public bool AddTemplateJson(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
            return false;
        try
        {
            TwinSnapshot snapshot = JsonUtility.FromJson<TwinSnapshot>(json);
            if (snapshot == null)
                return false;
            templates.Add(snapshot);
            return true;
        }
        catch (Exception exception)
        {
            Debug.LogWarning($"Skipped invalid simulated snapshot template: {exception.Message}");
            return false;
        }
    }

    private bool PublishNextTemplate()
    {
        if (templateIndex >= templates.Count)
        {
            if (!loop)
            {
                PausePlayer();
                return false;
            }
            templateIndex = 0;
        }

        TwinSnapshot snapshot = CloneSnapshot(templates[templateIndex++]);
        if (snapshot == null)
            return false;
        if (snapshot.metadata == null)
            snapshot.metadata = new TwinSnapshotMetadata();
        snapshot.metadata.sequenceNumber = sequenceNumber++;
        snapshot.metadata.sourceTimestampSeconds = simulatedTimeSeconds;
        snapshot.metadata.timelineDurationSeconds = 0d;
        snapshot.metadata.freshness = TwinDataFreshness.Fresh;
        if (snapshot.metadata.coordinateFrame == null)
            snapshot.metadata.coordinateFrame = TwinCoordinateFrame.UnityLocal("simulated stream origin");
        bool accepted = simulatedSource.AcceptSnapshot(snapshot);
        if (accepted) PublishedCount++;
        return accepted;
    }

    private void LoadSerializedTemplates()
    {
        templates.Clear();
        if (snapshotJsonTemplates == null)
            return;
        foreach (TextAsset template in snapshotJsonTemplates)
        {
            if (template != null)
                AddTemplateJson(template.text);
        }
    }

    private void ResolveSource()
    {
        if (simulatedSource == null)
            simulatedSource = FindAnyObjectByType<SimulatedStreamTwinSource>();
        if (isActiveAndEnabled)
            SubscribeToResolvedSource();
    }

    private void SubscribeToResolvedSource()
    {
        if (ReferenceEquals(subscribedSource, simulatedSource))
            return;
        if (subscribedSource != null)
            subscribedSource.SessionChanged -= HandleSourceSessionChanged;
        subscribedSource = simulatedSource;
        if (subscribedSource != null)
            subscribedSource.SessionChanged += HandleSourceSessionChanged;
    }

    private void HandleSourceSessionChanged(TwinSessionInfo session)
    {
        if (session == null)
            return;
        if (session.status == TwinSessionStatus.Running)
            IsRunning = templates.Count > 0;
        else if (session.status == TwinSessionStatus.Paused ||
                 session.status == TwinSessionStatus.Finished ||
                 session.status == TwinSessionStatus.Error ||
                 session.status == TwinSessionStatus.Disconnected)
            IsRunning = false;
    }

    private static TwinSnapshot CloneSnapshot(TwinSnapshot source)
    {
        if (source == null)
            return null;
        return JsonUtility.FromJson<TwinSnapshot>(JsonUtility.ToJson(source));
    }
}

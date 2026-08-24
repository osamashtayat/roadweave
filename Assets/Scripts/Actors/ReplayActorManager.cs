using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Materializes the canonical actor collection published by the active digital-
/// twin source. The historical class name is retained so existing scenes keep
/// their serialized component and prefab references.
/// </summary>
public class ReplayActorManager : MonoBehaviour
{
    [SerializeField] private DigitalTwinStateManager stateManager;
    [Tooltip("Create RuntimeActors under ReplayRoot and drag it here.")]
    [SerializeField] private Transform runtimeActorsRoot;

    [Header("Optional Prefabs")]
    [SerializeField] private GameObject carPrefab;
    [SerializeField] private GameObject truckPrefab;
    [SerializeField] private GameObject busPrefab;
    [SerializeField] private GameObject constructionVehiclePrefab;
    [SerializeField] private GameObject pedestrianPrefab;
    [SerializeField] private GameObject trafficConePrefab;
    [SerializeField] private GameObject genericObjectPrefab;
    [SerializeField] private bool scaleProvidedPrefabsToDatasetSize;
    [SerializeField] private bool groundRoadActorsToEgoPlane;
    [SerializeField, Min(0f)] private float groundClearance = 0.015f;

    [Header("Streaming Presentation")]
    [SerializeField, Min(0f)] private float maximumExtrapolationSeconds = 0.25f;

    private sealed class ActorInstance
    {
        public GameObject root;
        public ActorVisualRoot visualRoot;
        public TwinActorIdentity identity;
        public ScenarioActorMarker scenarioMarker;
        public TwinActorClass semanticClass;
        public TwinActorMotionState motionState;
        public Vector3 dimensions;
        public Vector3 latestLocalPosition;
        public Vector3 velocityMetersPerSecond;
        public float latestYawDegrees;
        public double receiptTimestampSeconds;
        public TwinActorState latestState;
    }

    private readonly Dictionary<string, ActorInstance> actorInstances =
        new Dictionary<string, ActorInstance>(StringComparer.Ordinal);
    private readonly Dictionary<string, GameObject> actorsByStableId =
        new Dictionary<string, GameObject>(StringComparer.Ordinal);
    private readonly HashSet<string> actorIdsInSnapshot = new HashSet<string>(StringComparer.Ordinal);
    private readonly List<string> actorIdsToRemove = new List<string>();

    private bool runtimeActorsVisible = true;
    private string activeSessionId;
    private TwinSourceKind activeSourceKind = TwinSourceKind.Unknown;

    public bool RuntimeActorsVisible => runtimeActorsVisible;
    public int RuntimeActorCount => actorInstances.Count;
    public event Action<TwinActorIdentity> ActorCreated;
    public event Action<string> ActorRemoved;

    private void Awake()
    {
        if (stateManager == null)
            stateManager = FindAnyObjectByType<DigitalTwinStateManager>();
        EnsureRuntimeRoot();
    }

    private void OnEnable()
    {
        if (stateManager == null)
            stateManager = FindAnyObjectByType<DigitalTwinStateManager>();
        if (stateManager != null)
            stateManager.SnapshotUpdated += SynchronizeSnapshot;
    }

    private void Start()
    {
        if (stateManager?.CurrentSnapshot != null)
            SynchronizeSnapshot(stateManager.CurrentSnapshot);
    }

    private void OnDisable()
    {
        if (stateManager != null)
            stateManager.SnapshotUpdated -= SynchronizeSnapshot;
    }

    private void OnDestroy()
    {
        if (stateManager != null)
            stateManager.SnapshotUpdated -= SynchronizeSnapshot;
        ClearSpawnedActors();
    }

    private void LateUpdate()
    {
        // Replay already publishes interpolated states each frame. A lower-rate
        // stream may extrapolate briefly using canonical actor velocity, then
        // freezes before stale data can be mistaken for a live observation.
        if (activeSourceKind == TwinSourceKind.Replay ||
            stateManager == null || !stateManager.IsFresh ||
            maximumExtrapolationSeconds <= 0f)
            return;

        double now = Time.realtimeSinceStartup;
        foreach (ActorInstance instance in actorInstances.Values)
        {
            if (instance.root == null || instance.latestState == null ||
                instance.latestState.freshness != TwinDataFreshness.Fresh)
                continue;
            float age = Mathf.Clamp(
                (float)(now - instance.receiptTimestampSeconds),
                0f,
                maximumExtrapolationSeconds);
            instance.root.transform.localPosition =
                instance.latestLocalPosition + instance.velocityMetersPerSecond * age;
            instance.root.transform.localRotation = Quaternion.Euler(0f, instance.latestYawDegrees, 0f);
        }
    }

    /// <summary>
    /// Applies one authoritative actor collection. An unavailable or stale state
    /// removes actors so an old source is never presented as the current source.
    /// </summary>
    public void SynchronizeSnapshot(TwinSnapshot snapshot)
    {
        if (snapshot == null)
        {
            ClearSpawnedActors();
            activeSessionId = null;
            activeSourceKind = TwinSourceKind.Unknown;
            return;
        }
        if (snapshot.metadata == null || snapshot.metadata.session == null)
            return;
        if (snapshot.metadata.validity == TwinDataValidity.Invalid ||
            snapshot.metadata.freshness == TwinDataFreshness.Stale)
        {
            ClearSpawnedActors();
            return;
        }

        EnsureRuntimeRoot();
        string sessionId = snapshot.metadata.session.sessionId ?? string.Empty;
        if (!string.Equals(activeSessionId, sessionId, StringComparison.Ordinal))
        {
            ClearSpawnedActors();
            activeSessionId = sessionId;
        }
        activeSourceKind = snapshot.metadata.session.sourceKind;

        actorIdsInSnapshot.Clear();
        TwinActorState[] actors = snapshot.actors ?? Array.Empty<TwinActorState>();
        foreach (TwinActorState actorState in actors)
        {
            if (!IsUsable(actorState) || !actorIdsInSnapshot.Add(actorState.id))
                continue;
            UpsertActor(actorState, snapshot);
        }

        actorIdsToRemove.Clear();
        foreach (string stableId in actorInstances.Keys)
        {
            if (!actorIdsInSnapshot.Contains(stableId))
                actorIdsToRemove.Add(stableId);
        }
        foreach (string stableId in actorIdsToRemove)
            RemoveActor(stableId);
    }

    public void SetRuntimeActorsVisible(bool visible)
    {
        runtimeActorsVisible = visible;
        EnsureRuntimeRoot();
        runtimeActorsRoot.gameObject.SetActive(visible);
    }

    public bool TryGetActor(string stableId, out GameObject actor)
    {
        if (string.IsNullOrEmpty(stableId))
        {
            actor = null;
            return false;
        }
        return actorsByStableId.TryGetValue(stableId, out actor) && actor != null;
    }

    public bool TryGetActorState(string stableId, out TwinActorState state)
    {
        if (!string.IsNullOrEmpty(stableId) &&
            actorInstances.TryGetValue(stableId, out ActorInstance instance))
        {
            state = instance.latestState;
            return state != null;
        }
        state = null;
        return false;
    }

    private void UpsertActor(TwinActorState state, TwinSnapshot snapshot)
    {
        if (actorInstances.TryGetValue(state.id, out ActorInstance existing) &&
            existing.semanticClass != state.semanticClass)
        {
            RemoveActor(state.id);
            existing = null;
        }

        ActorInstance instance = existing ?? CreateActor(state, snapshot.metadata.session);
        if (instance == null)
            return;

        instance.latestState = state;
        Vector3 dimensions = SanitizeDimensions(state.dimensionsMeters, state.semanticClass);
        if ((instance.dimensions - dimensions).sqrMagnitude > 0.0001f)
        {
            ApplyDimensions(instance, dimensions);
            instance.dimensions = dimensions;
        }

        if (instance.motionState != state.motionState)
        {
            instance.motionState = state.motionState;
            instance.scenarioMarker.Configure(
                ToScenarioClass(state.semanticClass),
                ToScenarioMotion(state.motionState),
                CanOvertake(state.semanticClass));
        }

        instance.latestLocalPosition = SanitizeVector(state.position, instance.latestLocalPosition);
        if (groundRoadActorsToEgoPlane && IsRoadActor(state.semanticClass) &&
            stateManager != null && stateManager.IsReady)
        {
            instance.latestLocalPosition.y = stateManager.Ego.position.y +
                                             dimensions.y * 0.5f + groundClearance;
        }
        instance.velocityMetersPerSecond = SanitizeVector(state.velocityMetersPerSecond, Vector3.zero);
        instance.latestYawDegrees = IsFinite(state.yawDegrees) ? state.yawDegrees : instance.latestYawDegrees;
        instance.receiptTimestampSeconds = snapshot.metadata.receiptTimestampSeconds;
        instance.root.transform.localPosition = instance.latestLocalPosition;
        instance.root.transform.localRotation = Quaternion.Euler(0f, instance.latestYawDegrees, 0f);
    }

    private ActorInstance CreateActor(TwinActorState state, TwinSessionInfo session)
    {
        GameObject prefab = GetPrefab(state);
        bool isPlaceholder = prefab == null;
        GameObject actorRoot = new GameObject($"{ActorLabel(state)}_{ShortId(state.id)}");
        actorRoot.transform.SetParent(runtimeActorsRoot, false);

        GameObject visual = isPlaceholder ? CreatePlaceholder(state) : Instantiate(prefab);
        visual.name = isPlaceholder ? "PlaceholderVisual" : "ModelVisual";
        ActorVisualRoot visualRoot = actorRoot.AddComponent<ActorVisualRoot>();
        visualRoot.SetVisual(visual.transform);

        TwinActorIdentity identity = actorRoot.AddComponent<TwinActorIdentity>();
        identity.Configure(state.id, session.sourceKind.ToString());

        ScenarioActorMarker marker = actorRoot.AddComponent<ScenarioActorMarker>();
        marker.Configure(
            ToScenarioClass(state.semanticClass),
            ToScenarioMotion(state.motionState),
            CanOvertake(state.semanticClass));

        ActorInstance instance = new ActorInstance
        {
            root = actorRoot,
            visualRoot = visualRoot,
            identity = identity,
            scenarioMarker = marker,
            semanticClass = state.semanticClass,
            motionState = state.motionState
        };
        actorInstances.Add(state.id, instance);
        actorsByStableId.Add(state.id, actorRoot);
        // RuntimeActorsRoot is the single visibility/isolation switch. Keeping
        // children activeSelf=true ensures actors created while Test Lab is open
        // correctly reappear when returning to Live Twin.
        actorRoot.SetActive(true);
        ActorCreated?.Invoke(identity);
        return instance;
    }

    private GameObject GetPrefab(TwinActorState state)
    {
        switch (state.semanticClass)
        {
            case TwinActorClass.Car: return carPrefab;
            case TwinActorClass.Truck: return truckPrefab;
            case TwinActorClass.Bus: return busPrefab;
            case TwinActorClass.Pedestrian: return pedestrianPrefab;
            case TwinActorClass.Barrier:
                return IsTrafficCone(state.sourceClass) ? trafficConePrefab : genericObjectPrefab;
            case TwinActorClass.Other:
                if (Contains(state.sourceClass, "construction")) return constructionVehiclePrefab;
                if (IsTrafficCone(state.sourceClass)) return trafficConePrefab;
                return genericObjectPrefab;
            default: return genericObjectPrefab;
        }
    }

    private static GameObject CreatePlaceholder(TwinActorState state)
    {
        if (state.semanticClass == TwinActorClass.Pedestrian)
            return FallbackActorVisualFactory.CreatePedestrian();

        PrimitiveType primitive = state.semanticClass == TwinActorClass.Pedestrian
            ? PrimitiveType.Capsule
            : IsTrafficCone(state.sourceClass) ? PrimitiveType.Cylinder : PrimitiveType.Cube;
        GameObject placeholder = GameObject.CreatePrimitive(primitive);
        Renderer actorRenderer = placeholder.GetComponent<Renderer>();
        if (actorRenderer != null)
            actorRenderer.material.color = PlaceholderColor(state.semanticClass, state.sourceClass);
        return placeholder;
    }

    private void ApplyDimensions(ActorInstance instance, Vector3 dimensions)
    {
        bool placeholder = instance.visualRoot?.Visual != null &&
                           instance.visualRoot.Visual.name == "PlaceholderVisual";
        if (placeholder)
        {
            if (instance.semanticClass == TwinActorClass.Pedestrian)
            {
                instance.visualRoot?.FitVisualToWorldSize(dimensions);
            }
            else
            {
                bool halfHeightPrimitive = IsTrafficCone(instance.latestState?.sourceClass);
                instance.visualRoot.Visual.localScale = new Vector3(
                    dimensions.x,
                    halfHeightPrimitive ? dimensions.y * 0.5f : dimensions.y,
                    dimensions.z);
            }
        }
        else if (scaleProvidedPrefabsToDatasetSize)
        {
            instance.visualRoot?.FitVisualToWorldSize(dimensions);
        }

        instance.visualRoot?.CenterVisualOnActorRoot();

        BoxCollider rootCollider = instance.root.GetComponent<BoxCollider>();
        if (rootCollider == null)
            rootCollider = instance.root.AddComponent<BoxCollider>();
        rootCollider.center = Vector3.zero;
        rootCollider.size = dimensions;
    }

    private void RemoveActor(string stableId)
    {
        if (!actorInstances.TryGetValue(stableId, out ActorInstance instance))
            return;
        actorInstances.Remove(stableId);
        actorsByStableId.Remove(stableId);
        ActorRemoved?.Invoke(stableId);
        if (instance.root != null)
        {
            if (Application.isPlaying)
                Destroy(instance.root);
            else
                DestroyImmediate(instance.root);
        }
    }

    private void ClearSpawnedActors()
    {
        actorIdsToRemove.Clear();
        actorIdsToRemove.AddRange(actorInstances.Keys);
        foreach (string stableId in actorIdsToRemove)
            RemoveActor(stableId);
        actorIdsToRemove.Clear();
        actorIdsInSnapshot.Clear();
    }

    private void EnsureRuntimeRoot()
    {
        if (runtimeActorsRoot != null)
            return;
        GameObject rootObject = new GameObject("RuntimeActors");
        rootObject.transform.SetParent(transform, false);
        runtimeActorsRoot = rootObject.transform;
        runtimeActorsRoot.gameObject.SetActive(runtimeActorsVisible);
    }

    private static bool IsUsable(TwinActorState state)
    {
        return state != null &&
               !string.IsNullOrWhiteSpace(state.id) &&
               (state.validity == TwinDataValidity.Valid || state.validity == TwinDataValidity.Partial) &&
               state.freshness == TwinDataFreshness.Fresh;
    }

    private static ScenarioActorClass ToScenarioClass(TwinActorClass actorClass)
    {
        switch (actorClass)
        {
            case TwinActorClass.Pedestrian: return ScenarioActorClass.Pedestrian;
            case TwinActorClass.Car: return ScenarioActorClass.Car;
            case TwinActorClass.Truck: return ScenarioActorClass.Truck;
            case TwinActorClass.Bus: return ScenarioActorClass.Bus;
            default: return ScenarioActorClass.Unknown;
        }
    }

    private static ScenarioActorMotionState ToScenarioMotion(TwinActorMotionState motionState)
    {
        switch (motionState)
        {
            case TwinActorMotionState.Stopped: return ScenarioActorMotionState.Stopped;
            case TwinActorMotionState.Slow: return ScenarioActorMotionState.Slow;
            case TwinActorMotionState.Moving: return ScenarioActorMotionState.Moving;
            default: return ScenarioActorMotionState.Unknown;
        }
    }

    private static bool CanOvertake(TwinActorClass actorClass)
    {
        return actorClass == TwinActorClass.Car ||
               actorClass == TwinActorClass.Truck ||
               actorClass == TwinActorClass.Bus ||
               actorClass == TwinActorClass.Pedestrian ||
               actorClass == TwinActorClass.Bicycle ||
               actorClass == TwinActorClass.Motorcycle;
    }

    private static bool IsRoadActor(TwinActorClass actorClass)
    {
        return actorClass == TwinActorClass.Car ||
               actorClass == TwinActorClass.Truck ||
               actorClass == TwinActorClass.Bus ||
               actorClass == TwinActorClass.Pedestrian ||
               actorClass == TwinActorClass.Bicycle ||
               actorClass == TwinActorClass.Motorcycle ||
               actorClass == TwinActorClass.Barrier ||
               actorClass == TwinActorClass.Other;
    }

    private static Vector3 SanitizeDimensions(Vector3 size, TwinActorClass actorClass)
    {
        Vector3 fallback = actorClass == TwinActorClass.Pedestrian
            ? new Vector3(0.7f, 1.8f, 0.7f)
            : actorClass == TwinActorClass.Truck || actorClass == TwinActorClass.Bus
                ? new Vector3(2.5f, 3f, 8f)
                : new Vector3(1.8f, 1.5f, 4.2f);
        return new Vector3(
            SafeDimension(size.x, fallback.x),
            SafeDimension(size.y, fallback.y),
            SafeDimension(size.z, fallback.z));
    }

    private static float SafeDimension(float value, float fallback)
    {
        return IsFinite(value) && value > 0.05f ? value : fallback;
    }

    private static Vector3 SanitizeVector(Vector3 value, Vector3 fallback)
    {
        return IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z) ? value : fallback;
    }

    private static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
    private static bool Contains(string text, string value) =>
        !string.IsNullOrEmpty(text) && text.IndexOf(value, StringComparison.OrdinalIgnoreCase) >= 0;
    private static bool IsTrafficCone(string sourceClass) =>
        Contains(sourceClass, "trafficcone") || Contains(sourceClass, "traffic_cone") || Contains(sourceClass, "cone");
    private static string ActorLabel(TwinActorState state) =>
        state.semanticClass == TwinActorClass.Other && !string.IsNullOrWhiteSpace(state.sourceClass)
            ? state.sourceClass.Replace('.', '_')
            : state.semanticClass.ToString();

    private static Color PlaceholderColor(TwinActorClass actorClass, string sourceClass)
    {
        switch (actorClass)
        {
            case TwinActorClass.Car: return new Color(0.15f, 0.45f, 0.95f);
            case TwinActorClass.Truck: return new Color(0.95f, 0.55f, 0.12f);
            case TwinActorClass.Bus: return new Color(0.95f, 0.82f, 0.12f);
            case TwinActorClass.Pedestrian: return new Color(0.2f, 0.85f, 0.35f);
            case TwinActorClass.Barrier: return new Color(1f, 0.3f, 0.05f);
            default: return Contains(sourceClass, "construction")
                ? new Color(0.95f, 0.75f, 0.1f)
                : Color.gray;
        }
    }

    private static string ShortId(string id)
    {
        if (string.IsNullOrEmpty(id)) return "unknown";
        return id.Length <= 8 ? id : id.Substring(0, 8);
    }
}

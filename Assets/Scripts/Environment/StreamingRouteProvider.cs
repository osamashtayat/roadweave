using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Mutable world-space route provider for simulated streams and future live
/// map/route services. Feed it route points without coupling Test Lab to the
/// protocol that delivered them.
/// </summary>
[DisallowMultipleComponent]
public sealed class StreamingRouteProvider : MonoBehaviour, ITestWorldRouteProvider
{
    [Header("Digital Twin Source")]
    [SerializeField] private DigitalTwinStateManager stateManager;
    [Tooltip("Transforms canonical source-local positions into Unity world space. Defaults to this transform.")]
    [SerializeField] private Transform canonicalCoordinateRoot;

    [SerializeField, Min(2)] private int maximumBufferedPoints = 2000;
    [SerializeField, Min(0f)] private float incomingPointSpacing = 0.25f;
    [SerializeField, Min(0f)] private float continuationLength = 250f;
    [SerializeField, Min(0.5f)] private float continuationPointSpacing = 2f;

    [Header("Generated Road")]
    [SerializeField, Min(6f)] private float roadWidth = 12f;
    [SerializeField, Min(0f)] private float egoRightLaneOffset = 2.75f;
    [SerializeField] private float roadYOffset = -0.06f;
    [SerializeField, Min(0.5f)] private float footpathWidth = 2.25f;
    [SerializeField, Min(0.01f)] private float footpathHeight = 0.14f;
    [SerializeField] private bool createMeshColliders = true;
    [SerializeField] private Color roadColor = new Color(0.12f, 0.13f, 0.14f, 1f);
    [SerializeField] private Color footpathColor = new Color(0.46f, 0.47f, 0.49f, 1f);

    private readonly List<Vector3> routePoints = new List<Vector3>();
    private readonly List<Vector3> renderedRoutePoints = new List<Vector3>();
    private ProceduralRouteSurface routeSurface;
    private string activeSessionId;
    private TwinSourceKind activeSourceKind = TwinSourceKind.Unknown;
    private bool routeOwnedByExternalAdapter;
    private string streamedRouteId;
    private long streamedRouteRevision = long.MinValue;

    public string ProviderName => "Streaming / simulation route";
    public bool IsRouteAvailable => routePoints.Count > 1;
    public int PointCount => routePoints.Count;
    public int RenderedRoadVertexCount => routeSurface != null ? routeSurface.RoadVertexCount : 0;
    public int RenderedFootpathVertexCount => routeSurface != null ? routeSurface.FootpathVertexCount : 0;

    private void Awake()
    {
        ResolveDependencies();
        EnsureRouteSurface();
    }

    private void OnEnable()
    {
        ResolveDependencies();
        if (stateManager != null)
            stateManager.SnapshotUpdated += HandleSnapshot;
    }

    private void OnDisable()
    {
        if (stateManager != null)
            stateManager.SnapshotUpdated -= HandleSnapshot;
    }

    public void SetRoute(IReadOnlyList<Vector3> worldPoints)
    {
        if (worldPoints == null)
        {
            Clear();
            return;
        }
        ClaimExternalRouteOwnership();
        routePoints.Clear();
        for (int index = 0; index < worldPoints.Count; index++)
            AppendPointInternal(worldPoints[index]);
        RebuildSurface();
    }

    public void AppendPoint(Vector3 worldPoint)
    {
        ClaimExternalRouteOwnership();
        if (AppendPointInternal(worldPoint))
            RebuildSurface();
    }

    public void Clear()
    {
        routeOwnedByExternalAdapter = false;
        ClearRouteData();
    }

    private void ClearRouteData()
    {
        routePoints.Clear();
        renderedRoutePoints.Clear();
        streamedRouteId = null;
        streamedRouteRevision = long.MinValue;
        EnsureRouteSurface();
        routeSurface.Clear();
    }

    private void ResolveDependencies()
    {
        if (stateManager == null)
            stateManager = FindAnyObjectByType<DigitalTwinStateManager>();
        if (canonicalCoordinateRoot == null)
            canonicalCoordinateRoot = transform;
    }

    private void ClaimExternalRouteOwnership()
    {
        ResolveDependencies();
        TwinSessionInfo session = stateManager?.Session;
        if (session != null &&
            (session.sourceKind == TwinSourceKind.SimulatedStream ||
             session.sourceKind == TwinSourceKind.LiveSensor))
        {
            if (activeSourceKind != session.sourceKind || activeSessionId != session.sessionId)
                ClearRouteData();
            activeSourceKind = session.sourceKind;
            activeSessionId = session.sessionId;
        }
        routeOwnedByExternalAdapter = true;
    }

    private void HandleSnapshot(TwinSnapshot snapshot)
    {
        TwinSessionInfo session = snapshot?.metadata?.session;
        if (session == null ||
            (session.sourceKind != TwinSourceKind.SimulatedStream &&
             session.sourceKind != TwinSourceKind.LiveSensor))
        {
            if (activeSourceKind != TwinSourceKind.Unknown || routePoints.Count > 0)
                ClearRouteData();
            activeSessionId = null;
            activeSourceKind = session?.sourceKind ?? TwinSourceKind.Unknown;
            routeOwnedByExternalAdapter = false;
            return;
        }

        if (activeSourceKind != session.sourceKind || activeSessionId != session.sessionId)
        {
            ClearRouteData();
            routeOwnedByExternalAdapter = false;
            activeSourceKind = session.sourceKind;
            activeSessionId = session.sessionId;
        }

        if (routeOwnedByExternalAdapter || snapshot.metadata.freshness != TwinDataFreshness.Fresh ||
            snapshot.ego == null ||
            (snapshot.ego.validity != TwinDataValidity.Valid &&
             snapshot.ego.validity != TwinDataValidity.Partial))
            return;

        // A streaming adapter may publish an optional look-ahead route. This
        // is preferred over the historical ego breadcrumb because it lets the
        // road curve before the vehicle reaches it. Older sources can omit the
        // field and keep the existing breadcrumb behavior.
        if (TryApplySnapshotRoute(snapshot.route))
            return;

        // Route-ahead geometry is intentionally published less frequently
        // than 30 Hz telemetry. Once received, retain it when later snapshots
        // omit the optional route field. A source that never supplies a route
        // still uses the ego-breadcrumb fallback below.
        if (streamedRouteId != null)
            return;

        Transform coordinateRoot = canonicalCoordinateRoot != null ? canonicalCoordinateRoot : transform;
        Vector3 worldPoint = coordinateRoot.TransformPoint(snapshot.ego.position);
        if (AppendPointInternal(worldPoint))
            RebuildSurface();
    }

    private bool TryApplySnapshotRoute(TwinRouteState route)
    {
        if (route == null ||
            (route.validity != TwinDataValidity.Valid &&
             route.validity != TwinDataValidity.Partial) ||
            route.points == null || route.points.Length < 2)
            return false;

        string routeId = route.routeId ?? string.Empty;
        if (string.Equals(streamedRouteId, routeId, System.StringComparison.Ordinal) &&
            streamedRouteRevision == route.revision)
            return true;

        routePoints.Clear();
        renderedRoutePoints.Clear();
        Transform coordinateRoot = canonicalCoordinateRoot != null ? canonicalCoordinateRoot : transform;
        for (int index = 0; index < route.points.Length; index++)
            AppendPointInternal(coordinateRoot.TransformPoint(route.points[index]));

        streamedRouteId = routeId;
        streamedRouteRevision = route.revision;
        RebuildSurface();
        return true;
    }

    public void EnterTestMode()
    {
        // A future streamed map renderer can use this lifecycle hook. Route
        // delivery itself does not require a particular visualization.
    }

    public void ExitTestMode()
    {
    }

    public bool TryBuildTestRoute(
        Vector3 worldStartPosition,
        float sourceTimeSeconds,
        float minimumPointSpacing,
        List<Vector3> worldRoute)
    {
        if (worldRoute == null)
            return false;

        worldRoute.Clear();
        worldRoute.Add(worldStartPosition);
        if (!IsRouteAvailable)
            return false;

        int closestIndex = FindClosestPointIndex(worldStartPosition);
        float outputSpacing = Mathf.Max(0.1f, minimumPointSpacing);
        Vector3 lastOutput = worldStartPosition;
        for (int index = closestIndex + 1; index < routePoints.Count; index++)
        {
            Vector3 candidate = routePoints[index];
            if (Vector3.Distance(lastOutput, candidate) < outputSpacing)
                continue;
            worldRoute.Add(candidate);
            lastOutput = candidate;
        }

        AppendBoundedContinuation(worldRoute, FindBufferedForwardDirection(closestIndex));
        return worldRoute.Count > 1;
    }

    private Vector3 FindBufferedForwardDirection(int anchorIndex)
    {
        Vector3 anchor = routePoints[anchorIndex];
        for (int index = anchorIndex - 1; index >= 0; index--)
        {
            Vector3 direction = anchor - routePoints[index];
            direction.y = 0f;
            if (direction.sqrMagnitude > 0.001f)
                return direction.normalized;
        }

        for (int index = anchorIndex + 1; index < routePoints.Count; index++)
        {
            Vector3 direction = routePoints[index] - anchor;
            direction.y = 0f;
            if (direction.sqrMagnitude > 0.001f)
                return direction.normalized;
        }

        return Vector3.zero;
    }

    private int FindClosestPointIndex(Vector3 worldPosition)
    {
        int closestIndex = 0;
        float closestDistance = float.PositiveInfinity;
        for (int index = 0; index < routePoints.Count; index++)
        {
            float distance = (routePoints[index] - worldPosition).sqrMagnitude;
            if (distance >= closestDistance)
                continue;
            closestDistance = distance;
            closestIndex = index;
        }
        return closestIndex;
    }

    private bool AppendPointInternal(Vector3 worldPoint)
    {
        if (routePoints.Count > 0 &&
            Vector3.Distance(routePoints[routePoints.Count - 1], worldPoint) < incomingPointSpacing)
            return false;

        routePoints.Add(worldPoint);
        int overflow = routePoints.Count - Mathf.Max(2, maximumBufferedPoints);
        if (overflow > 0)
            routePoints.RemoveRange(0, overflow);
        return true;
    }

    private void RebuildSurface()
    {
        EnsureRouteSurface();
        renderedRoutePoints.Clear();
        renderedRoutePoints.AddRange(routePoints);
        AppendBoundedContinuation(renderedRoutePoints, Vector3.zero);
        routeSurface.Rebuild(
            renderedRoutePoints,
            roadWidth,
            egoRightLaneOffset,
            roadYOffset,
            footpathWidth,
            footpathHeight,
            roadColor,
            footpathColor,
            createMeshColliders);
    }

    private void EnsureRouteSurface()
    {
        if (routeSurface == null)
            routeSurface = GetComponent<ProceduralRouteSurface>();
        if (routeSurface == null)
            routeSurface = gameObject.AddComponent<ProceduralRouteSurface>();
    }

    private void AppendBoundedContinuation(List<Vector3> worldRoute, Vector3 fallbackDirection)
    {
        if (worldRoute.Count == 0 || continuationLength <= 0f)
            return;

        Vector3 direction = worldRoute.Count > 1
            ? worldRoute[worldRoute.Count - 1] - worldRoute[worldRoute.Count - 2]
            : fallbackDirection;
        direction.y = 0f;
        if (direction.sqrMagnitude <= 0.001f)
            return;
        direction.Normalize();

        Vector3 start = worldRoute[worldRoute.Count - 1];
        float spacing = Mathf.Max(0.5f, continuationPointSpacing);
        int steps = Mathf.CeilToInt(continuationLength / spacing);
        for (int step = 1; step <= steps; step++)
        {
            float distance = Mathf.Min(step * spacing, continuationLength);
            worldRoute.Add(start + direction * distance);
        }
    }
}

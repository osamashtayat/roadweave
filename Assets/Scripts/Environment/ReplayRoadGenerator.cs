using System.Collections.Generic;
using UnityEngine;

public class ReplayRoadGenerator : MonoBehaviour, ITestWorldRouteProvider
{
    public string ProviderName => "Replay road";
    public bool IsRouteAvailable => roadPoints.Count > 1;

    [Header("RoadWeave Reference")]
    [SerializeField] private DigitalTwinStateManager stateManager;

    [Header("Road Shape")]
    [SerializeField, Min(2f)] private float roadWidth = 12f;
    [SerializeField, Min(0.25f)] private float pointSpacing = 0.8f;
    [SerializeField] private float roadYOffset = -0.06f;
    [SerializeField, Min(0.5f)] private float textureLengthMeters = 4f;
    [SerializeField] private bool createMeshCollider = true;
    [Tooltip("Places the recorded ego trajectory in the right lane instead of on the center line.")]
    [SerializeField, Min(0f)] private float egoRightLaneOffset = 2.75f;

    [Header("Test Route Continuation")]
    [SerializeField, Min(25f)] private float continuationLength = 250f;
    [SerializeField, Min(0.5f)] private float continuationPointSpacing = 2f;
    [SerializeField, Min(1f)] private float continuationReferenceSpeed = 6f;

    [Header("Road Appearance")]
    [SerializeField] private Material roadMaterial;
    [SerializeField] private Color roadColor = new Color(0.12f, 0.13f, 0.14f, 1f);
    [SerializeField] private bool createEdgeLines = true;
    [SerializeField] private Material edgeLineMaterial;
    [SerializeField] private Color edgeLineColor = Color.white;
    [SerializeField, Min(0.02f)] private float edgeLineWidth = 0.12f;
    [SerializeField] private bool createCenterLine = true;
    [SerializeField] private Color centerLineColor = new Color(1f, 0.82f, 0.12f, 1f);
    [SerializeField, Min(0.02f)] private float centerLineWidth = 0.1f;

    [Header("Sidewalks")]
    [SerializeField] private bool createSidewalks = true;
    [SerializeField, Min(0.5f)] private float sidewalkWidth = 2.25f;
    [SerializeField, Min(0.01f)] private float sidewalkHeight = 0.14f;
    [SerializeField] private Material sidewalkMaterial;
    [SerializeField] private Color sidewalkColor = new Color(0.46f, 0.47f, 0.49f, 1f);

    [Header("Road Reveal")]
    [Tooltip("When enabled, more road becomes visible as replay time advances.")]
    [SerializeField] private bool revealGradually = true;
    [SerializeField, Min(0f)] private float revealAheadSeconds = 3f;

    private readonly List<Vector3> roadPoints = new List<Vector3>();
    private readonly List<float> roadTimes = new List<float>();
    private Vector3[] leftEdgePoints;
    private Vector3[] rightEdgePoints;
    private int[] completeTriangles;
    private Mesh roadMesh;
    private MeshCollider roadCollider;
    private LineRenderer leftEdgeLine;
    private LineRenderer rightEdgeLine;
    private LineRenderer centerLine;
    private Vector3[] centerLinePoints;
    private Vector3[] leftSidewalkOuterPoints;
    private Vector3[] leftSidewalkInnerPoints;
    private Vector3[] rightSidewalkInnerPoints;
    private Vector3[] rightSidewalkOuterPoints;
    private Mesh leftSidewalkMesh;
    private Mesh rightSidewalkMesh;
    private MeshCollider leftSidewalkCollider;
    private MeshCollider rightSidewalkCollider;
    private GameObject generatedRoadRoot;
    private Material generatedRoadMaterial;
    private Material generatedLineMaterial;
    private Material generatedCenterLineMaterial;
    private Material generatedSidewalkMaterial;
    private int currentVisiblePointCount = -1;
    private bool showCompleteRoadOverride;

    private void Awake()
    {
        roadWidth = Mathf.Max(12f, roadWidth);
        if (stateManager == null)
            stateManager = FindFirstObjectByType<DigitalTwinStateManager>();
    }

    private void OnEnable()
    {
        if (stateManager == null)
            stateManager = FindFirstObjectByType<DigitalTwinStateManager>();

        if (stateManager != null)
            stateManager.Initialized += BuildRoad;
    }

    private void Start()
    {
        if (stateManager != null && stateManager.IsReady)
            BuildRoad();
    }

    private void OnDisable()
    {
        if (stateManager != null)
            stateManager.Initialized -= BuildRoad;
    }

    private void Update()
    {
        if (roadMesh == null || stateManager == null || !stateManager.IsReady)
            return;

        int visiblePoints = revealGradually && !showCompleteRoadOverride
            ? FindVisiblePointCount(stateManager.CurrentTime + revealAheadSeconds)
            : roadPoints.Count;

        ApplyVisibility(visiblePoints);
    }

    public void BuildRoad()
    {
        ClearGeneratedRoad();

        if (stateManager == null || stateManager.Package == null)
            return;

        EgoReplayFrame[] frames = stateManager.Package.egoFrames;
        if (frames == null || frames.Length < 2)
        {
            Debug.LogWarning("ReplayRoadGenerator needs at least two ego frames.");
            return;
        }

        CollectRoadPoints(frames);
        if (roadPoints.Count < 2)
        {
            Debug.LogWarning("ReplayRoadGenerator could not create a usable route.");
            return;
        }

        generatedRoadRoot = new GameObject("GeneratedRoad");
        generatedRoadRoot.transform.SetParent(transform, false);

        CreateRoadSurface();
        if (createSidewalks)
            CreateSidewalkSurfaces();
        if (createEdgeLines)
            CreateRoadEdgeLines();
        if (createCenterLine)
            CreateRoadCenterLine();

        int startingPointCount = revealGradually
            ? FindVisiblePointCount(revealAheadSeconds)
            : roadPoints.Count;
        ApplyVisibility(startingPointCount);

        Debug.Log($"ReplayRoadGenerator created a road with {roadPoints.Count} route points.");
    }

    public void ShowCompleteRoad()
    {
        showCompleteRoadOverride = true;
        if (roadMesh != null)
            ApplyVisibility(roadPoints.Count);
    }

    public void ResumeReplayReveal()
    {
        showCompleteRoadOverride = false;
        currentVisiblePointCount = -1;
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
        float spacing = Mathf.Max(0.1f, minimumPointSpacing);
        Vector3 lastPoint = worldStartPosition;

        for (int index = 0; index < roadPoints.Count; index++)
        {
            if (index < roadTimes.Count && roadTimes[index] < sourceTimeSeconds)
                continue;

            Vector3 worldPoint = transform.TransformPoint(roadPoints[index]);
            if (Vector3.Distance(lastPoint, worldPoint) < spacing)
                continue;

            worldRoute.Add(worldPoint);
            lastPoint = worldPoint;
        }

        return worldRoute.Count > 1;
    }

    public void EnterTestMode()
    {
        ShowCompleteRoad();
    }

    public void ExitTestMode()
    {
        ResumeReplayReveal();
    }

    private void CollectRoadPoints(EgoReplayFrame[] frames)
    {
        roadPoints.Clear();
        roadTimes.Clear();

        for (int index = 0; index < frames.Length; index++)
        {
            Vector3 point = frames[index].position.ToVector3();
            point.y += roadYOffset;

            bool isFirst = roadPoints.Count == 0;
            bool isLastFrame = index == frames.Length - 1;
            bool farEnough = isFirst || Vector3.Distance(roadPoints[roadPoints.Count - 1], point) >= pointSpacing;

            if (isFirst || farEnough || isLastFrame)
            {
                if (!isFirst && Vector3.Distance(roadPoints[roadPoints.Count - 1], point) < 0.01f)
                    continue;

                roadPoints.Add(point);
                roadTimes.Add(frames[index].time);
            }
        }

        AppendRoadContinuation();
    }

    private void AppendRoadContinuation()
    {
        if (roadPoints.Count < 2 || continuationLength <= 0f)
            return;

        Vector3 direction = roadPoints[roadPoints.Count - 1] - roadPoints[roadPoints.Count - 2];
        direction.y = 0f;
        if (direction.sqrMagnitude <= 0.001f)
            return;
        direction.Normalize();

        Vector3 startingPoint = roadPoints[roadPoints.Count - 1];
        float startingTime = roadTimes[roadTimes.Count - 1];
        int stepCount = Mathf.CeilToInt(continuationLength / continuationPointSpacing);
        for (int step = 1; step <= stepCount; step++)
        {
            float distance = Mathf.Min(step * continuationPointSpacing, continuationLength);
            roadPoints.Add(startingPoint + direction * distance);
            roadTimes.Add(startingTime + distance / continuationReferenceSpeed);
        }
    }

    private void CreateRoadSurface()
    {
        GameObject surface = new GameObject("RoadSurface");
        surface.transform.SetParent(generatedRoadRoot.transform, false);

        MeshFilter meshFilter = surface.AddComponent<MeshFilter>();
        MeshRenderer meshRenderer = surface.AddComponent<MeshRenderer>();
        surface.AddComponent<RoadSurfaceMarker>();
        roadMesh = new Mesh { name = "RoadWeave Generated Replay Road" };
        roadMesh.MarkDynamic();
        meshFilter.sharedMesh = roadMesh;

        int pointCount = roadPoints.Count;
        Vector3[] vertices = new Vector3[pointCount * 2];
        Vector2[] uv = new Vector2[vertices.Length];
        leftEdgePoints = new Vector3[pointCount];
        rightEdgePoints = new Vector3[pointCount];
        centerLinePoints = new Vector3[pointCount];
        leftSidewalkOuterPoints = new Vector3[pointCount];
        leftSidewalkInnerPoints = new Vector3[pointCount];
        rightSidewalkInnerPoints = new Vector3[pointCount];
        rightSidewalkOuterPoints = new Vector3[pointCount];
        completeTriangles = new int[(pointCount - 1) * 6];
        float accumulatedDistance = 0f;

        for (int index = 0; index < pointCount; index++)
        {
            Vector3 previous = roadPoints[Mathf.Max(0, index - 1)];
            Vector3 next = roadPoints[Mathf.Min(pointCount - 1, index + 1)];
            Vector3 tangent = next - previous;
            tangent.y = 0f;
            if (tangent.sqrMagnitude < 0.0001f)
                tangent = Vector3.forward;
            tangent.Normalize();

            Vector3 right = Vector3.Cross(Vector3.up, tangent).normalized;
            Vector3 roadCenterPoint = roadPoints[index] - right * egoRightLaneOffset;
            Vector3 leftPoint = roadCenterPoint - right * (roadWidth * 0.5f);
            Vector3 rightPoint = roadCenterPoint + right * (roadWidth * 0.5f);
            vertices[index * 2] = leftPoint;
            vertices[index * 2 + 1] = rightPoint;
            leftEdgePoints[index] = leftPoint + Vector3.up * 0.015f;
            rightEdgePoints[index] = rightPoint + Vector3.up * 0.015f;
            centerLinePoints[index] = roadCenterPoint + Vector3.up * 0.02f;
            Vector3 sidewalkLift = Vector3.up * sidewalkHeight;
            leftSidewalkInnerPoints[index] = leftPoint + sidewalkLift;
            leftSidewalkOuterPoints[index] = leftPoint - right * sidewalkWidth + sidewalkLift;
            rightSidewalkInnerPoints[index] = rightPoint + sidewalkLift;
            rightSidewalkOuterPoints[index] = rightPoint + right * sidewalkWidth + sidewalkLift;

            if (index > 0)
                accumulatedDistance += Vector3.Distance(roadPoints[index - 1], roadPoints[index]);

            float v = accumulatedDistance / textureLengthMeters;
            uv[index * 2] = new Vector2(0f, v);
            uv[index * 2 + 1] = new Vector2(1f, v);
        }

        for (int segment = 0; segment < pointCount - 1; segment++)
        {
            int vertex = segment * 2;
            int triangle = segment * 6;
            completeTriangles[triangle] = vertex;
            completeTriangles[triangle + 1] = vertex + 2;
            completeTriangles[triangle + 2] = vertex + 1;
            completeTriangles[triangle + 3] = vertex + 1;
            completeTriangles[triangle + 4] = vertex + 2;
            completeTriangles[triangle + 5] = vertex + 3;
        }

        roadMesh.vertices = vertices;
        roadMesh.uv = uv;
        roadMesh.triangles = completeTriangles;
        roadMesh.RecalculateNormals();
        roadMesh.RecalculateBounds();

        meshRenderer.sharedMaterial = roadMaterial != null
            ? roadMaterial
            : CreateRuntimeMaterial("RoadWeave Road Material", roadColor, false);

        if (roadMaterial == null)
            generatedRoadMaterial = meshRenderer.sharedMaterial;

        if (createMeshCollider)
        {
            roadCollider = surface.AddComponent<MeshCollider>();
            roadCollider.sharedMesh = roadMesh;
        }
    }

    private void CreateSidewalkSurfaces()
    {
        Material material = sidewalkMaterial;
        if (material == null)
        {
            generatedSidewalkMaterial = CreateRuntimeMaterial(
                "RoadWeave Sidewalk Material",
                sidewalkColor,
                false
            );
            material = generatedSidewalkMaterial;
        }

        CreateSidewalkStrip(
            "LeftSidewalk",
            leftSidewalkOuterPoints,
            leftSidewalkInnerPoints,
            material,
            out leftSidewalkMesh,
            out leftSidewalkCollider
        );
        CreateSidewalkStrip(
            "RightSidewalk",
            rightSidewalkInnerPoints,
            rightSidewalkOuterPoints,
            material,
            out rightSidewalkMesh,
            out rightSidewalkCollider
        );
    }

    private void CreateSidewalkStrip(
        string objectName,
        Vector3[] leftBoundary,
        Vector3[] rightBoundary,
        Material material,
        out Mesh mesh,
        out MeshCollider meshCollider)
    {
        GameObject sidewalk = new GameObject(objectName);
        sidewalk.transform.SetParent(generatedRoadRoot.transform, false);
        sidewalk.AddComponent<RoadSurfaceMarker>();

        MeshFilter meshFilter = sidewalk.AddComponent<MeshFilter>();
        MeshRenderer meshRenderer = sidewalk.AddComponent<MeshRenderer>();
        mesh = new Mesh { name = $"RoadWeave {objectName} Mesh" };
        mesh.MarkDynamic();

        Vector3[] vertices = new Vector3[leftBoundary.Length * 2];
        Vector2[] uv = new Vector2[vertices.Length];
        float accumulatedDistance = 0f;
        for (int index = 0; index < leftBoundary.Length; index++)
        {
            vertices[index * 2] = leftBoundary[index];
            vertices[index * 2 + 1] = rightBoundary[index];
            if (index > 0)
                accumulatedDistance += Vector3.Distance(leftBoundary[index - 1], leftBoundary[index]);
            uv[index * 2] = new Vector2(0f, accumulatedDistance / 2f);
            uv[index * 2 + 1] = new Vector2(1f, accumulatedDistance / 2f);
        }

        mesh.vertices = vertices;
        mesh.uv = uv;
        mesh.triangles = completeTriangles;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        meshFilter.sharedMesh = mesh;
        meshRenderer.sharedMaterial = material;

        meshCollider = null;
        if (createMeshCollider)
        {
            meshCollider = sidewalk.AddComponent<MeshCollider>();
            meshCollider.sharedMesh = mesh;
        }
    }

    private void CreateRoadEdgeLines()
    {
        Material material = edgeLineMaterial;
        if (material == null)
        {
            generatedLineMaterial = CreateRuntimeMaterial("RoadWeave Edge Line Material", edgeLineColor, true);
            material = generatedLineMaterial;
        }

        leftEdgeLine = CreateEdgeLine("LeftEdgeLine", material);
        rightEdgeLine = CreateEdgeLine("RightEdgeLine", material);
    }

    private LineRenderer CreateEdgeLine(string objectName, Material material)
    {
        GameObject lineObject = new GameObject(objectName);
        lineObject.transform.SetParent(generatedRoadRoot.transform, false);
        LineRenderer line = lineObject.AddComponent<LineRenderer>();
        line.useWorldSpace = false;
        line.widthMultiplier = edgeLineWidth;
        line.numCornerVertices = 2;
        line.numCapVertices = 2;
        line.textureMode = LineTextureMode.Tile;
        line.sharedMaterial = material;
        line.startColor = edgeLineColor;
        line.endColor = edgeLineColor;
        return line;
    }

    private void CreateRoadCenterLine()
    {
        generatedCenterLineMaterial = CreateRuntimeMaterial(
            "RoadWeave Center Line Material",
            centerLineColor,
            true
        );
        centerLine = CreateEdgeLine("CenterLaneLine", generatedCenterLineMaterial);
        centerLine.widthMultiplier = centerLineWidth;
        centerLine.startColor = centerLineColor;
        centerLine.endColor = centerLineColor;
    }

    private int FindVisiblePointCount(float targetTime)
    {
        int count = 2;
        for (int index = 0; index < roadTimes.Count; index++)
        {
            if (roadTimes[index] <= targetTime)
                count = index + 1;
            else
                break;
        }
        return Mathf.Clamp(count, 2, roadPoints.Count);
    }

    private void ApplyVisibility(int visiblePointCount)
    {
        visiblePointCount = Mathf.Clamp(visiblePointCount, 2, roadPoints.Count);
        if (visiblePointCount == currentVisiblePointCount)
            return;

        currentVisiblePointCount = visiblePointCount;
        int[] visibleTriangles = new int[(visiblePointCount - 1) * 6];
        System.Array.Copy(completeTriangles, visibleTriangles, visibleTriangles.Length);
        roadMesh.triangles = visibleTriangles;
        roadMesh.RecalculateBounds();

        if (roadCollider != null)
        {
            roadCollider.sharedMesh = null;
            roadCollider.sharedMesh = roadMesh;
        }

        ApplyStripVisibility(leftSidewalkMesh, leftSidewalkCollider, visibleTriangles);
        ApplyStripVisibility(rightSidewalkMesh, rightSidewalkCollider, visibleTriangles);

        UpdateEdgeLine(leftEdgeLine, leftEdgePoints, visiblePointCount);
        UpdateEdgeLine(rightEdgeLine, rightEdgePoints, visiblePointCount);
        UpdateEdgeLine(centerLine, centerLinePoints, visiblePointCount);
    }

    private static void ApplyStripVisibility(Mesh mesh, MeshCollider collider, int[] visibleTriangles)
    {
        if (mesh == null)
            return;

        mesh.triangles = visibleTriangles;
        mesh.RecalculateBounds();
        if (collider != null)
        {
            collider.sharedMesh = null;
            collider.sharedMesh = mesh;
        }
    }

    private static void UpdateEdgeLine(LineRenderer line, Vector3[] points, int visiblePointCount)
    {
        if (line == null || points == null)
            return;

        line.positionCount = visiblePointCount;
        for (int index = 0; index < visiblePointCount; index++)
            line.SetPosition(index, points[index]);
    }

    private static Material CreateRuntimeMaterial(string materialName, Color color, bool unlit)
    {
        Shader shader = unlit
            ? Shader.Find("Universal Render Pipeline/Unlit")
            : Shader.Find("Universal Render Pipeline/Lit");

        if (shader == null)
            shader = Shader.Find(unlit ? "Unlit/Color" : "Standard");

        Material material = new Material(shader) { name = materialName, color = color };
        if (material.HasProperty("_BaseColor"))
            material.SetColor("_BaseColor", color);
        if (material.HasProperty("_Smoothness"))
            material.SetFloat("_Smoothness", 0.05f);
        return material;
    }

    private void ClearGeneratedRoad()
    {
        roadPoints.Clear();
        roadTimes.Clear();
        currentVisiblePointCount = -1;

        if (generatedRoadRoot != null)
            Destroy(generatedRoadRoot);
        if (roadMesh != null)
            Destroy(roadMesh);
        if (leftSidewalkMesh != null)
            Destroy(leftSidewalkMesh);
        if (rightSidewalkMesh != null)
            Destroy(rightSidewalkMesh);
        if (generatedRoadMaterial != null)
            Destroy(generatedRoadMaterial);
        if (generatedLineMaterial != null)
            Destroy(generatedLineMaterial);
        if (generatedCenterLineMaterial != null)
            Destroy(generatedCenterLineMaterial);
        if (generatedSidewalkMaterial != null)
            Destroy(generatedSidewalkMaterial);

        generatedRoadRoot = null;
        roadMesh = null;
        roadCollider = null;
        leftSidewalkMesh = null;
        rightSidewalkMesh = null;
        leftSidewalkCollider = null;
        rightSidewalkCollider = null;
        leftEdgeLine = null;
        rightEdgeLine = null;
        centerLine = null;
        centerLinePoints = null;
        leftSidewalkOuterPoints = null;
        leftSidewalkInnerPoints = null;
        rightSidewalkInnerPoints = null;
        rightSidewalkOuterPoints = null;
        generatedRoadMaterial = null;
        generatedLineMaterial = null;
        generatedCenterLineMaterial = null;
        generatedSidewalkMaterial = null;
    }

    private void OnDestroy()
    {
        ClearGeneratedRoad();
    }
}

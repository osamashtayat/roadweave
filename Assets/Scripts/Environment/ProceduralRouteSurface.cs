using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Builds a two-lane road ribbon and raised footpaths from ordered world-space
/// route points. It contains no replay or streaming knowledge and can be reused
/// by future route providers.
/// </summary>
[DisallowMultipleComponent]
public sealed class ProceduralRouteSurface : MonoBehaviour
{
    private GameObject generatedRoot;
    private Mesh roadMesh;
    private Mesh leftFootpathMesh;
    private Mesh rightFootpathMesh;
    private MeshCollider roadCollider;
    private MeshCollider leftFootpathCollider;
    private MeshCollider rightFootpathCollider;
    private Material roadMaterial;
    private Material footpathMaterial;

    public int RoadVertexCount => roadMesh != null ? roadMesh.vertexCount : 0;
    public int FootpathVertexCount =>
        (leftFootpathMesh != null ? leftFootpathMesh.vertexCount : 0) +
        (rightFootpathMesh != null ? rightFootpathMesh.vertexCount : 0);

    public void Rebuild(
        IReadOnlyList<Vector3> worldPoints,
        float roadWidth,
        float egoRightLaneOffset,
        float roadYOffset,
        float footpathWidth,
        float footpathHeight,
        Color roadColor,
        Color footpathColor,
        bool createColliders)
    {
        EnsureObjects(roadColor, footpathColor, createColliders);
        if (worldPoints == null || worldPoints.Count < 2)
        {
            Clear();
            return;
        }

        int pointCount = worldPoints.Count;
        Vector3[] roadVertices = new Vector3[pointCount * 2];
        Vector3[] leftVertices = new Vector3[pointCount * 2];
        Vector3[] rightVertices = new Vector3[pointCount * 2];
        Vector2[] uv = new Vector2[pointCount * 2];
        int[] triangles = BuildTriangles(pointCount);
        float halfRoadWidth = Mathf.Max(3f, roadWidth * 0.5f);
        float safeFootpathWidth = Mathf.Max(0.5f, footpathWidth);
        float accumulatedDistance = 0f;

        for (int index = 0; index < pointCount; index++)
        {
            Vector3 previous = worldPoints[Mathf.Max(0, index - 1)];
            Vector3 next = worldPoints[Mathf.Min(pointCount - 1, index + 1)];
            Vector3 tangent = next - previous;
            tangent.y = 0f;
            if (tangent.sqrMagnitude <= 0.0001f)
                tangent = index > 0
                    ? worldPoints[index] - worldPoints[index - 1]
                    : Vector3.forward;
            tangent.y = 0f;
            tangent = tangent.sqrMagnitude > 0.0001f ? tangent.normalized : Vector3.forward;
            Vector3 routeRight = Vector3.Cross(Vector3.up, tangent).normalized;

            Vector3 roadCenter = worldPoints[index] - routeRight * egoRightLaneOffset;
            roadCenter.y += roadYOffset;
            Vector3 leftEdge = roadCenter - routeRight * halfRoadWidth;
            Vector3 rightEdge = roadCenter + routeRight * halfRoadWidth;
            Vector3 footpathLift = Vector3.up * Mathf.Max(0.01f, footpathHeight);

            roadVertices[index * 2] = transform.InverseTransformPoint(leftEdge);
            roadVertices[index * 2 + 1] = transform.InverseTransformPoint(rightEdge);
            leftVertices[index * 2] = transform.InverseTransformPoint(leftEdge - routeRight * safeFootpathWidth + footpathLift);
            leftVertices[index * 2 + 1] = transform.InverseTransformPoint(leftEdge + footpathLift);
            rightVertices[index * 2] = transform.InverseTransformPoint(rightEdge + footpathLift);
            rightVertices[index * 2 + 1] = transform.InverseTransformPoint(rightEdge + routeRight * safeFootpathWidth + footpathLift);

            if (index > 0)
                accumulatedDistance += Vector3.Distance(worldPoints[index - 1], worldPoints[index]);
            uv[index * 2] = new Vector2(0f, accumulatedDistance / 4f);
            uv[index * 2 + 1] = new Vector2(1f, accumulatedDistance / 4f);
        }

        ApplyMesh(roadMesh, roadVertices, uv, triangles);
        ApplyMesh(leftFootpathMesh, leftVertices, uv, triangles);
        ApplyMesh(rightFootpathMesh, rightVertices, uv, triangles);
        RefreshCollider(roadCollider, roadMesh, createColliders);
        RefreshCollider(leftFootpathCollider, leftFootpathMesh, createColliders);
        RefreshCollider(rightFootpathCollider, rightFootpathMesh, createColliders);
        generatedRoot.SetActive(true);
    }

    public void Clear()
    {
        if (roadMesh != null) roadMesh.Clear();
        if (leftFootpathMesh != null) leftFootpathMesh.Clear();
        if (rightFootpathMesh != null) rightFootpathMesh.Clear();
        if (roadCollider != null) roadCollider.sharedMesh = null;
        if (leftFootpathCollider != null) leftFootpathCollider.sharedMesh = null;
        if (rightFootpathCollider != null) rightFootpathCollider.sharedMesh = null;
        if (generatedRoot != null) generatedRoot.SetActive(false);
    }

    private void EnsureObjects(Color roadColor, Color footpathColor, bool createColliders)
    {
        if (generatedRoot != null)
            return;

        generatedRoot = new GameObject("GeneratedStreamingRoad");
        generatedRoot.transform.SetParent(transform, false);
        roadMaterial = CreateMaterial("Streaming Road", roadColor);
        footpathMaterial = CreateMaterial("Streaming Footpath", footpathColor);

        CreateStrip("RoadSurface", roadMaterial, createColliders, out roadMesh, out roadCollider);
        CreateStrip("LeftFootpath", footpathMaterial, createColliders, out leftFootpathMesh, out leftFootpathCollider);
        CreateStrip("RightFootpath", footpathMaterial, createColliders, out rightFootpathMesh, out rightFootpathCollider);
    }

    private void CreateStrip(
        string objectName,
        Material material,
        bool createCollider,
        out Mesh mesh,
        out MeshCollider meshCollider)
    {
        GameObject strip = new GameObject(objectName);
        strip.transform.SetParent(generatedRoot.transform, false);
        strip.AddComponent<RoadSurfaceMarker>();
        MeshFilter filter = strip.AddComponent<MeshFilter>();
        MeshRenderer renderer = strip.AddComponent<MeshRenderer>();
        mesh = new Mesh { name = $"RoadWeave {objectName}" };
        mesh.MarkDynamic();
        filter.sharedMesh = mesh;
        renderer.sharedMaterial = material;
        meshCollider = createCollider ? strip.AddComponent<MeshCollider>() : null;
    }

    private static int[] BuildTriangles(int pointCount)
    {
        int[] triangles = new int[(pointCount - 1) * 6];
        for (int segment = 0; segment < pointCount - 1; segment++)
        {
            int vertex = segment * 2;
            int triangle = segment * 6;
            triangles[triangle] = vertex;
            triangles[triangle + 1] = vertex + 2;
            triangles[triangle + 2] = vertex + 1;
            triangles[triangle + 3] = vertex + 1;
            triangles[triangle + 4] = vertex + 2;
            triangles[triangle + 5] = vertex + 3;
        }
        return triangles;
    }

    private static void ApplyMesh(Mesh mesh, Vector3[] vertices, Vector2[] uv, int[] triangles)
    {
        mesh.Clear();
        mesh.vertices = vertices;
        mesh.uv = uv;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
    }

    private static void RefreshCollider(MeshCollider collider, Mesh mesh, bool enabled)
    {
        if (collider == null)
            return;
        collider.enabled = enabled;
        collider.sharedMesh = null;
        if (enabled)
            collider.sharedMesh = mesh;
    }

    private static Material CreateMaterial(string materialName, Color color)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null) shader = Shader.Find("Standard");
        Material material = new Material(shader) { name = materialName, color = color };
        if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", color);
        if (material.HasProperty("_Smoothness")) material.SetFloat("_Smoothness", 0.05f);
        return material;
    }

    private void OnDestroy()
    {
        if (roadMesh != null) Destroy(roadMesh);
        if (leftFootpathMesh != null) Destroy(leftFootpathMesh);
        if (rightFootpathMesh != null) Destroy(rightFootpathMesh);
        if (roadMaterial != null) Destroy(roadMaterial);
        if (footpathMaterial != null) Destroy(footpathMaterial);
    }
}

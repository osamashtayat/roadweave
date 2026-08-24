using UnityEngine;

/// <summary>
/// Keeps source coordinates, identity, and colliders on a stable actor root
/// while imported model-specific scale and orientation remain on a child.
/// </summary>
[DisallowMultipleComponent]
public sealed class ActorVisualRoot : MonoBehaviour
{
    [SerializeField] private Transform visual;

    public Transform Visual => visual;

    public void SetVisual(Transform newVisual)
    {
        visual = newVisual;
        if (visual == null)
            return;

        visual.SetParent(transform, false);
        visual.localPosition = Vector3.zero;
    }

    public void FitVisualToWorldSize(Vector3 targetSize)
    {
        if (visual == null)
            return;

        Renderer[] renderers = visual.GetComponentsInChildren<Renderer>(true);
        if (renderers.Length == 0)
            return;

        Bounds bounds = renderers[0].bounds;
        for (int index = 1; index < renderers.Length; index++)
            bounds.Encapsulate(renderers[index].bounds);

        Vector3 currentSize = bounds.size;
        Vector3 scale = visual.localScale;
        scale.x *= SafeRatio(targetSize.x, currentSize.x);
        scale.y *= SafeRatio(targetSize.y, currentSize.y);
        scale.z *= SafeRatio(targetSize.z, currentSize.z);
        visual.localScale = scale;
    }

    public void CenterVisualOnActorRoot()
    {
        if (visual == null)
            return;

        Renderer[] renderers = visual.GetComponentsInChildren<Renderer>(true);
        bool foundBounds = false;
        Bounds bounds = default;
        foreach (Renderer visualRenderer in renderers)
        {
            if (visualRenderer == null || !visualRenderer.enabled)
                continue;
            if (!foundBounds)
            {
                bounds = visualRenderer.bounds;
                foundBounds = true;
            }
            else
            {
                bounds.Encapsulate(visualRenderer.bounds);
            }
        }

        if (!foundBounds)
            return;

        Vector3 centerInActorSpace = transform.InverseTransformPoint(bounds.center);
        visual.localPosition -= centerInActorSpace;
    }

    private static float SafeRatio(float target, float current)
    {
        return current > 0.001f && target > 0.001f ? target / current : 1f;
    }
}

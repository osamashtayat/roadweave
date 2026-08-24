using UnityEngine;

/// <summary>
/// Shared fallback visuals used when a source does not provide a model prefab.
/// Keeping this outside Test Lab makes the same pedestrian available to Live Twin.
/// </summary>
public static class FallbackActorVisualFactory
{
    public static GameObject CreatePedestrian()
    {
        GameObject root = new GameObject("FallbackHumanoid");
        Color clothing = new Color(0.12f, 0.38f, 0.82f);
        Color trousers = new Color(0.08f, 0.12f, 0.2f);
        Color skin = new Color(0.82f, 0.58f, 0.42f);

        CreatePart(root.transform, "Torso", PrimitiveType.Capsule,
            new Vector3(0f, 1.12f, 0f), new Vector3(0.42f, 0.5f, 0.3f), Quaternion.identity, clothing);
        CreatePart(root.transform, "Head", PrimitiveType.Sphere,
            new Vector3(0f, 1.78f, 0f), Vector3.one * 0.36f, Quaternion.identity, skin);
        CreatePart(root.transform, "LeftArm", PrimitiveType.Cylinder,
            new Vector3(-0.36f, 1.12f, 0f), new Vector3(0.09f, 0.42f, 0.09f),
            Quaternion.Euler(0f, 0f, -8f), clothing);
        CreatePart(root.transform, "RightArm", PrimitiveType.Cylinder,
            new Vector3(0.36f, 1.12f, 0f), new Vector3(0.09f, 0.42f, 0.09f),
            Quaternion.Euler(0f, 0f, 8f), clothing);
        CreatePart(root.transform, "LeftLeg", PrimitiveType.Cylinder,
            new Vector3(-0.16f, 0.43f, 0f), new Vector3(0.11f, 0.42f, 0.11f), Quaternion.identity, trousers);
        CreatePart(root.transform, "RightLeg", PrimitiveType.Cylinder,
            new Vector3(0.16f, 0.43f, 0f), new Vector3(0.11f, 0.42f, 0.11f), Quaternion.identity, trousers);
        return root;
    }

    private static void CreatePart(
        Transform parent,
        string partName,
        PrimitiveType primitive,
        Vector3 localPosition,
        Vector3 localScale,
        Quaternion localRotation,
        Color color)
    {
        GameObject part = GameObject.CreatePrimitive(primitive);
        part.name = partName;
        part.transform.SetParent(parent, false);
        part.transform.localPosition = localPosition;
        part.transform.localScale = localScale;
        part.transform.localRotation = localRotation;

        Collider collider = part.GetComponent<Collider>();
        if (collider != null)
        {
            if (Application.isPlaying)
                Object.Destroy(collider);
            else
                Object.DestroyImmediate(collider);
        }

        Renderer renderer = part.GetComponent<Renderer>();
        if (renderer != null)
            renderer.material.color = color;
    }
}

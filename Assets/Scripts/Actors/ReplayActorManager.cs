using System.Collections.Generic;
using UnityEngine;

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

    private readonly List<SpawnedActor> spawnedActors = new List<SpawnedActor>();

    private class SpawnedActor
    {
        public ActorReplayTrack track;
        public GameObject gameObject;
    }

    private void Awake()
    {
        if (stateManager == null)
            stateManager = FindFirstObjectByType<DigitalTwinStateManager>();

        if (runtimeActorsRoot == null)
        {
            GameObject rootObject = new GameObject("RuntimeActors");
            rootObject.transform.SetParent(transform, false);
            runtimeActorsRoot = rootObject.transform;
        }
    }

    private void OnEnable()
    {
        if (stateManager == null)
            stateManager = FindFirstObjectByType<DigitalTwinStateManager>();

        if (stateManager != null)
            stateManager.Initialized += BuildActors;
    }

    private void Start()
    {
        if (stateManager != null && stateManager.IsReady)
            BuildActors();
    }

    private void OnDisable()
    {
        if (stateManager != null)
            stateManager.Initialized -= BuildActors;
    }

    private void LateUpdate()
    {
        if (stateManager == null || !stateManager.IsReady)
            return;

        float replayTime = stateManager.CurrentTime;
        float visibilityPadding = Mathf.Max(0.05f, stateManager.Package.sampleIntervalSeconds * 0.55f);

        foreach (SpawnedActor spawned in spawnedActors)
        {
            ActorReplayFrame[] frames = spawned.track.frames;
            if (frames == null || frames.Length == 0)
                continue;

            bool shouldBeVisible = replayTime >= frames[0].time - visibilityPadding
                && replayTime <= frames[frames.Length - 1].time + visibilityPadding;

            if (spawned.gameObject.activeSelf != shouldBeVisible)
                spawned.gameObject.SetActive(shouldBeVisible);

            if (!shouldBeVisible)
                continue;

            FindFramePair(frames, replayTime, out ActorReplayFrame from, out ActorReplayFrame to, out float t);
            spawned.gameObject.transform.localPosition = Vector3.Lerp(
                from.position.ToVector3(),
                to.position.ToVector3(),
                t
            );
            float yaw = Mathf.LerpAngle(from.yawDegrees, to.yawDegrees, t);
            spawned.gameObject.transform.localRotation = Quaternion.Euler(0f, yaw, 0f);
        }
    }

    private void BuildActors()
    {
        ClearSpawnedActors();

        if (stateManager == null || stateManager.Package == null || stateManager.Package.actors == null)
            return;

        foreach (ActorReplayTrack track in stateManager.Package.actors)
        {
            if (track.frames == null || track.frames.Length == 0)
                continue;

            GameObject prefab = GetPrefab(track.prefabType);
            bool isPlaceholder = prefab == null;
            GameObject actor = isPlaceholder
                ? CreatePlaceholder(track.prefabType)
                : Instantiate(prefab);

            actor.name = $"{track.prefabType}_{ShortId(track.id)}";
            actor.transform.SetParent(runtimeActorsRoot, false);

            if (isPlaceholder || scaleProvidedPrefabsToDatasetSize)
            {
                Vector3 datasetSize = track.size.ToVector3();
                // Unity capsules and cylinders are two units tall by default.
                if (isPlaceholder && (track.prefabType == "pedestrian" || track.prefabType == "trafficCone"))
                    datasetSize.y *= 0.5f;
                actor.transform.localScale = datasetSize;
            }

            EnsureActorCollider(actor, track);
            RegisterRoadActor(actor, track.prefabType);

            spawnedActors.Add(new SpawnedActor { track = track, gameObject = actor });
        }

        Debug.Log($"ReplayActorManager created {spawnedActors.Count} dataset actors.");
    }

    private GameObject GetPrefab(string prefabType)
    {
        switch (prefabType)
        {
            case "car": return carPrefab;
            case "truck": return truckPrefab;
            case "bus": return busPrefab;
            case "constructionVehicle": return constructionVehiclePrefab;
            case "pedestrian": return pedestrianPrefab;
            case "trafficCone": return trafficConePrefab;
            default: return genericObjectPrefab;
        }
    }

    private static GameObject CreatePlaceholder(string prefabType)
    {
        PrimitiveType primitive = prefabType == "pedestrian"
            ? PrimitiveType.Capsule
            : prefabType == "trafficCone" ? PrimitiveType.Cylinder : PrimitiveType.Cube;

        GameObject placeholder = GameObject.CreatePrimitive(primitive);
        Renderer renderer = placeholder.GetComponent<Renderer>();
        if (renderer != null)
            renderer.material.color = PlaceholderColor(prefabType);
        return placeholder;
    }

    private static void EnsureActorCollider(GameObject actor, ActorReplayTrack track)
    {
        // A prefab may contain only wheel or decorative mesh colliders. A root
        // safety box based on the dataset dimensions guarantees one continuous
        // obstacle volume for autonomous detection.
        if (actor.GetComponent<BoxCollider>() != null)
            return;

        Vector3 datasetSize = track.size != null
            ? track.size.ToVector3()
            : new Vector3(1.8f, 1.5f, 4.2f);
        Vector3 scale = actor.transform.localScale;
        BoxCollider collider = actor.AddComponent<BoxCollider>();
        collider.center = Vector3.zero;
        collider.size = new Vector3(
            datasetSize.x / Mathf.Max(0.001f, Mathf.Abs(scale.x)),
            datasetSize.y / Mathf.Max(0.001f, Mathf.Abs(scale.y)),
            datasetSize.z / Mathf.Max(0.001f, Mathf.Abs(scale.z))
        );
    }

    private static void RegisterRoadActor(GameObject actor, string prefabType)
    {
        bool isPedestrian = prefabType == "pedestrian";
        bool isVehicle = prefabType == "car"
            || prefabType == "truck"
            || prefabType == "bus"
            || prefabType == "constructionVehicle";
        if (!isPedestrian && !isVehicle)
            return;

        ScenarioActorMarker marker = actor.GetComponent<ScenarioActorMarker>();
        if (marker == null)
            marker = actor.AddComponent<ScenarioActorMarker>();

        ScenarioActorType actorType = isPedestrian
            ? ScenarioActorType.Pedestrian
            : prefabType == "car" ? ScenarioActorType.SlowCar : ScenarioActorType.Truck;
        marker.Configure(actorType, isVehicle);
    }

    private static Color PlaceholderColor(string prefabType)
    {
        switch (prefabType)
        {
            case "car": return new Color(0.15f, 0.45f, 0.95f);
            case "truck": return new Color(0.95f, 0.55f, 0.12f);
            case "bus": return new Color(0.95f, 0.82f, 0.12f);
            case "constructionVehicle": return new Color(0.95f, 0.75f, 0.1f);
            case "pedestrian": return new Color(0.2f, 0.85f, 0.35f);
            case "trafficCone": return new Color(1f, 0.3f, 0.05f);
            default: return Color.gray;
        }
    }

    private static string ShortId(string id)
    {
        if (string.IsNullOrEmpty(id))
            return "unknown";
        return id.Length <= 8 ? id : id.Substring(0, 8);
    }

    private static void FindFramePair(ActorReplayFrame[] frames, float time, out ActorReplayFrame from, out ActorReplayFrame to, out float t)
    {
        if (frames.Length == 1 || time <= frames[0].time)
        {
            from = to = frames[0];
            t = 0f;
            return;
        }

        if (time >= frames[frames.Length - 1].time)
        {
            from = to = frames[frames.Length - 1];
            t = 0f;
            return;
        }

        int low = 0;
        int high = frames.Length - 1;
        while (low <= high)
        {
            int middle = (low + high) / 2;
            if (frames[middle].time <= time)
                low = middle + 1;
            else
                high = middle - 1;
        }

        int fromIndex = Mathf.Clamp(high, 0, frames.Length - 1);
        int toIndex = Mathf.Min(fromIndex + 1, frames.Length - 1);
        from = frames[fromIndex];
        to = frames[toIndex];
        t = Mathf.Approximately(from.time, to.time)
            ? 0f
            : Mathf.InverseLerp(from.time, to.time, time);
    }

    private void ClearSpawnedActors()
    {
        foreach (SpawnedActor spawned in spawnedActors)
        {
            if (spawned.gameObject != null)
                Destroy(spawned.gameObject);
        }
        spawnedActors.Clear();
    }
}

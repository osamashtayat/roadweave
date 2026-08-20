using System.Collections.Generic;
using UnityEngine;

public class TestScenarioManager : MonoBehaviour
{
    [SerializeField] private Transform scenarioActorsRoot;
    [SerializeField] private GameObject pedestrianPrefab;
    [SerializeField] private GameObject carPrefab;
    [SerializeField] private GameObject truckPrefab;
    [SerializeField, Min(3f)] private float roadWidth = 12f;
    [SerializeField, Min(0f)] private float egoRightLaneOffset = 2.75f;

    [Header("Pedestrian Environment")]
    [SerializeField, Min(0.5f)] private float sidewalkWidth = 2.25f;
    [SerializeField, Min(0.01f)] private float sidewalkHeight = 0.14f;
    [SerializeField, Range(4, 12)] private int crosswalkStripeCount = 8;
    [SerializeField, Min(1f)] private float crosswalkLengthAlongRoad = 4f;
    [SerializeField, Min(0.1f)] private float crosswalkStripeDepth = 0.32f;

    public string CurrentScenarioName { get; private set; } = "Clear Route";

    private readonly List<GameObject> spawnedActors = new List<GameObject>();
    private readonly List<GameObject> spawnedDecorations = new List<GameObject>();
    private AutonomousTestVehicleController testVehicle;

    private void Awake()
    {
        roadWidth = Mathf.Max(12f, roadWidth);

        if (scenarioActorsRoot == null)
        {
            GameObject root = new GameObject("ScenarioActors");
            root.transform.SetParent(transform, false);
            scenarioActorsRoot = root.transform;
        }
    }

    public void SetTestVehicle(AutonomousTestVehicleController vehicle)
    {
        testVehicle = vehicle;
    }

    public void AddPedestrian()
    {
        SpawnPedestrian(20f);
    }

    public void AddStoppedCar()
    {
        SpawnStoppedVehicle(
            carPrefab,
            "StoppedCar",
            ScenarioActorType.StoppedCar,
            22f,
            new Vector3(1.8f, 1.5f, 4.2f),
            new Color(0.15f, 0.45f, 0.95f)
        );
    }

    public void AddTruck()
    {
        GameObject actor = SpawnStoppedVehicle(
            truckPrefab,
            "MovingTruck",
            ScenarioActorType.Truck,
            28f,
            new Vector3(2.5f, 3f, 8f),
            new Color(0.95f, 0.55f, 0.12f)
        );
        if (actor != null)
        {
            ScenarioForwardMover mover = actor.GetComponent<ScenarioForwardMover>();
            if (mover == null)
                mover = actor.AddComponent<ScenarioForwardMover>();
            mover.Configure(testVehicle, 3f);
            CurrentScenarioName = "Moving Truck Ahead";
        }
    }

    public void AddSlowCar()
    {
        GameObject actor = SpawnStoppedVehicle(
            carPrefab,
            "SlowCar",
            ScenarioActorType.SlowCar,
            25f,
            new Vector3(1.8f, 1.5f, 4.2f),
            new Color(0.15f, 0.45f, 0.95f)
        );
        if (actor != null)
        {
            ScenarioForwardMover mover = actor.AddComponent<ScenarioForwardMover>();
            mover.Configure(testVehicle, 4f);
            CurrentScenarioName = "Slow Car Ahead";
        }
    }

    public void ClearScenarios()
    {
        foreach (GameObject actor in spawnedActors)
        {
            if (actor != null)
                Destroy(actor);
        }
        spawnedActors.Clear();
        foreach (GameObject decoration in spawnedDecorations)
        {
            if (decoration != null)
                Destroy(decoration);
        }
        spawnedDecorations.Clear();
        CurrentScenarioName = "Clear Route";
    }

    private void SpawnPedestrian(float distanceAhead)
    {
        if (!CanSpawn())
            return;

        Transform vehicleTransform = testVehicle.transform;
        Vector3 vehicleForward = testVehicle.MovementForward;
        Vector3 vehicleRight = testVehicle.MovementRight;
        Vector3 crossingDirection = -vehicleRight;
        CreateCrosswalk(vehicleTransform, vehicleForward, vehicleRight, distanceAhead);
        Vector3 position = vehicleTransform.position
            + vehicleForward * distanceAhead
            + vehicleRight * (roadWidth * 0.5f + sidewalkWidth * 0.5f - egoRightLaneOffset);

        GameObject pedestrian = pedestrianPrefab != null
            ? Instantiate(pedestrianPrefab)
            : GameObject.CreatePrimitive(PrimitiveType.Capsule);

        pedestrian.name = "TestPedestrian";
        pedestrian.transform.SetParent(scenarioActorsRoot, true);
        pedestrian.transform.position = position + Vector3.up * (0.9f + sidewalkHeight);
        if (pedestrianPrefab == null)
        {
            pedestrian.transform.localScale = new Vector3(0.6f, 0.9f, 0.6f);
            SetColor(pedestrian, new Color(0.2f, 0.85f, 0.35f));
        }

        ScenarioActorMarker marker = pedestrian.GetComponent<ScenarioActorMarker>();
        if (marker == null)
            marker = pedestrian.AddComponent<ScenarioActorMarker>();
        marker.Configure(ScenarioActorType.Pedestrian, false);

        ScenarioPedestrianMover movement = pedestrian.GetComponent<ScenarioPedestrianMover>();
        if (movement == null)
            movement = pedestrian.AddComponent<ScenarioPedestrianMover>();
        movement.Configure(testVehicle, crossingDirection, roadWidth + sidewalkWidth, 1.4f);

        spawnedActors.Add(pedestrian);
        CurrentScenarioName = spawnedActors.Count > 1 ? "Mixed Scenario" : "Pedestrian Crossing";
    }

    private void CreateCrosswalk(
        Transform vehicleTransform,
        Vector3 vehicleForward,
        Vector3 vehicleRight,
        float distanceAhead)
    {
        GameObject crosswalkRoot = new GameObject("PedestrianCrosswalk");
        crosswalkRoot.transform.SetParent(scenarioActorsRoot, true);
        crosswalkRoot.transform.position = vehicleTransform.position
            + vehicleForward * distanceAhead
            - vehicleRight * egoRightLaneOffset;
        crosswalkRoot.transform.rotation = Quaternion.LookRotation(vehicleForward, Vector3.up);

        int stripeCount = Mathf.Max(4, crosswalkStripeCount);
        float spacing = crosswalkLengthAlongRoad / stripeCount;
        float firstOffset = -crosswalkLengthAlongRoad * 0.5f + spacing * 0.5f;
        for (int index = 0; index < stripeCount; index++)
        {
            GameObject stripe = GameObject.CreatePrimitive(PrimitiveType.Cube);
            stripe.name = $"CrosswalkStripe_{index + 1}";
            stripe.transform.SetParent(crosswalkRoot.transform, false);
            stripe.transform.localPosition = new Vector3(
                0f,
                0.025f,
                firstOffset + spacing * index
            );
            stripe.transform.localScale = new Vector3(
                roadWidth - 0.3f,
                0.025f,
                Mathf.Min(crosswalkStripeDepth, spacing * 0.75f)
            );
            stripe.AddComponent<RoadSurfaceMarker>();
            SetColor(stripe, new Color(0.94f, 0.94f, 0.9f));
        }

        spawnedDecorations.Add(crosswalkRoot);
    }

    private GameObject SpawnStoppedVehicle(
        GameObject prefab,
        string objectName,
        ScenarioActorType actorType,
        float distanceAhead,
        Vector3 placeholderSize,
        Color placeholderColor)
    {
        if (!CanSpawn())
            return null;

        Transform vehicleTransform = testVehicle.transform;
        GameObject actor = prefab != null
            ? Instantiate(prefab)
            : GameObject.CreatePrimitive(PrimitiveType.Cube);

        actor.name = objectName;
        actor.transform.SetParent(scenarioActorsRoot, true);
        actor.transform.position = vehicleTransform.position
            + testVehicle.MovementForward * distanceAhead
            + Vector3.up * (placeholderSize.y * 0.5f);
        actor.transform.rotation = Quaternion.LookRotation(testVehicle.MovementForward, Vector3.up);

        if (prefab == null)
        {
            actor.transform.localScale = placeholderSize;
            SetColor(actor, placeholderColor);
        }

        ScenarioActorMarker marker = actor.GetComponent<ScenarioActorMarker>();
        if (marker == null)
            marker = actor.AddComponent<ScenarioActorMarker>();
        marker.Configure(actorType, true);

        spawnedActors.Add(actor);
        CurrentScenarioName = spawnedActors.Count > 1 ? "Mixed Scenario" : objectName;
        return actor;
    }

    private bool CanSpawn()
    {
        if (testVehicle != null)
            return true;

        Debug.LogWarning("Create a test before adding scenario actors.");
        return false;
    }

    private static void SetColor(GameObject target, Color color)
    {
        Renderer renderer = target.GetComponentInChildren<Renderer>();
        if (renderer != null)
            renderer.material.color = color;
    }
}

public enum ScenarioActorType
{
    Pedestrian,
    StoppedCar,
    SlowCar,
    Truck
}

public class ScenarioActorMarker : MonoBehaviour
{
    private static readonly List<ScenarioActorMarker> activeActors = new List<ScenarioActorMarker>();

    public static IReadOnlyList<ScenarioActorMarker> ActiveActors => activeActors;
    public ScenarioActorType ActorType { get; private set; }
    public bool CanOvertake { get; private set; }
    public bool HasCrossingMotion { get; private set; }
    public bool CrossingComplete { get; private set; }

    private Rigidbody physicsBody;

    private void OnEnable()
    {
        if (!activeActors.Contains(this))
            activeActors.Add(this);
    }

    private void OnDisable()
    {
        activeActors.Remove(this);
    }

    public void Configure(ScenarioActorType actorType, bool canOvertake)
    {
        ActorType = actorType;
        CanOvertake = canOvertake;
        HasCrossingMotion = false;
        CrossingComplete = false;

        physicsBody = GetComponent<Rigidbody>();
        if (physicsBody == null)
            physicsBody = gameObject.AddComponent<Rigidbody>();
        physicsBody.isKinematic = true;
        physicsBody.useGravity = false;
        physicsBody.interpolation = RigidbodyInterpolation.Interpolate;
        physicsBody.collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;
    }

    public void BeginCrossing()
    {
        HasCrossingMotion = true;
        CrossingComplete = false;
    }

    public void CompleteCrossing()
    {
        if (HasCrossingMotion)
            CrossingComplete = true;
    }

    public void MoveTo(Vector3 position)
    {
        if (physicsBody != null)
            physicsBody.MovePosition(position);
        else
            transform.position = position;
    }
}

public class ScenarioPedestrianMover : MonoBehaviour
{
    private AutonomousTestVehicleController testVehicle;
    private Vector3 direction;
    private float remainingDistance;
    private float speed;
    private ScenarioActorMarker actorMarker;

    public void Configure(AutonomousTestVehicleController vehicle, Vector3 newDirection, float distance, float metersPerSecond)
    {
        testVehicle = vehicle;
        direction = newDirection.normalized;
        remainingDistance = Mathf.Max(0f, distance);
        speed = Mathf.Max(0f, metersPerSecond);
        actorMarker = GetComponent<ScenarioActorMarker>();
        actorMarker?.BeginCrossing();
    }

    private void FixedUpdate()
    {
        if (remainingDistance <= 0f || testVehicle == null || !testVehicle.IsRunning)
            return;

        float movement = Mathf.Min(speed * Time.fixedDeltaTime, remainingDistance);
        Vector3 nextPosition = transform.position + direction * movement;
        if (actorMarker != null)
            actorMarker.MoveTo(nextPosition);
        else
            transform.position = nextPosition;
        remainingDistance -= movement;
        if (remainingDistance <= 0.001f)
            actorMarker?.CompleteCrossing();
    }
}

public class ScenarioForwardMover : MonoBehaviour
{
    private AutonomousTestVehicleController testVehicle;
    private float speedMetersPerSecond = 2.5f;
    private ScenarioActorMarker actorMarker;

    public void Configure(AutonomousTestVehicleController vehicle, float speed)
    {
        testVehicle = vehicle;
        speedMetersPerSecond = Mathf.Max(0f, speed);
        actorMarker = GetComponent<ScenarioActorMarker>();
    }

    private void FixedUpdate()
    {
        if (testVehicle != null && testVehicle.IsRunning)
        {
            Vector3 nextPosition = transform.position
                + transform.forward * speedMetersPerSecond * Time.fixedDeltaTime;
            if (actorMarker != null)
                actorMarker.MoveTo(nextPosition);
            else
                transform.position = nextPosition;
        }
    }
}

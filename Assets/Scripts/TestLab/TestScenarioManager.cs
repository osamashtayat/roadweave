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
            36f,
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
            ScenarioForwardMover mover = actor.GetComponent<ScenarioForwardMover>();
            if (mover == null)
                mover = actor.AddComponent<ScenarioForwardMover>();
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

        testVehicle.PrepareForNewScenario();

        Transform vehicleTransform = testVehicle.transform;
        Vector3 vehicleForward = testVehicle.RouteForward;
        Vector3 vehicleRight = testVehicle.RouteRight;
        Vector3 crossingDirection = -vehicleRight;
        CreateCrosswalk(vehicleTransform, vehicleForward, vehicleRight, distanceAhead);
        Vector3 position = vehicleTransform.position
            + vehicleForward * distanceAhead
            + vehicleRight * (roadWidth * 0.5f + sidewalkWidth * 0.5f - egoRightLaneOffset);

        bool usingFallback = pedestrianPrefab == null;
        GameObject pedestrian = usingFallback
            ? CreateFallbackPedestrian()
            : Instantiate(pedestrianPrefab);

        pedestrian.name = "TestPedestrian";
        pedestrian.transform.SetParent(scenarioActorsRoot, true);
        pedestrian.transform.position = position + Vector3.up *
            (usingFallback ? sidewalkHeight : 0.9f + sidewalkHeight);

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

        testVehicle.PrepareForNewScenario();

        Transform vehicleTransform = testVehicle.transform;
        bool usingFallback = prefab == null;
        GameObject actor = usingFallback
            ? CreateFallbackVehicle(actorType, placeholderSize, placeholderColor)
            : Instantiate(prefab);

        actor.name = objectName;
        actor.transform.SetParent(scenarioActorsRoot, true);
        Vector3 routeForward = testVehicle.RouteForward;
        Vector3 roadPosition = vehicleTransform.position
            + routeForward * distanceAhead;
        actor.transform.position = roadPosition;
        actor.transform.rotation = Quaternion.LookRotation(routeForward, Vector3.up);

        ScenarioActorMarker marker = actor.GetComponent<ScenarioActorMarker>();
        if (marker == null)
            marker = actor.AddComponent<ScenarioActorMarker>();
        marker.Configure(actorType, true);
        GroundActor(actor, roadPosition.y);

        spawnedActors.Add(actor);
        CurrentScenarioName = spawnedActors.Count > 1 ? "Mixed Scenario" : objectName;
        return actor;
    }

    private static void GroundActor(GameObject actor, float roadHeight)
    {
        if (actor == null)
            return;

        Renderer[] renderers = actor.GetComponentsInChildren<Renderer>(false);
        bool foundBounds = false;
        Bounds combined = default;
        foreach (Renderer actorRenderer in renderers)
        {
            if (actorRenderer == null || !actorRenderer.enabled)
                continue;
            if (!foundBounds)
            {
                combined = actorRenderer.bounds;
                foundBounds = true;
            }
            else
            {
                combined.Encapsulate(actorRenderer.bounds);
            }
        }

        if (!foundBounds)
        {
            Collider[] colliders = actor.GetComponentsInChildren<Collider>(false);
            foreach (Collider actorCollider in colliders)
            {
                if (actorCollider == null || !actorCollider.enabled)
                    continue;
                if (!foundBounds)
                {
                    combined = actorCollider.bounds;
                    foundBounds = true;
                }
                else
                {
                    combined.Encapsulate(actorCollider.bounds);
                }
            }
        }

        if (!foundBounds)
            return;

        const float tireClearance = 0.015f;
        actor.transform.position += Vector3.up *
            (roadHeight + tireClearance - combined.min.y);
        Physics.SyncTransforms();
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
        Renderer[] renderers = target.GetComponentsInChildren<Renderer>();
        foreach (Renderer renderer in renderers)
            renderer.material.color = color;
    }

    private static GameObject CreateFallbackPedestrian()
    {
        return FallbackActorVisualFactory.CreatePedestrian();
    }

    private static GameObject CreateFallbackVehicle(
        ScenarioActorType actorType,
        Vector3 size,
        Color bodyColor)
    {
        GameObject root = new GameObject(actorType == ScenarioActorType.Truck
            ? "FallbackTruck"
            : "FallbackCar");
        if (actorType == ScenarioActorType.Truck)
        {
            CreateVisualPart(root.transform, "Chassis", PrimitiveType.Cube,
                new Vector3(0f, 0.45f, 0f), new Vector3(size.x, 0.55f, size.z), Quaternion.identity,
                new Color(0.12f, 0.12f, 0.14f));
            CreateVisualPart(root.transform, "Cargo", PrimitiveType.Cube,
                new Vector3(0f, 1.75f, -1f), new Vector3(size.x * 0.95f, 2.45f, size.z * 0.62f),
                Quaternion.identity, bodyColor);
            CreateVisualPart(root.transform, "Cab", PrimitiveType.Cube,
                new Vector3(0f, 1.3f, size.z * 0.34f), new Vector3(size.x * 0.92f, 1.7f, size.z * 0.25f),
                Quaternion.identity, new Color(0.95f, 0.4f, 0.08f));
            AddFallbackWheels(root.transform, size.x, size.z, new[] { -0.36f, 0.08f, 0.37f });
        }
        else
        {
            CreateVisualPart(root.transform, "Body", PrimitiveType.Cube,
                new Vector3(0f, 0.58f, 0f), new Vector3(size.x, 0.72f, size.z), Quaternion.identity, bodyColor);
            CreateVisualPart(root.transform, "Cabin", PrimitiveType.Cube,
                new Vector3(0f, 1.12f, -0.18f), new Vector3(size.x * 0.78f, 0.72f, size.z * 0.5f),
                Quaternion.identity, new Color(0.2f, 0.32f, 0.48f));
            AddFallbackWheels(root.transform, size.x, size.z, new[] { -0.33f, 0.33f });
        }
        return root;
    }

    private static void AddFallbackWheels(Transform root, float width, float length, float[] axleOffsets)
    {
        foreach (float axleOffset in axleOffsets)
        {
            for (int side = -1; side <= 1; side += 2)
            {
                CreateVisualPart(root, side < 0 ? "LeftWheel" : "RightWheel", PrimitiveType.Cylinder,
                    new Vector3(side * width * 0.52f, 0.38f, axleOffset * length),
                    new Vector3(0.34f, 0.16f, 0.34f), Quaternion.Euler(0f, 0f, 90f),
                    new Color(0.035f, 0.035f, 0.04f));
            }
        }
    }

    private static GameObject CreateVisualPart(
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
        Collider primitiveCollider = part.GetComponent<Collider>();
        if (primitiveCollider != null)
        {
            if (Application.isPlaying)
                Destroy(primitiveCollider);
            else
                DestroyImmediate(primitiveCollider);
        }
        SetColor(part, color);
        return part;
    }
}

public enum ScenarioActorType
{
    Pedestrian,
    StoppedCar,
    SlowCar,
    Truck
}

public enum ScenarioActorClass
{
    Unknown,
    Pedestrian,
    Car,
    Truck,
    Bus
}

public enum ScenarioActorMotionState
{
    Unknown,
    Stopped,
    Slow,
    Moving
}

public class ScenarioActorMarker : MonoBehaviour
{
    private static readonly List<ScenarioActorMarker> activeActors = new List<ScenarioActorMarker>();

    public static IReadOnlyList<ScenarioActorMarker> ActiveActors => activeActors;
    // ActorType remains for ReplayActorManager and existing scenes. New driving
    // decisions use SemanticClass and MotionState separately.
    public ScenarioActorType ActorType { get; private set; }
    public ScenarioActorClass SemanticClass { get; private set; } = ScenarioActorClass.Unknown;
    public ScenarioActorMotionState MotionState { get; private set; } = ScenarioActorMotionState.Unknown;
    public Vector3 Velocity { get; private set; }
    public float SpeedMetersPerSecond => Velocity.magnitude;
    public bool CanOvertake { get; private set; }
    public bool HasCrossingMotion { get; private set; }
    public bool CrossingComplete { get; private set; }

    private Rigidbody physicsBody;
    private Vector3 lastSamplePosition;
    private bool hasPositionSample;
    private bool usesCommandedVelocity;
    private ScenarioActorMotionState configuredMotionFallback = ScenarioActorMotionState.Unknown;

    private void OnEnable()
    {
        if (!activeActors.Contains(this))
            activeActors.Add(this);
        lastSamplePosition = transform.position;
        hasPositionSample = false;
        usesCommandedVelocity = false;
    }

    private void OnDisable()
    {
        activeActors.Remove(this);
    }

    public void Configure(ScenarioActorType actorType, bool canOvertake)
    {
        ActorType = actorType;
        SemanticClass = LegacySemanticClass(actorType);
        configuredMotionFallback = LegacyMotionState(actorType);
        MotionState = configuredMotionFallback;
        Velocity = Vector3.zero;
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
        EnsurePracticalRootCollider(SemanticClass);
        lastSamplePosition = transform.position;
        hasPositionSample = false;
        usesCommandedVelocity = false;
    }

    public void Configure(
        ScenarioActorClass semanticClass,
        ScenarioActorMotionState initialMotionState,
        bool canOvertake)
    {
        SemanticClass = semanticClass;
        configuredMotionFallback = initialMotionState;
        MotionState = initialMotionState;
        Velocity = Vector3.zero;
        CanOvertake = canOvertake;
        ActorType = LegacyActorType(semanticClass, initialMotionState);
        ConfigurePhysicsAndCollider();
    }

    public void BeginCrossing()
    {
        HasCrossingMotion = true;
        CrossingComplete = false;
    }

    public void CompleteCrossing()
    {
        if (HasCrossingMotion)
        {
            CrossingComplete = true;
            Velocity = Vector3.zero;
            foreach (Collider actorCollider in GetComponentsInChildren<Collider>(false))
            {
                if (actorCollider != null)
                    actorCollider.enabled = false;
            }
        }
    }

    public void MoveTo(Vector3 position)
    {
        if (physicsBody != null)
            physicsBody.MovePosition(position);
        else
            transform.position = position;
    }

    public void MoveTo(Vector3 position, Vector3 commandedVelocity)
    {
        // Physics movement happens in FixedUpdate, so calculating velocity in
        // LateUpdate produced alternating zero/high readings. Store the known
        // commanded velocity directly for stable TTC and passing decisions.
        Velocity = commandedVelocity;
        MotionState = ClassifyMotion(commandedVelocity.magnitude);
        usesCommandedVelocity = true;
        lastSamplePosition = position;
        hasPositionSample = true;
        if (physicsBody != null)
            physicsBody.MovePosition(position);
        else
            transform.position = position;
    }

    private void LateUpdate()
    {
        Vector3 currentPosition = transform.position;
        if (usesCommandedVelocity)
        {
            lastSamplePosition = currentPosition;
            hasPositionSample = true;
            return;
        }
        if (!hasPositionSample)
        {
            lastSamplePosition = currentPosition;
            hasPositionSample = true;
            Velocity = Vector3.zero;
            MotionState = configuredMotionFallback;
            return;
        }

        float deltaTime = Mathf.Max(Time.deltaTime, 0.0001f);
        Vector3 measuredVelocity = (currentPosition - lastSamplePosition) / deltaTime;
        Velocity = Vector3.Lerp(Velocity, measuredVelocity, 0.45f);
        lastSamplePosition = currentPosition;
        MotionState = ClassifyMotion(Velocity.magnitude);
    }

    private void ConfigurePhysicsAndCollider()
    {
        HasCrossingMotion = false;
        CrossingComplete = false;
        physicsBody = GetComponent<Rigidbody>();
        if (physicsBody == null)
            physicsBody = gameObject.AddComponent<Rigidbody>();
        physicsBody.isKinematic = true;
        physicsBody.useGravity = false;
        physicsBody.interpolation = RigidbodyInterpolation.Interpolate;
        physicsBody.collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;
        EnsurePracticalRootCollider(SemanticClass);
        lastSamplePosition = transform.position;
        hasPositionSample = false;
        usesCommandedVelocity = false;
    }

    private void EnsurePracticalRootCollider(ScenarioActorClass semanticClass)
    {
        Collider rootCollider = GetComponent<Collider>();
        if (rootCollider != null)
            return;

        Bounds worldBounds;
        bool hasRendererBounds = TryGetRendererBounds(out worldBounds);
        Vector3 fallbackSize = semanticClass == ScenarioActorClass.Pedestrian
            ? new Vector3(0.7f, 1.8f, 0.7f)
            : semanticClass == ScenarioActorClass.Truck || semanticClass == ScenarioActorClass.Bus
                ? new Vector3(2.5f, 3f, 8f)
                : new Vector3(1.8f, 1.5f, 4.2f);

        BoxCollider box = gameObject.AddComponent<BoxCollider>();
        if (!hasRendererBounds)
        {
            box.center = Vector3.up * fallbackSize.y * 0.5f;
            box.size = fallbackSize;
            return;
        }

        Vector3 scale = transform.lossyScale;
        box.center = transform.InverseTransformPoint(worldBounds.center);
        box.size = new Vector3(
            worldBounds.size.x / Mathf.Max(0.001f, Mathf.Abs(scale.x)),
            worldBounds.size.y / Mathf.Max(0.001f, Mathf.Abs(scale.y)),
            worldBounds.size.z / Mathf.Max(0.001f, Mathf.Abs(scale.z))
        );
    }

    private bool TryGetRendererBounds(out Bounds combined)
    {
        combined = default;
        Renderer[] renderers = GetComponentsInChildren<Renderer>(false);
        bool found = false;
        foreach (Renderer actorRenderer in renderers)
        {
            if (actorRenderer == null || !actorRenderer.enabled)
                continue;
            if (!found)
            {
                combined = actorRenderer.bounds;
                found = true;
            }
            else
            {
                combined.Encapsulate(actorRenderer.bounds);
            }
        }
        return found;
    }

    private static ScenarioActorClass LegacySemanticClass(ScenarioActorType type)
    {
        switch (type)
        {
            case ScenarioActorType.Pedestrian: return ScenarioActorClass.Pedestrian;
            case ScenarioActorType.Truck: return ScenarioActorClass.Truck;
            default: return ScenarioActorClass.Car;
        }
    }

    private static ScenarioActorMotionState LegacyMotionState(ScenarioActorType type)
    {
        switch (type)
        {
            case ScenarioActorType.StoppedCar: return ScenarioActorMotionState.Stopped;
            case ScenarioActorType.SlowCar:
            case ScenarioActorType.Truck: return ScenarioActorMotionState.Slow;
            case ScenarioActorType.Pedestrian: return ScenarioActorMotionState.Moving;
            default: return ScenarioActorMotionState.Unknown;
        }
    }

    private static ScenarioActorType LegacyActorType(
        ScenarioActorClass semanticClass,
        ScenarioActorMotionState motionState)
    {
        if (semanticClass == ScenarioActorClass.Pedestrian)
            return ScenarioActorType.Pedestrian;
        if (semanticClass == ScenarioActorClass.Truck || semanticClass == ScenarioActorClass.Bus)
            return ScenarioActorType.Truck;
        return motionState == ScenarioActorMotionState.Stopped
            ? ScenarioActorType.StoppedCar
            : ScenarioActorType.SlowCar;
    }

    private static ScenarioActorMotionState ClassifyMotion(float speedMetersPerSecond)
    {
        if (speedMetersPerSecond < 0.25f)
            return ScenarioActorMotionState.Stopped;
        if (speedMetersPerSecond < 7f)
            return ScenarioActorMotionState.Slow;
        return ScenarioActorMotionState.Moving;
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
            actorMarker.MoveTo(nextPosition, direction * speed);
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
                actorMarker.MoveTo(nextPosition, transform.forward * speedMetersPerSecond);
            else
                transform.position = nextPosition;
        }
    }
}

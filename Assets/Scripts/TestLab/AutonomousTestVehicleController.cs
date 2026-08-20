using System;
using System.Collections.Generic;
using UnityEngine;

public class AutonomousTestVehicleController : MonoBehaviour
{
    [Header("Route Following")]
    [SerializeField, Min(1f)] private float maximumSpeedKph = 30f;
    [SerializeField, Min(0.1f)] private float accelerationMetersPerSecondSquared = 2.5f;
    [SerializeField, Min(0.1f)] private float brakingMetersPerSecondSquared = 6f;
    [SerializeField, Min(1f)] private float maximumTurnDegreesPerSecond = 80f;
    [SerializeField, Min(0.2f)] private float waypointReachedDistance = 1.5f;
    [SerializeField, Min(0.2f)] private float destinationReachedDistance = 1.5f;
    [SerializeField, Min(2f)] private float steeringLookAheadDistance = 7f;
    [Tooltip("Use 180 if the imported model's visible front points toward local -Z.")]
    [SerializeField] private float modelForwardYawOffset;

    [Header("Obstacle Detection")]
    [SerializeField] private LayerMask obstacleLayers = ~0;
    [SerializeField, Min(1f)] private float baseDetectionRange = 35f;
    [SerializeField, Min(0.1f)] private float detectionRadius = 0.9f;
    [SerializeField, Min(0.1f)] private float stoppedDistance = 4.5f;
    [SerializeField] private Vector3 detectorOffset = new Vector3(0f, 0.8f, 1.4f);
    [Tooltip("Pedestrians inside this corridor are treated as crossing risks before they reach the lane center.")]
    [SerializeField, Min(1f)] private float pedestrianPredictionHalfWidth = 6.5f;

    [Header("Vehicle Overtaking")]
    [SerializeField] private bool enableOvertaking = true;
    [SerializeField, Min(6f)] private float overtakeStartDistance = 18f;
    [SerializeField, Min(1.5f)] private float overtakeLaneOffset = 5.5f;
    [SerializeField, Min(0.5f)] private float laneOffsetChangePerSecond = 1.8f;
    [SerializeField, Min(2f)] private float distancePastActorBeforeReturn = 6f;
    [SerializeField, Min(5f)] private float laneChangeSpeedKph = 15f;
    [SerializeField, Min(5f)] private float passingSpeedKph = 30f;

    public event Action<float> ObstacleDetected;
    public event Action BrakingStarted;
    public event Action DestinationReached;
    public event Action CollisionOccurred;

    public float CurrentSpeedKph => currentSpeedMps * 3.6f;
    public Vector3 MovementForward => GetMovementForward(transform.rotation);
    public Vector3 MovementRight => Quaternion.Euler(0f, 90f, 0f) * MovementForward;
    public float NearestObstacleDistance { get; private set; } = float.PositiveInfinity;
    public bool IsRunning { get; private set; }
    public bool HasReachedDestination { get; private set; }
    public string SensorSummary { get; private set; } = "No sensor scan yet";
    public string CurrentManeuver
    {
        get
        {
            switch (overtakePhase)
            {
                case OvertakePhase.MovingOut: return "Changing to left lane";
                case OvertakePhase.Passing: return "Passing in left lane";
                case OvertakePhase.CruisingLeft: return "Cruising in left lane";
                case OvertakePhase.Returning: return "Returning to right lane";
                default: return CurrentSpeedKph < 0.2f && IsRunning ? "Waiting / stopped" : "Following right lane";
            }
        }
    }

    private enum OvertakePhase
    {
        None,
        MovingOut,
        Passing,
        CruisingLeft,
        Returning
    }

    private readonly List<Vector3> route = new List<Vector3>();
    private Rigidbody physicsBody;
    private SimulatedVehicleSensorSuite sensorSuite;
    private int waypointIndex;
    private float currentSpeedMps;
    private float speedFactor = 1f;
    private float brakingFactor = 1f;
    private float detectionFactor = 1f;
    private bool obstacleWasDetected;
    private bool wasBraking;
    private bool collisionReported;
    private Collider nearestObstacleCollider;
    private ScenarioActorMarker nearestScenarioActor;
    private ScenarioActorMarker overtakingActor;
    private OvertakePhase overtakePhase;
    private bool remainLeftAfterCurrentPass;
    private bool returnWasRequestedFromLeftCruise;
    private float currentLaneOffset;
    private float targetLaneOffset;
    private float physicalLaneOffset;

    private void Awake()
    {
        EnsurePhysicsBody();
        sensorSuite = GetComponent<SimulatedVehicleSensorSuite>();
        if (sensorSuite == null)
            sensorSuite = gameObject.AddComponent<SimulatedVehicleSensorSuite>();
    }

    private void FixedUpdate()
    {
        if (!IsRunning || route.Count == 0 || HasReachedDestination)
            return;

        float deltaTime = Time.fixedDeltaTime;
        AdvanceReachedWaypoints();

        Vector3 routeTarget = route[Mathf.Clamp(waypointIndex, 0, route.Count - 1)];
        Vector3 routeDirection = routeTarget - physicsBody.position;
        routeDirection.y = 0f;
        Vector3 routeForward = GetRouteForwardDirection();
        bool passedFinalRoutePoint = routeForward.sqrMagnitude > 0.001f &&
            Vector3.Dot(routeDirection, routeForward) <= 0f;

        if (waypointIndex >= route.Count - 1 &&
            (routeDirection.magnitude <= destinationReachedDistance || passedFinalRoutePoint))
        {
            currentSpeedMps = Mathf.MoveTowards(
                currentSpeedMps,
                0f,
                brakingMetersPerSecondSquared * brakingFactor * deltaTime
            );

            if (currentSpeedMps <= 0.05f)
                FinishRoute();
            return;
        }

        Vector3 steeringBaseTarget = GetSteeringLookAheadTarget();
        Vector3 steeringBaseDirection = steeringBaseTarget - physicsBody.position;
        steeringBaseDirection.y = 0f;
        // The road direction must not point back toward the center line when
        // the vehicle is already in the left lane.
        if (routeForward.sqrMagnitude <= 0.001f)
        {
            routeForward = steeringBaseDirection.sqrMagnitude > 0.001f
                ? steeringBaseDirection.normalized
                : MovementForward;
        }

        Vector3 routeRight = Quaternion.Euler(0f, 90f, 0f) * routeForward;
        physicalLaneOffset = CalculatePhysicalLaneOffset(routeRight);
        SimulatedSensorSnapshot sensorSnapshot = sensorSuite.Scan(
            physicsBody.position,
            routeForward,
            routeRight,
            physicalLaneOffset,
            transform
        );
        UpdateSensorSummary(sensorSnapshot);
        UpdateOvertakeState(routeForward, deltaTime, sensorSnapshot, physicalLaneOffset);
        Vector3 steeringTarget = steeringBaseTarget + routeRight * currentLaneOffset;
        Vector3 steeringDirection = steeringTarget - physicsBody.position;
        steeringDirection.y = 0f;

        Quaternion nextRotation = physicsBody.rotation;
        if (steeringDirection.sqrMagnitude > 0.001f)
        {
            Quaternion desiredRotation = Quaternion.LookRotation(steeringDirection.normalized, Vector3.up)
                * Quaternion.Euler(0f, -modelForwardYawOffset, 0f);
            nextRotation = Quaternion.RotateTowards(
                physicsBody.rotation,
                desiredRotation,
                maximumTurnDegreesPerSecond * deltaTime
            );
        }

        Vector3 movementForward = GetMovementForward(nextRotation);
        float desiredSpeedMps = maximumSpeedKph * speedFactor / 3.6f;
        float turnAngle = steeringDirection.sqrMagnitude > 0.001f
            ? Vector3.Angle(movementForward, steeringDirection.normalized)
            : 0f;
        if (turnAngle > 35f)
            desiredSpeedMps *= 0.55f;

        ApplySensorDecision(sensorSnapshot, ref desiredSpeedMps);

        bool braking = desiredSpeedMps < currentSpeedMps - 0.05f;
        if (braking && !wasBraking)
            BrakingStarted?.Invoke();
        wasBraking = braking;

        float rate = braking
            ? brakingMetersPerSecondSquared * brakingFactor
            : accelerationMetersPerSecondSquared;
        currentSpeedMps = Mathf.MoveTowards(currentSpeedMps, desiredSpeedMps, rate * deltaTime);

        float movementDistance = currentSpeedMps * deltaTime;
        if (movementDistance > 0f && SweepForCollision(movementForward, movementDistance, out Collider collision))
        {
            ReportCollision(collision);
            return;
        }

        physicsBody.MoveRotation(nextRotation);
        physicsBody.MovePosition(physicsBody.position + movementForward * movementDistance);
    }

    public void SetRoute(IReadOnlyList<Vector3> worldRoute, float initialSpeedKph)
    {
        route.Clear();
        if (worldRoute != null)
        {
            for (int index = 0; index < worldRoute.Count; index++)
                route.Add(worldRoute[index]);
        }

        waypointIndex = FindClosestForwardWaypoint();
        currentSpeedMps = Mathf.Max(0f, initialSpeedKph / 3.6f);
        HasReachedDestination = false;
        collisionReported = false;
        obstacleWasDetected = false;
        wasBraking = false;
        NearestObstacleDistance = float.PositiveInfinity;
        ResetOvertake();
        AlignWithRoute();
    }

    public void SetRunning(bool shouldRun)
    {
        IsRunning = shouldRun && route.Count > 1 && !HasReachedDestination;
    }

    public void SetEnvironmentModifiers(float newSpeedFactor, float newBrakingFactor, float newDetectionFactor)
    {
        speedFactor = Mathf.Clamp(newSpeedFactor, 0.1f, 1.5f);
        brakingFactor = Mathf.Clamp(newBrakingFactor, 0.1f, 1.5f);
        detectionFactor = Mathf.Clamp(newDetectionFactor, 0.1f, 1.5f);
    }

    public void ConfigureMaximumSpeed(float speedKph)
    {
        maximumSpeedKph = Mathf.Max(1f, speedKph);
    }

    public void ConfigureModelForwardYawOffset(float yawOffset)
    {
        modelForwardYawOffset = yawOffset;
    }

    private void AdvanceReachedWaypoints()
    {
        float allowedDistance = waypointReachedDistance + Mathf.Abs(currentLaneOffset);
        while (waypointIndex < route.Count - 1)
        {
            Vector3 difference = route[waypointIndex] - physicsBody.position;
            difference.y = 0f;
            Vector3 waypointForward = GetRouteDirectionAtIndex(waypointIndex);
            float longitudinalDistance = Vector3.Dot(difference, waypointForward);
            bool isCloseToWaypoint = difference.magnitude <= allowedDistance;
            bool hasReachedOrPassedWaypoint = longitudinalDistance <= waypointReachedDistance;

            // A passed waypoint must never remain selected behind the vehicle;
            // otherwise the steering controller turns around and drives circles.
            if (!isCloseToWaypoint && !hasReachedOrPassedWaypoint)
                break;
            waypointIndex++;
        }
    }

    private int FindClosestForwardWaypoint()
    {
        if (route.Count == 0)
            return 0;

        int closestIndex = 0;
        float closestDistance = float.PositiveInfinity;
        for (int index = 0; index < route.Count; index++)
        {
            float distance = (route[index] - transform.position).sqrMagnitude;
            if (distance < closestDistance)
            {
                closestDistance = distance;
                closestIndex = index;
            }
        }
        return Mathf.Min(closestIndex + 1, route.Count - 1);
    }

    private void AlignWithRoute()
    {
        if (route.Count < 2)
            return;

        Vector3 direction = GetSteeringLookAheadTarget() - physicsBody.position;
        direction.y = 0f;
        if (direction.sqrMagnitude <= 0.001f)
            return;

        Quaternion alignedRotation = Quaternion.LookRotation(direction.normalized, Vector3.up)
            * Quaternion.Euler(0f, -modelForwardYawOffset, 0f);
        physicsBody.rotation = alignedRotation;
        transform.rotation = alignedRotation;
    }

    private Vector3 GetSteeringLookAheadTarget()
    {
        if (route.Count == 0)
            return physicsBody.position;

        float requiredLookAhead = steeringLookAheadDistance;
        if (overtakePhase != OvertakePhase.None)
        {
            // Look farther ahead during an overtake so the car changes lane in
            // one smooth arc, then points straight along the left lane.
            requiredLookAhead = Mathf.Max(
                requiredLookAhead,
                Mathf.Abs(targetLaneOffset) * 2.75f
            );
        }

        int selectedIndex = Mathf.Clamp(waypointIndex, 0, route.Count - 1);
        for (int index = selectedIndex; index < route.Count; index++)
        {
            Vector3 difference = route[index] - physicsBody.position;
            difference.y = 0f;
            selectedIndex = index;
            if (difference.magnitude >= requiredLookAhead)
                break;
        }
        return route[selectedIndex];
    }

    private Vector3 GetRouteForwardDirection()
    {
        if (route.Count < 2)
            return MovementForward;

        int centerIndex = Mathf.Clamp(waypointIndex, 0, route.Count - 1);
        int startIndex = Mathf.Max(0, centerIndex - 2);
        int endIndex = Mathf.Min(route.Count - 1, centerIndex + 2);
        if (startIndex == endIndex)
            startIndex = Mathf.Max(0, endIndex - 1);

        Vector3 direction = route[endIndex] - route[startIndex];
        direction.y = 0f;
        return direction.sqrMagnitude > 0.001f
            ? direction.normalized
            : MovementForward;
    }

    private Vector3 GetRouteDirectionAtIndex(int index)
    {
        if (route.Count < 2)
            return MovementForward;

        int safeIndex = Mathf.Clamp(index, 0, route.Count - 1);
        int previousIndex = Mathf.Max(0, safeIndex - 1);
        int nextIndex = Mathf.Min(route.Count - 1, safeIndex + 1);
        if (previousIndex == nextIndex)
            return MovementForward;

        Vector3 direction = route[nextIndex] - route[previousIndex];
        direction.y = 0f;
        return direction.sqrMagnitude > 0.001f
            ? direction.normalized
            : MovementForward;
    }

    private float CalculatePhysicalLaneOffset(Vector3 routeRight)
    {
        if (route.Count == 0)
            return 0f;

        int startIndex = Mathf.Max(0, waypointIndex - 8);
        int endIndex = Mathf.Min(route.Count - 1, waypointIndex + 8);
        Vector3 closestPoint = route[Mathf.Clamp(waypointIndex, 0, route.Count - 1)];
        float closestDistance = float.PositiveInfinity;
        for (int index = startIndex; index <= endIndex; index++)
        {
            Vector3 difference = route[index] - physicsBody.position;
            difference.y = 0f;
            float distance = difference.sqrMagnitude;
            if (distance < closestDistance)
            {
                closestDistance = distance;
                closestPoint = route[index];
            }
        }

        Vector3 offset = physicsBody.position - closestPoint;
        offset.y = 0f;
        return Vector3.Dot(offset, routeRight);
    }

    private void DetectObstacle(Vector3 movementForward)
    {
        float range = baseDetectionRange * detectionFactor;
        Vector3 origin = GetDetectorOrigin(movementForward);
        RaycastHit[] hits = Physics.SphereCastAll(
            origin,
            detectionRadius,
            movementForward,
            range,
            obstacleLayers,
            QueryTriggerInteraction.Ignore
        );

        float closest = float.PositiveInfinity;
        nearestObstacleCollider = null;
        nearestScenarioActor = null;
        foreach (RaycastHit hit in hits)
        {
            if (ShouldIgnoreCollider(hit.collider) || hit.distance >= closest)
                continue;

            closest = hit.distance;
            nearestObstacleCollider = hit.collider;
            nearestScenarioActor = hit.collider.GetComponentInParent<ScenarioActorMarker>();
        }

        IncludePredictedPedestrians(movementForward, range, ref closest);

        NearestObstacleDistance = closest;
        bool detected = !float.IsInfinity(closest);
        if (detected && !obstacleWasDetected)
            ObstacleDetected?.Invoke(closest);
        obstacleWasDetected = detected;
    }

    private void TryStartOvertake(Vector3 routeForward)
    {
        if (!enableOvertaking || overtakePhase != OvertakePhase.None)
            return;
        if (nearestScenarioActor == null || !nearestScenarioActor.CanOvertake)
            return;
        if (NearestObstacleDistance > overtakeStartDistance)
            return;
        Vector3 actorDifference = nearestScenarioActor.transform.position - physicsBody.position;
        float actorLateralPosition = Vector3.Dot(actorDifference, MovementRight);
        // An actor already occupying the left lane must never be treated as an
        // object that can be passed by moving farther left.
        if (actorLateralPosition < -1.25f)
            return;
        if (!IsOvertakeLaneClear(routeForward, nearestScenarioActor))
            return;

        overtakingActor = nearestScenarioActor;
        remainLeftAfterCurrentPass = ShouldRemainLeftAfterPassing(nearestScenarioActor);
        returnWasRequestedFromLeftCruise = false;
        targetLaneOffset = -overtakeLaneOffset;
        overtakePhase = OvertakePhase.MovingOut;
    }

    private void UpdateOvertakeState(
        Vector3 routeForward,
        float deltaTime,
        SimulatedSensorSnapshot sensors,
        float actualLaneOffset)
    {
        if (overtakePhase == OvertakePhase.None)
            return;

        if (overtakingActor == null && overtakePhase != OvertakePhase.CruisingLeft)
        {
            if (remainLeftAfterCurrentPass && overtakePhase == OvertakePhase.Passing)
            {
                targetLaneOffset = -overtakeLaneOffset;
                overtakePhase = OvertakePhase.CruisingLeft;
            }
            else
            {
                targetLaneOffset = 0f;
                overtakePhase = OvertakePhase.Returning;
            }
        }

        currentLaneOffset = Mathf.MoveTowards(
            currentLaneOffset,
            targetLaneOffset,
            laneOffsetChangePerSecond * deltaTime
        );

        if (overtakePhase == OvertakePhase.MovingOut &&
            Mathf.Abs(currentLaneOffset - targetLaneOffset) <= 0.05f)
        {
            overtakePhase = OvertakePhase.Passing;
        }

        if (overtakePhase == OvertakePhase.Passing && overtakingActor != null)
        {
            Vector3 actorDifference = overtakingActor.transform.position - physicsBody.position;
            actorDifference.y = 0f;
            float actorAheadDistance = Vector3.Dot(actorDifference, routeForward);
            bool targetPassed = actorAheadDistance <= -distancePastActorBeforeReturn;
            if (targetPassed && remainLeftAfterCurrentPass)
            {
                targetLaneOffset = -overtakeLaneOffset;
                overtakePhase = OvertakePhase.CruisingLeft;
            }
            else if (targetPassed && sensors.IsRightLaneClear(15f, 10f))
            {
                returnWasRequestedFromLeftCruise = false;
                targetLaneOffset = 0f;
                overtakePhase = OvertakePhase.Returning;
            }
        }

        if (overtakePhase == OvertakePhase.Returning &&
            !sensors.IsRightLaneClear(12f, 8f) &&
            actualLaneOffset < -2.5f)
        {
            targetLaneOffset = -overtakeLaneOffset;
            overtakePhase = returnWasRequestedFromLeftCruise
                ? OvertakePhase.CruisingLeft
                : OvertakePhase.Passing;
        }

        if (overtakePhase == OvertakePhase.Returning &&
            Mathf.Abs(currentLaneOffset) <= 0.05f &&
            Mathf.Abs(actualLaneOffset) <= 0.6f)
            ResetOvertake();
    }

    private void ApplySensorDecision(
        SimulatedSensorSnapshot sensors,
        ref float desiredSpeedMps)
    {
        SimulatedActorObservation relevantObservation = null;

        if (sensors.pedestrianHazard != null)
        {
            relevantObservation = sensors.pedestrianHazard;
            ApplyFollowingSpeed(relevantObservation.gapDistance, ref desiredSpeedMps, 1.2f);
            UpdateDetectedObstacle(relevantObservation.gapDistance);
            return;
        }

        if (overtakePhase == OvertakePhase.None)
        {
            relevantObservation = sensors.rightLaneFront;
            if (relevantObservation != null)
            {
                bool canOvertakeActor = enableOvertaking
                    && relevantObservation.actor != null
                    && relevantObservation.actor.CanOvertake;
                bool decisionDistanceReached = relevantObservation.gapDistance <= overtakeStartDistance;
                bool leftLaneClear = sensors.IsLeftLaneClear(25f, 12f);

                if (canOvertakeActor && decisionDistanceReached && leftLaneClear)
                {
                    overtakingActor = relevantObservation.actor;
                    remainLeftAfterCurrentPass = ShouldRemainLeftAfterPassing(overtakingActor);
                    returnWasRequestedFromLeftCruise = false;
                    targetLaneOffset = -overtakeLaneOffset;
                    overtakePhase = OvertakePhase.MovingOut;
                    desiredSpeedMps = Mathf.Min(desiredSpeedMps, GetLaneChangeSpeedMps());
                }
                else
                {
                    ApplyFollowingSpeed(relevantObservation.gapDistance, ref desiredSpeedMps, 0.8f);
                }
            }
        }
        else if (overtakePhase == OvertakePhase.MovingOut)
        {
            float laneProgress = Mathf.Abs(physicalLaneOffset) / Mathf.Max(0.1f, overtakeLaneOffset);
            relevantObservation = sensors.rightLaneFront;
            desiredSpeedMps = Mathf.Min(desiredSpeedMps, GetLaneChangeSpeedMps());
            if (relevantObservation != null && relevantObservation.gapDistance < 2.5f && laneProgress < 0.55f)
                desiredSpeedMps = 0f;
        }
        else if (overtakePhase == OvertakePhase.Passing)
        {
            relevantObservation = sensors.leftLaneFront;
            desiredSpeedMps = Mathf.Max(desiredSpeedMps, GetOvertakeCruiseSpeedMps());
            if (relevantObservation != null)
                ApplyFollowingSpeed(relevantObservation.gapDistance, ref desiredSpeedMps, 0.8f);
        }
        else if (overtakePhase == OvertakePhase.CruisingLeft)
        {
            relevantObservation = sensors.leftLaneFront;
            desiredSpeedMps = Mathf.Max(desiredSpeedMps, GetOvertakeCruiseSpeedMps());
            bool stoppedCarAhead = relevantObservation != null
                && relevantObservation.actor != null
                && relevantObservation.actor.ActorType == ScenarioActorType.StoppedCar;
            bool decisionDistanceReached = relevantObservation != null
                && relevantObservation.gapDistance <= overtakeStartDistance;

            if (stoppedCarAhead &&
                decisionDistanceReached &&
                sensors.IsRightLaneClear(25f, 12f))
            {
                overtakingActor = relevantObservation.actor;
                returnWasRequestedFromLeftCruise = true;
                targetLaneOffset = 0f;
                overtakePhase = OvertakePhase.Returning;
                desiredSpeedMps = Mathf.Min(desiredSpeedMps, GetLaneChangeSpeedMps());
            }
            else if (relevantObservation != null)
            {
                ApplyFollowingSpeed(relevantObservation.gapDistance, ref desiredSpeedMps, 0.8f);
            }
        }
        else if (overtakePhase == OvertakePhase.Returning)
        {
            relevantObservation = sensors.rightLaneFront;
            if (relevantObservation != null && relevantObservation.actor != overtakingActor)
                ApplyFollowingSpeed(relevantObservation.gapDistance, ref desiredSpeedMps, 0.8f);
        }

        UpdateDetectedObstacle(
            relevantObservation != null
                ? relevantObservation.gapDistance
                : float.PositiveInfinity
        );
    }

    private void ApplyFollowingSpeed(
        float obstacleGap,
        ref float desiredSpeedMps,
        float extraTimeGap)
    {
        float safeGap = stoppedDistance + currentSpeedMps * extraTimeGap;
        if (obstacleGap <= safeGap)
        {
            desiredSpeedMps = 0f;
            return;
        }

        float fullSpeedGap = safeGap + 16f;
        float gapFactor = Mathf.InverseLerp(safeGap, fullSpeedGap, obstacleGap);
        desiredSpeedMps = Mathf.Min(desiredSpeedMps, maximumSpeedKph * speedFactor / 3.6f * gapFactor);
    }

    private float GetLaneChangeSpeedMps()
    {
        // Existing scene objects may still store the old value of 15 km/h.
        // Use at least 20 km/h on a dry road without exceeding the normal limit.
        float normalMaximumKph = maximumSpeedKph * speedFactor;
        float laneChangeTargetKph = Mathf.Max(laneChangeSpeedKph, 20f * speedFactor);
        return Mathf.Min(normalMaximumKph, laneChangeTargetKph) / 3.6f;
    }

    private float GetOvertakeCruiseSpeedMps()
    {
        // Passing speed may intentionally be set above the normal cruise speed
        // in the Inspector. Weather still scales the resulting target speed.
        float overtakeTargetKph = Mathf.Max(30f, maximumSpeedKph, passingSpeedKph);
        return overtakeTargetKph * speedFactor / 3.6f;
    }

    private void UpdateDetectedObstacle(float distance)
    {
        NearestObstacleDistance = distance;
        bool detected = !float.IsInfinity(distance);
        if (detected && !obstacleWasDetected)
            ObstacleDetected?.Invoke(distance);
        obstacleWasDetected = detected;
    }

    private void UpdateSensorSummary(SimulatedSensorSnapshot sensors)
    {
        SensorSummary =
            $"Right front: {DescribeObservation(sensors.rightLaneFront)}\n" +
            $"Right rear: {DescribeObservation(sensors.rightLaneRear)}\n" +
            $"Left front: {DescribeObservation(sensors.leftLaneFront)}\n" +
            $"Left rear: {DescribeObservation(sensors.leftLaneRear)}\n" +
            $"Pedestrian risk: {DescribeObservation(sensors.pedestrianHazard)}";
    }

    private static string DescribeObservation(SimulatedActorObservation observation)
    {
        if (observation == null)
            return "Clear";

        string actorName = observation.actor != null
            ? observation.actor.ActorType.ToString()
            : "Unregistered object";
        return $"{actorName}, {observation.gapDistance:F1} m";
    }

    private void ApplyObstacleSpeed(ref float desiredSpeedMps)
    {
        if (float.IsInfinity(NearestObstacleDistance))
            return;

        bool isCurrentOvertakeActor = overtakingActor != null && nearestScenarioActor == overtakingActor;
        if (isCurrentOvertakeActor && overtakePhase != OvertakePhase.None)
        {
            float laneProgress = Mathf.Abs(currentLaneOffset) / Mathf.Max(0.1f, overtakeLaneOffset);
            if (laneProgress >= 0.72f)
                return;

            desiredSpeedMps = Mathf.Min(desiredSpeedMps, laneChangeSpeedKph / 3.6f);
            if (NearestObstacleDistance <= stoppedDistance)
                desiredSpeedMps = 0f;
            return;
        }

        float safeDistance = stoppedDistance + currentSpeedMps * 0.65f;
        if (NearestObstacleDistance <= safeDistance)
            desiredSpeedMps = 0f;
        else
            desiredSpeedMps *= Mathf.InverseLerp(
                safeDistance,
                baseDetectionRange * detectionFactor,
                NearestObstacleDistance
            );
    }

    private bool IsOvertakeLaneClear(Vector3 routeForward, ScenarioActorMarker actorToPass)
    {
        Vector3 routeRight = Quaternion.Euler(0f, 90f, 0f) * routeForward;

        foreach (ScenarioActorMarker marker in ScenarioActorMarker.ActiveActors)
        {
            if (marker == null || marker == actorToPass || !marker.gameObject.activeInHierarchy)
                continue;

            Vector3 difference = marker.transform.position - physicsBody.position;
            float forwardDistance = Vector3.Dot(difference, routeForward);
            float lateralDistance = Vector3.Dot(difference, routeRight);
            bool alongsidePassingArea = forwardDistance >= -8f && forwardDistance <= overtakeStartDistance + 12f;
            bool insideLeftLane = Mathf.Abs(lateralDistance + overtakeLaneOffset) <= 2.1f;
            if (alongsidePassingArea && insideLeftLane)
                return false;
        }

        Vector3 origin = GetDetectorOrigin(routeForward) - routeRight * overtakeLaneOffset;
        RaycastHit[] hits = Physics.SphereCastAll(
            origin,
            detectionRadius + 0.25f,
            routeForward,
            overtakeStartDistance + 10f,
            obstacleLayers,
            QueryTriggerInteraction.Ignore
        );

        foreach (RaycastHit hit in hits)
        {
            if (ShouldIgnoreCollider(hit.collider))
                continue;
            ScenarioActorMarker marker = hit.collider.GetComponentInParent<ScenarioActorMarker>();
            if (marker == actorToPass)
                continue;
            return false;
        }
        return true;
    }

    private void IncludePredictedPedestrians(Vector3 movementForward, float range, ref float closest)
    {
        Vector3 movementRight = Quaternion.Euler(0f, 90f, 0f) * movementForward;
        foreach (ScenarioActorMarker marker in ScenarioActorMarker.ActiveActors)
        {
            if (marker == null || !marker.gameObject.activeInHierarchy)
                continue;
            if (marker.ActorType != ScenarioActorType.Pedestrian)
                continue;

            Vector3 difference = marker.transform.position - physicsBody.position;
            difference.y = 0f;
            float forwardDistance = Vector3.Dot(difference, movementForward);
            float lateralDistance = Mathf.Abs(Vector3.Dot(difference, movementRight));
            if (forwardDistance <= 0f || forwardDistance > range)
                continue;
            if (lateralDistance > pedestrianPredictionHalfWidth)
                continue;

            float predictedDistance = Mathf.Max(0f, forwardDistance - detectorOffset.z - detectionRadius);
            if (predictedDistance >= closest)
                continue;

            closest = predictedDistance;
            nearestScenarioActor = marker;
            nearestObstacleCollider = marker.GetComponentInChildren<Collider>();
        }
    }

    private bool SweepForCollision(Vector3 movementForward, float movementDistance, out Collider collision)
    {
        Vector3 origin = GetDetectorOrigin(movementForward);
        Collider[] overlaps = Physics.OverlapSphere(
            origin,
            detectionRadius,
            obstacleLayers,
            QueryTriggerInteraction.Ignore
        );
        foreach (Collider overlap in overlaps)
        {
            if (ShouldIgnoreCollider(overlap))
                continue;
            collision = overlap;
            return true;
        }

        RaycastHit[] hits = Physics.SphereCastAll(
            origin,
            detectionRadius,
            movementForward,
            movementDistance + 0.15f,
            obstacleLayers,
            QueryTriggerInteraction.Ignore
        );
        foreach (RaycastHit hit in hits)
        {
            if (ShouldIgnoreCollider(hit.collider))
                continue;
            collision = hit.collider;
            return true;
        }

        collision = null;
        return false;
    }

    private Vector3 GetDetectorOrigin(Vector3 movementForward)
    {
        Vector3 movementRight = Quaternion.Euler(0f, 90f, 0f) * movementForward;
        return physicsBody.position
            + movementRight * detectorOffset.x
            + Vector3.up * detectorOffset.y
            + movementForward * detectorOffset.z;
    }

    private bool ShouldIgnoreCollider(Collider candidate)
    {
        if (candidate == null)
            return true;
        Transform candidateTransform = candidate.transform;
        if (candidateTransform == transform || candidateTransform.IsChildOf(transform))
            return true;
        if (candidate.GetComponentInParent<RoadSurfaceMarker>() != null)
            return true;
        // Test Lab physics reacts only to actors registered by the scenario or
        // replay adapters. Decorative city meshes must not become phantom cars.
        if (candidate.GetComponentInParent<ScenarioActorMarker>() == null)
            return true;
        return false;
    }

    private Vector3 GetMovementForward(Quaternion visualRotation)
    {
        Vector3 localMovementForward = Quaternion.Euler(0f, modelForwardYawOffset, 0f) * Vector3.forward;
        return (visualRotation * localMovementForward).normalized;
    }

    private void ResetOvertake()
    {
        overtakingActor = null;
        overtakePhase = OvertakePhase.None;
        remainLeftAfterCurrentPass = false;
        returnWasRequestedFromLeftCruise = false;
        currentLaneOffset = 0f;
        targetLaneOffset = 0f;
    }

    private static bool ShouldRemainLeftAfterPassing(ScenarioActorMarker actor)
    {
        if (actor == null)
            return false;

        return actor.ActorType == ScenarioActorType.SlowCar ||
               actor.ActorType == ScenarioActorType.Truck;
    }

    private void FinishRoute()
    {
        HasReachedDestination = true;
        IsRunning = false;
        currentSpeedMps = 0f;
        DestinationReached?.Invoke();
    }

    private void EnsurePhysicsBody()
    {
        physicsBody = GetComponent<Rigidbody>();
        if (physicsBody == null)
            physicsBody = gameObject.AddComponent<Rigidbody>();
        physicsBody.isKinematic = true;
        physicsBody.useGravity = false;
        physicsBody.interpolation = RigidbodyInterpolation.Interpolate;
        physicsBody.collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;

        if (GetComponentInChildren<Collider>() == null)
        {
            BoxCollider box = gameObject.AddComponent<BoxCollider>();
            box.center = new Vector3(0f, 0.75f, 0f);
            box.size = new Vector3(1.8f, 1.5f, 4.2f);
        }
    }

    private void ReportCollision(Collider other)
    {
        if (collisionReported || other == null || ShouldIgnoreCollider(other))
            return;

        collisionReported = true;
        IsRunning = false;
        currentSpeedMps = 0f;
        CollisionOccurred?.Invoke();
    }

    private void OnCollisionEnter(Collision collision)
    {
        ReportCollision(collision.collider);
    }

    private void OnTriggerEnter(Collider other)
    {
        ReportCollision(other);
    }
}

public class RoadSurfaceMarker : MonoBehaviour
{
}

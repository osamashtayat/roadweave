using System;
using System.Collections.Generic;
using UnityEngine;

public enum VehicleDecelerationReason
{
    None,
    Obstacle,
    Curve,
    Weather,
    Destination,
    Maneuver
}

public class AutonomousTestVehicleController : MonoBehaviour
{
    [Header("Route Following")]
    [SerializeField, Min(1f)] private float maximumSpeedKph = 30f;
    [SerializeField, Min(0.1f)] private float accelerationMetersPerSecondSquared = 3.8f;
    [SerializeField, Min(0.1f)] private float brakingMetersPerSecondSquared = 6f;
    [SerializeField, Min(1f)] private float maximumTurnDegreesPerSecond = 80f;
    [SerializeField, Min(0.2f)] private float waypointReachedDistance = 1.5f;
    [SerializeField, Min(0.2f)] private float destinationReachedDistance = 1.5f;
    [SerializeField, Min(2f)] private float steeringLookAheadDistance = 7f;
    [SerializeField, Range(5f, 45f)] private float maximumLaneCorrectionAngleDegrees = 28f;
    [Tooltip("Use 180 if the imported model's visible front points toward local -Z.")]
    [SerializeField] private float modelForwardYawOffset;

    [Header("Obstacle Detection")]
    [SerializeField] private LayerMask obstacleLayers = ~0;
    [SerializeField, Min(0.1f)] private float stoppedDistance = 4.5f;

    [Header("Vehicle Overtaking")]
    [SerializeField] private bool enableOvertaking = true;
    [SerializeField, Min(6f)] private float overtakeStartDistance = 24f;
    [SerializeField, Min(1.5f)] private float overtakeLaneOffset = 5.5f;
    [SerializeField, Min(0.5f)] private float laneOffsetChangePerSecond = 2.6f;
    [SerializeField, Min(2f)] private float distancePastActorBeforeReturn = 6f;
    [SerializeField, Min(5f)] private float laneChangeSpeedKph = 28f;
    [SerializeField, Min(5f)] private float passingSpeedKph = 42f;

    [Header("Maneuver Safety")]
    [SerializeField, Min(0.1f)] private float laneCenterTolerance = 0.65f;
    [SerializeField, Min(1f)] private float laneHeadingToleranceDegrees = 18f;
    [SerializeField, Min(0.1f)] private float clearLaneHoldSeconds = 0.65f;
    [SerializeField, Min(0.1f)] private float blockedLaneHoldSeconds = 0.4f;
    [SerializeField, Min(2f)] private float laneChangeTimeoutSeconds = 9f;
    [SerializeField, Min(0.5f)] private float blockedRecoveryDelaySeconds = 1.25f;
    [SerializeField, Min(0.5f)] private float retryCooldownSeconds = 3f;

    [Header("Risk Driver Demonstration")]
    [Tooltip("Unsafe speed requested by the simulated human driver. ML and the deterministic safety supervisor may reduce it.")]
    [SerializeField, Min(35f)] private float riskDriverRequestedSpeedKph = 80f;
    [Tooltip("How long the reckless driver remains visibly in control before the ML takeover is allowed to command the vehicle.")]
    [SerializeField, Min(1f)] private float riskDriverExposureSeconds = 8f;

    public event Action<float> ObstacleDetected;
    // Preserved for the UI/metrics API. It now means obstacle braking only.
    public event Action BrakingStarted;
    public event Action<VehicleDecelerationReason> DecelerationStarted;
    public event Action DestinationReached;
    public event Action CollisionOccurred;

    public float CurrentSpeedKph => currentSpeedMps * 3.6f;
    public Vector3 MovementForward => GetMovementForward(transform.rotation);
    public Vector3 MovementRight => Quaternion.Euler(0f, 90f, 0f) * MovementForward;
    public Vector3 RouteForward
    {
        get
        {
            if (route.Count < 2)
                return MovementForward;
            Vector3 tangent;
            SampleRoute(routeProgress + 2f, out tangent);
            return tangent.sqrMagnitude > 0.001f ? tangent.normalized : MovementForward;
        }
    }
    public Vector3 RouteRight => Quaternion.Euler(0f, 90f, 0f) * RouteForward;
    public float NearestObstacleDistance { get; private set; } = float.PositiveInfinity;
    public bool IsRunning { get; private set; }
    public bool HasReachedDestination { get; private set; }
    public bool HasConfirmedCollision => collisionReported;
    public bool IsPredictiveEmergencyStopActive { get; private set; }
    public bool IsRiskDrivingActive { get; private set; }
    public bool IsMlRiskTakeoverActive =>
        IsRiskDrivingActive && riskDrivingElapsedSeconds >= riskDriverExposureSeconds;
    public float RiskDrivingSecondsUntilTakeover => IsRiskDrivingActive
        ? Mathf.Max(0f, riskDriverExposureSeconds - riskDrivingElapsedSeconds)
        : 0f;
    public int MlRiskInterventionCount { get; private set; }
    public int SafetyRiskInterventionCount { get; private set; }
    public float CurrentDriverRequestedSpeedKph { get; private set; }
    public float CurrentMlTargetSpeedKph { get; private set; }
    public float CurrentSensorReliability => activeMlDecision != null &&
        activeMlDecision.sensorReliability != null &&
        activeMlDecision.sensorReliability.Length > 0
            ? Mathf.Clamp01(activeMlDecision.overallSensorReliability)
            : 1f;
    public string SensorSafetyMode => activeMlDecision != null &&
        !string.IsNullOrWhiteSpace(activeMlDecision.sensorSafetyMode)
            ? activeMlDecision.sensorSafetyMode
            : "Unavailable";
    public string SensorSummary { get; private set; } = "No sensor scan yet";
    public string DecisionSourceSummary
    {
        get
        {
            if (IsRiskDrivingActive && !IsMlRiskTakeoverActive)
                return $"ML: observing reckless driver; takeover in {RiskDrivingSecondsUntilTakeover:F1} s";
            return mlDecisionBridge != null
                ? mlDecisionBridge.StatusSummary
                : "ML: bridge unavailable (rule controller active)";
        }
    }
    public VehicleDecelerationReason CurrentDecelerationReason { get; private set; }
    public string RiskDrivingSummary
    {
        get
        {
            if (!IsRiskDrivingActive)
                return "Risk driver: inactive";

            if (!IsMlRiskTakeoverActive)
            {
                return $"Reckless driver requests {CurrentDriverRequestedSpeedKph:F1} km/h; " +
                       $"ML is observing and takes over in {RiskDrivingSecondsUntilTakeover:F1} s; " +
                       $"safety interventions {SafetyRiskInterventionCount}";
            }

            string mlStatus = activeMlDecision != null
                ? $"ML takeover target {CurrentMlTargetSpeedKph:F1} km/h"
                : "ML response unavailable; deterministic safety remains active";
            return $"Risk driver requests {CurrentDriverRequestedSpeedKph:F1} km/h; {mlStatus}; " +
                   $"ML interventions {MlRiskInterventionCount}, safety interventions {SafetyRiskInterventionCount}";
        }
    }
    public string CurrentManeuver
    {
        get
        {
            switch (overtakePhase)
            {
                case OvertakePhase.WaitingForGap: return "Waiting for a safe left-lane gap";
                case OvertakePhase.MovingOut: return "Changing to left lane";
                case OvertakePhase.Passing: return "Passing in left lane";
                case OvertakePhase.CruisingLeft: return "Cruising in left lane";
                case OvertakePhase.Returning: return "Returning to right lane";
                case OvertakePhase.Aborting: return "Safely aborting lane change";
                case OvertakePhase.Blocked: return "Blocked - monitoring both lanes";
                default: return CurrentSpeedKph < 0.2f && IsRunning ? "Waiting / stopped" : "Following right lane";
            }
        }
    }

    private enum OvertakePhase
    {
        None,
        WaitingForGap,
        MovingOut,
        Passing,
        CruisingLeft,
        Returning,
        Aborting,
        Blocked
    }

    private readonly List<Vector3> route = new List<Vector3>();
    private readonly List<float> routeDistances = new List<float>();
    private Rigidbody physicsBody;
    private SimulatedVehicleSensorSuite sensorSuite;
    private VirtualPerceptionSensorRig virtualSensorRig;
    private VirtualSensorHealthFeatures[] latestSensorHealth =
        Array.Empty<VirtualSensorHealthFeatures>();
    private TestLabMlDecisionBridge mlDecisionBridge;
    private TestLabMlDecision activeMlDecision;
    private int waypointIndex;
    private float routeProgress;
    private float routeLength;
    private Vector3 routeProjection;
    private float currentSpeedMps;
    private float speedFactor = 1f;
    private float brakingFactor = 1f;
    private float detectionFactor = 1f;
    private float weatherCruisingFactor = 1f;
    private float targetWeatherCruisingFactor = 1f;
    private string weatherContext = "Dry";
    private float previousMlSpeedMps;
    private Vector3 previousMlForward;
    private bool hasMlMotionSample;
    private bool obstacleWasDetected;
    private bool wasObstacleBraking;
    private bool wasMlRiskIntervention;
    private bool wasSafetyRiskIntervention;
    private float riskDrivingElapsedSeconds;
    private bool collisionReported;
    private bool hasEverRun;
    private ScenarioActorMarker overtakingActor;
    private OvertakePhase overtakePhase;
    private bool remainLeftAfterCurrentPass;
    private bool returnWasRequestedFromLeftCruise;
    private float currentLaneOffset;
    private float targetLaneOffset;
    private float physicalLaneOffset;
    private float phaseElapsed;
    private float clearLaneTimer;
    private float blockedLaneTimer;
    private float retryAllowedTime;
    private Vector3 egoLocalCollisionCenter = new Vector3(0f, 0.75f, 0f);
    private Vector3 egoLocalCollisionHalfExtents = new Vector3(0.9f, 0.75f, 2.1f);

    private void Awake()
    {
        EnsurePhysicsBody();
        sensorSuite = GetComponent<SimulatedVehicleSensorSuite>();
        if (sensorSuite == null)
            sensorSuite = gameObject.AddComponent<SimulatedVehicleSensorSuite>();
        virtualSensorRig = GetComponent<VirtualPerceptionSensorRig>();
        if (virtualSensorRig == null)
            virtualSensorRig = gameObject.AddComponent<VirtualPerceptionSensorRig>();
        mlDecisionBridge = GetComponent<TestLabMlDecisionBridge>();
        RefreshEgoCollisionEnvelope();
    }

    private void FixedUpdate()
    {
        if (!IsRunning || route.Count < 2 || HasReachedDestination)
            return;

        IsPredictiveEmergencyStopActive = false;
        float deltaTime = Time.fixedDeltaTime;
        UpdateRiskDrivingPhase(deltaTime);
        UpdateRouteProgress();
        float remainingRouteDistance = routeLength - routeProgress;
        if (remainingRouteDistance <= destinationReachedDistance)
        {
            ApplyDestinationBraking(deltaTime);
            return;
        }

        Vector3 routeForward;
        SampleRoute(routeProgress + 2f, out routeForward);
        if (routeForward.sqrMagnitude <= 0.001f)
            routeForward = MovementForward;
        Vector3 routeRight = Quaternion.Euler(0f, 90f, 0f) * routeForward;
        physicalLaneOffset = Vector3.Dot(physicsBody.position - routeProjection, routeRight);

        Vector3 currentMovementForward = MovementForward;
        float measuredAcceleration = 0f;
        float measuredYawRate = 0f;
        if (hasMlMotionSample)
        {
            measuredAcceleration = (currentSpeedMps - previousMlSpeedMps) / Mathf.Max(0.001f, deltaTime);
            measuredYawRate = Vector3.SignedAngle(
                previousMlForward,
                currentMovementForward,
                Vector3.up
            ) / Mathf.Max(0.001f, deltaTime);
        }
        previousMlSpeedMps = currentSpeedMps;
        previousMlForward = currentMovementForward;
        hasMlMotionSample = true;

        Vector3 egoVelocity = currentMovementForward * currentSpeedMps;
        SimulatedSensorSnapshot groundTruthSensors = sensorSuite.Scan(
            physicsBody.position,
            routeForward,
            routeRight,
            physicalLaneOffset,
            transform,
            egoVelocity
        );
        VirtualPerceptionFrame perception = virtualSensorRig.Observe(
            groundTruthSensors,
            weatherContext,
            Time.realtimeSinceStartup,
            currentSpeedMps,
            measuredYawRate
        );
        SimulatedSensorSnapshot sensors = perception.fusedSnapshot;
        latestSensorHealth = perception.sensorHealth;
        if (mlDecisionBridge != null)
        {
            mlDecisionBridge.RequestDecision(
                sensors,
                currentSpeedMps,
                measuredAcceleration,
                measuredYawRate,
                physicalLaneOffset,
                -overtakeLaneOffset,
                GetMlCruiseSpeedMps(),
                weatherContext,
                latestSensorHealth
            );
            if (!mlDecisionBridge.TryGetFreshDecision(out activeMlDecision))
                activeMlDecision = null;
        }
        else
        {
            activeMlDecision = null;
        }
        UpdateSensorSummary(sensors);
        UpdateOvertakeState(routeForward, sensors, deltaTime);

        float activeLookAhead = steeringLookAheadDistance;
        if (overtakePhase != OvertakePhase.None)
            activeLookAhead = Mathf.Max(activeLookAhead, Mathf.Abs(targetLaneOffset) * 2.75f);
        Vector3 steeringForward;
        Vector3 steeringBaseTarget = SampleRoute(routeProgress + activeLookAhead, out steeringForward);
        Vector3 steeringRight = Quaternion.Euler(0f, 90f, 0f) * steeringForward;
        Vector3 steeringTarget = steeringBaseTarget + steeringRight * currentLaneOffset;
        Vector3 steeringDirection = steeringTarget - physicsBody.position;
        steeringDirection.y = 0f;

        // Keep lane corrections forward-facing. Without this limit, a large
        // lateral error can make the steering target fall effectively behind
        // the car and create the visible forward/back oscillation.
        if (steeringDirection.sqrMagnitude > 0.001f && steeringForward.sqrMagnitude > 0.001f)
        {
            steeringDirection = Vector3.RotateTowards(
                steeringForward.normalized,
                steeringDirection.normalized,
                maximumLaneCorrectionAngleDegrees * Mathf.Deg2Rad,
                0f
            );
        }

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

        float unrestrictedSpeedKph = overtakePhase == OvertakePhase.Passing ||
                                     overtakePhase == OvertakePhase.CruisingLeft
            ? Mathf.Max(maximumSpeedKph, passingSpeedKph, 30f)
            : IsRiskDrivingActive
                ? Mathf.Max(maximumSpeedKph, riskDriverRequestedSpeedKph)
                : maximumSpeedKph;
        float unrestrictedSpeedMps = unrestrictedSpeedKph / 3.6f;
        float effectiveWeatherFactor = targetWeatherCruisingFactor;
        if (activeMlDecision != null &&
            activeMlDecision.weatherModelUsed &&
            activeMlDecision.weatherTargetSpeedMps > 0f &&
            string.Equals(
                activeMlDecision.weatherContext,
                weatherContext,
                StringComparison.OrdinalIgnoreCase
            ))
        {
            // The model learns an absolute cautious speed against its 50 km/h
            // training reference. Convert that cap into a factor for the
            // vehicle's current cruise/overtake/risk-driving request.
            effectiveWeatherFactor = Mathf.Clamp(
                activeMlDecision.weatherTargetSpeedMps /
                    Mathf.Max(0.1f, unrestrictedSpeedMps),
                0.35f,
                1f
            );
        }
        weatherCruisingFactor = Mathf.MoveTowards(
            weatherCruisingFactor,
            effectiveWeatherFactor,
            0.20f * deltaTime
        );
        float desiredSpeedMps = unrestrictedSpeedMps * speedFactor * weatherCruisingFactor;
        CurrentDriverRequestedSpeedKph = IsRiskDrivingActive
            ? Mathf.Max(maximumSpeedKph, riskDriverRequestedSpeedKph)
            : maximumSpeedKph;
        bool mlControlAllowed = !IsRiskDrivingActive || IsMlRiskTakeoverActive;
        CurrentMlTargetSpeedKph = activeMlDecision != null && mlControlAllowed
            ? Mathf.Max(0f, activeMlDecision.targetSpeedMps) * 3.6f
            : 0f;
        VehicleDecelerationReason requestedReason = GetEnvironmentDecelerationReason();

        if (activeMlDecision != null &&
            activeMlDecision.sensorReliability != null &&
            activeMlDecision.sensorReliability.Length > 0)
        {
            string safetyMode = activeMlDecision.sensorSafetyMode ?? "NORMAL";
            if (string.Equals(safetyMode, "MINIMAL_RISK", StringComparison.OrdinalIgnoreCase))
                desiredSpeedMps = 0f;
            else if (string.Equals(safetyMode, "RESTRICTED", StringComparison.OrdinalIgnoreCase))
                desiredSpeedMps = Mathf.Min(desiredSpeedMps, unrestrictedSpeedMps * 0.45f);
            else if (string.Equals(safetyMode, "CAUTIOUS", StringComparison.OrdinalIgnoreCase))
                desiredSpeedMps = Mathf.Min(desiredSpeedMps, unrestrictedSpeedMps * 0.75f);
            if (!string.Equals(safetyMode, "NORMAL", StringComparison.OrdinalIgnoreCase))
                requestedReason = VehicleDecelerationReason.Maneuver;
        }

        // The learned speed target is advisory during cruise/following only.
        // During an active overtake the rule state machine owns the speed, and
        // the ML would otherwise keep braking for the vehicle being passed
        // (still "in front" until the lane change completes) and stop the
        // overtake.
        if (mlControlAllowed && activeMlDecision != null && overtakePhase == OvertakePhase.None)
        {
            float modelTarget = Mathf.Max(0f, activeMlDecision.targetSpeedMps);
            if (modelTarget < desiredSpeedMps - 0.05f)
                requestedReason = VehicleDecelerationReason.Maneuver;
            desiredSpeedMps = Mathf.Min(desiredSpeedMps, modelTarget);
        }

        bool mlRiskIntervention = IsRiskDrivingActive &&
                                  IsMlRiskTakeoverActive &&
                                  activeMlDecision != null &&
                                  overtakePhase == OvertakePhase.None &&
                                  activeMlDecision.targetSpeedMps < unrestrictedSpeedMps - 0.05f;
        if (mlRiskIntervention && !wasMlRiskIntervention)
            MlRiskInterventionCount++;
        wasMlRiskIntervention = mlRiskIntervention;

        float turnAngle = steeringDirection.sqrMagnitude > 0.001f
            ? Vector3.Angle(GetMovementForward(nextRotation), steeringDirection.normalized)
            : 0f;
        if (turnAngle > 35f)
        {
            desiredSpeedMps *= 0.55f;
            requestedReason = VehicleDecelerationReason.Curve;
        }

        // The learned policy and lane planner consume the noisy fused frame.
        // The deterministic collision envelope intentionally retains the
        // untouched Unity truth as a final research-prototype safety backstop.
        bool obstacleLimited = ApplySensorDecision(groundTruthSensors, ref desiredSpeedMps);
        bool safetyRiskIntervention = IsRiskDrivingActive && obstacleLimited;
        if (safetyRiskIntervention && !wasSafetyRiskIntervention)
            SafetyRiskInterventionCount++;
        wasSafetyRiskIntervention = safetyRiskIntervention;
        if (obstacleLimited)
            requestedReason = VehicleDecelerationReason.Obstacle;
        else if (IsLaneChangePhase() && desiredSpeedMps < unrestrictedSpeedMps * speedFactor - 0.05f)
            requestedReason = VehicleDecelerationReason.Maneuver;

        bool decelerating = desiredSpeedMps < currentSpeedMps - 0.05f;
        SetDecelerationReason(decelerating ? requestedReason : VehicleDecelerationReason.None);
        bool obstacleBraking = decelerating && obstacleLimited;
        if (obstacleBraking && !wasObstacleBraking)
            BrakingStarted?.Invoke();
        wasObstacleBraking = obstacleBraking;

        float rate = decelerating
            ? brakingMetersPerSecondSquared * brakingFactor
            : accelerationMetersPerSecondSquared;
        currentSpeedMps = Mathf.MoveTowards(currentSpeedMps, desiredSpeedMps, rate * deltaTime);

        float forwardDistance = currentSpeedMps * deltaTime;
        float nextRouteProgress = Mathf.Min(routeLength, routeProgress + forwardDistance);
        Vector3 nextRouteForward;
        Vector3 nextRoutePosition = SampleRoute(nextRouteProgress, out nextRouteForward);
        if (nextRouteForward.sqrMagnitude <= 0.001f)
            nextRouteForward = routeForward;
        Vector3 nextRouteRight = Quaternion.Euler(0f, 90f, 0f) * nextRouteForward;
        float nextPhysicalLaneOffset = Mathf.MoveTowards(
            physicalLaneOffset,
            currentLaneOffset,
            laneOffsetChangePerSecond * deltaTime
        );
        Vector3 nextPosition = nextRoutePosition + nextRouteRight * nextPhysicalLaneOffset;
        nextPosition.y = physicsBody.position.y;
        Vector3 displacement = nextPosition - physicsBody.position;
        displacement.y = 0f;
        float movementDistance = displacement.magnitude;
        Vector3 movementDirection = movementDistance > 0.0001f
            ? displacement / movementDistance
            : nextRouteForward;
        if (TryGetConfirmedActorOverlap(out Collider overlappingActor))
        {
            ReportCollision(overlappingActor);
            return;
        }

        if (movementDistance > 0f &&
            PredictCollision(nextRotation, movementDirection, movementDistance, out RaycastHit predictedHit) &&
            !CanSafelyContinueCurrentLaneChange(predictedHit.collider))
        {
            ApplyPredictiveEmergencyStop(predictedHit.distance);
            return;
        }

        physicsBody.MoveRotation(nextRotation);
        physicsBody.MovePosition(nextPosition);
    }

    public void SetRoute(IReadOnlyList<Vector3> worldRoute, float initialSpeedKph)
    {
        route.Clear();
        if (worldRoute != null)
        {
            for (int index = 0; index < worldRoute.Count; index++)
                route.Add(worldRoute[index]);
        }

        BuildRouteDistances();
        routeProgress = FindClosestRouteProgress(transform.position, out routeProjection, out waypointIndex);
        currentSpeedMps = Mathf.Max(0f, initialSpeedKph / 3.6f);
        HasReachedDestination = false;
        collisionReported = false;
        hasEverRun = false;
        IsRiskDrivingActive = false;
        riskDrivingElapsedSeconds = 0f;
        MlRiskInterventionCount = 0;
        SafetyRiskInterventionCount = 0;
        CurrentDriverRequestedSpeedKph = maximumSpeedKph;
        CurrentMlTargetSpeedKph = 0f;
        wasMlRiskIntervention = false;
        wasSafetyRiskIntervention = false;
        activeMlDecision = null;
        hasMlMotionSample = false;
        previousMlSpeedMps = currentSpeedMps;
        previousMlForward = MovementForward;
        IsPredictiveEmergencyStopActive = false;
        obstacleWasDetected = false;
        wasObstacleBraking = false;
        NearestObstacleDistance = float.PositiveInfinity;
        SetDecelerationReason(VehicleDecelerationReason.None);
        ResetOvertake();
        RefreshEgoCollisionEnvelope();
        AlignWithRoute();
        mlDecisionBridge?.BeginSession();
    }

    public void SetMlDecisionBridge(TestLabMlDecisionBridge bridge)
    {
        mlDecisionBridge = bridge;
    }

    public void SetRunning(bool shouldRun)
    {
        IsRunning = shouldRun && route.Count > 1 && !HasReachedDestination;
        if (IsRunning)
            hasEverRun = true;
    }

    public void SetRiskDriving(bool enabled)
    {
        if (IsRiskDrivingActive == enabled)
            return;

        IsRiskDrivingActive = enabled;
        riskDrivingElapsedSeconds = 0f;
        MlRiskInterventionCount = 0;
        SafetyRiskInterventionCount = 0;
        CurrentDriverRequestedSpeedKph = enabled
            ? Mathf.Max(maximumSpeedKph, riskDriverRequestedSpeedKph)
            : maximumSpeedKph;
        CurrentMlTargetSpeedKph = 0f;
        wasMlRiskIntervention = false;
        wasSafetyRiskIntervention = false;
    }

    private void UpdateRiskDrivingPhase(float deltaTime)
    {
        if (!IsRiskDrivingActive || IsMlRiskTakeoverActive)
            return;

        riskDrivingElapsedSeconds = Mathf.Min(
            riskDriverExposureSeconds,
            riskDrivingElapsedSeconds + Mathf.Max(0f, deltaTime)
        );
    }

    public void SetEnvironmentModifiers(float newSpeedFactor, float newBrakingFactor, float newDetectionFactor)
    {
        speedFactor = Mathf.Clamp(newSpeedFactor, 0.1f, 1.5f);
        brakingFactor = Mathf.Clamp(newBrakingFactor, 0.1f, 1.5f);
        detectionFactor = Mathf.Clamp(newDetectionFactor, 0.1f, 1.5f);
        if (sensorSuite != null)
            sensorSuite.SetRangeFactor(detectionFactor);
    }

    public void SetWeatherCruisingFactor(float factor)
    {
        // This remains the deterministic fallback when the Python service or
        // weather model is unavailable. A fresh ML response supersedes it.
        targetWeatherCruisingFactor = Mathf.Clamp(factor, 0.35f, 1f);
    }

    public void SetWeatherContext(string context)
    {
        weatherContext = string.IsNullOrWhiteSpace(context) ? "Dry" : context;
    }

    public void ConfigureMaximumSpeed(float speedKph)
    {
        maximumSpeedKph = Mathf.Max(1f, speedKph);
    }

    public void ConfigureModelForwardYawOffset(float yawOffset)
    {
        modelForwardYawOffset = yawOffset;
    }

    public void PrepareForNewScenario()
    {
        // Do not let a newly added obstacle inherit the recovery cooldown or a
        // stale blocked maneuver from the previous scenario.
        retryAllowedTime = 0f;
        IsPredictiveEmergencyStopActive = false;
        obstacleWasDetected = false;
        wasObstacleBraking = false;
        NearestObstacleDistance = float.PositiveInfinity;

        if (overtakePhase == OvertakePhase.WaitingForGap ||
            overtakePhase == OvertakePhase.Blocked ||
            overtakePhase == OvertakePhase.Aborting)
            ResetOvertake();

        if (hasEverRun && !HasReachedDestination)
        {
            collisionReported = false;
            IsRunning = true;
        }
    }

    private void BuildRouteDistances()
    {
        routeDistances.Clear();
        routeLength = 0f;
        if (route.Count == 0)
            return;
        routeDistances.Add(0f);
        for (int index = 1; index < route.Count; index++)
        {
            Vector3 segment = route[index] - route[index - 1];
            segment.y = 0f;
            routeLength += segment.magnitude;
            routeDistances.Add(routeLength);
        }
    }

    private void UpdateRouteProgress()
    {
        if (route.Count < 2)
            return;

        int startSegment = Mathf.Max(0, waypointIndex - 8);
        int endSegment = Mathf.Min(route.Count - 2, waypointIndex + 24);
        float bestDistanceSquared = float.PositiveInfinity;
        float bestProgress = routeProgress;
        Vector3 bestProjection = routeProjection;

        for (int index = startSegment; index <= endSegment; index++)
        {
            Vector3 start = route[index];
            Vector3 segment = route[index + 1] - start;
            start.y = physicsBody.position.y;
            segment.y = 0f;
            float segmentLengthSquared = segment.sqrMagnitude;
            if (segmentLengthSquared <= 0.0001f)
                continue;
            float t = Mathf.Clamp01(Vector3.Dot(physicsBody.position - start, segment) / segmentLengthSquared);
            Vector3 projection = start + segment * t;
            float distanceSquared = (physicsBody.position - projection).sqrMagnitude;
            float candidateProgress = routeDistances[index] + Mathf.Sqrt(segmentLengthSquared) * t;
            if (distanceSquared < bestDistanceSquared && candidateProgress >= routeProgress - waypointReachedDistance)
            {
                bestDistanceSquared = distanceSquared;
                bestProgress = candidateProgress;
                bestProjection = projection;
            }
        }

        routeProgress = Mathf.Max(routeProgress, bestProgress);
        routeProjection = bestProjection;
        while (waypointIndex < route.Count - 1 &&
               routeDistances[waypointIndex] <= routeProgress + waypointReachedDistance)
            waypointIndex++;
    }

    private float FindClosestRouteProgress(Vector3 position, out Vector3 projection, out int nextWaypoint)
    {
        projection = route.Count > 0 ? route[0] : position;
        nextWaypoint = route.Count > 1 ? 1 : 0;
        float bestDistanceSquared = float.PositiveInfinity;
        float bestProgress = 0f;
        for (int index = 0; index < route.Count - 1; index++)
        {
            Vector3 start = route[index];
            Vector3 segment = route[index + 1] - start;
            start.y = position.y;
            segment.y = 0f;
            float lengthSquared = segment.sqrMagnitude;
            if (lengthSquared <= 0.0001f)
                continue;
            float t = Mathf.Clamp01(Vector3.Dot(position - start, segment) / lengthSquared);
            Vector3 candidate = start + segment * t;
            float distanceSquared = (position - candidate).sqrMagnitude;
            if (distanceSquared >= bestDistanceSquared)
                continue;
            bestDistanceSquared = distanceSquared;
            projection = candidate;
            bestProgress = routeDistances[index] + Mathf.Sqrt(lengthSquared) * t;
            nextWaypoint = index + 1;
        }
        return bestProgress;
    }

    private Vector3 SampleRoute(float distance, out Vector3 tangent)
    {
        if (route.Count < 2)
        {
            tangent = MovementForward;
            return route.Count == 1 ? route[0] : physicsBody.position;
        }

        float clamped = Mathf.Clamp(distance, 0f, routeLength);
        int segmentIndex = Mathf.Clamp(waypointIndex - 1, 0, route.Count - 2);
        while (segmentIndex > 0 && routeDistances[segmentIndex] > clamped)
            segmentIndex--;
        while (segmentIndex < route.Count - 2 && routeDistances[segmentIndex + 1] < clamped)
            segmentIndex++;

        Vector3 segment = route[segmentIndex + 1] - route[segmentIndex];
        segment.y = 0f;
        float length = Mathf.Max(0.0001f, segment.magnitude);
        tangent = segment / length;
        float t = Mathf.Clamp01((clamped - routeDistances[segmentIndex]) / length);
        return Vector3.Lerp(route[segmentIndex], route[segmentIndex + 1], t);
    }

    private void AlignWithRoute()
    {
        if (route.Count < 2)
            return;
        Vector3 direction;
        SampleRoute(routeProgress + steeringLookAheadDistance, out direction);
        if (direction.sqrMagnitude <= 0.001f)
            return;
        Quaternion aligned = Quaternion.LookRotation(direction, Vector3.up)
            * Quaternion.Euler(0f, -modelForwardYawOffset, 0f);
        physicsBody.rotation = aligned;
        transform.rotation = aligned;
    }

    private void UpdateOvertakeState(
        Vector3 routeForward,
        SimulatedSensorSnapshot sensors,
        float deltaTime)
    {
        phaseElapsed += deltaTime;
        currentLaneOffset = Mathf.MoveTowards(
            currentLaneOffset,
            targetLaneOffset,
            laneOffsetChangePerSecond * deltaTime
        );

        if (overtakePhase == OvertakePhase.None)
        {
            SimulatedActorObservation actorAhead = sensors.rightLaneFront;
            float triggerDistance = GetOvertakeTriggerDistance(actorAhead);
            bool modelRequestsLeft = activeMlDecision != null &&
                activeMlDecision.Action == TestLabMlAction.ChangeLeft;
            bool ruleRequestsOvertake = IsOvertakeCandidate(actorAhead) &&
                actorAhead.gapDistance <= triggerDistance;
            if (Time.time >= retryAllowedTime && IsOvertakeCandidate(actorAhead) &&
                (modelRequestsLeft || ruleRequestsOvertake))
            {
                overtakingActor = actorAhead.actor;
                remainLeftAfterCurrentPass = ShouldRemainLeftAfterPassing(overtakingActor);
                returnWasRequestedFromLeftCruise = false;
                targetLaneOffset = 0f;
                TransitionTo(OvertakePhase.WaitingForGap);
            }
            return;
        }

        if (overtakingActor == null &&
            overtakePhase != OvertakePhase.CruisingLeft &&
            overtakePhase != OvertakePhase.Returning &&
            overtakePhase != OvertakePhase.Aborting)
        {
            if (remainLeftAfterCurrentPass && IsCenteredInLeftLane(routeForward))
                TransitionTo(OvertakePhase.CruisingLeft);
            else
            {
                targetLaneOffset = 0f;
                TransitionTo(OvertakePhase.Aborting);
            }
        }

        // While changing lanes, the vehicle being passed can overlap the
        // sensor's left-lane boundary. It must not block its own overtake.
        bool leftClear = IsLaneClearIgnoring(
            sensors.leftLaneFront,
            sensors.leftLaneRear,
            overtakingActor,
            25f,
            12f
        );
        bool rightClear = sensors.IsRightLaneClear(18f, 10f);

        switch (overtakePhase)
        {
            case OvertakePhase.WaitingForGap:
                // A scenario mover starts with the test vehicle, so refresh this
                // policy after its measured velocity becomes available.
                if (overtakingActor != null)
                    remainLeftAfterCurrentPass = ShouldRemainLeftAfterPassing(overtakingActor);
                HoldClearTimer(leftClear, deltaTime);
                if (clearLaneTimer >= clearLaneHoldSeconds)
                {
                    targetLaneOffset = -overtakeLaneOffset;
                    TransitionTo(OvertakePhase.MovingOut);
                }
                else if (phaseElapsed >= laneChangeTimeoutSeconds)
                {
                    targetLaneOffset = 0f;
                    TransitionTo(OvertakePhase.Blocked);
                }
                break;

            case OvertakePhase.MovingOut:
                HoldBlockedTimer(!leftClear, deltaTime);
                if (IsCenteredInLeftLane(routeForward))
                {
                    targetLaneOffset = -overtakeLaneOffset;
                    TransitionTo(OvertakePhase.Passing);
                }
                else if (blockedLaneTimer >= blockedLaneHoldSeconds)
                {
                    targetLaneOffset = physicalLaneOffset > -overtakeLaneOffset * 0.55f && rightClear
                        ? 0f
                        : -overtakeLaneOffset;
                    TransitionTo(targetLaneOffset == 0f ? OvertakePhase.Aborting : OvertakePhase.Blocked);
                }
                else if (phaseElapsed >= laneChangeTimeoutSeconds)
                {
                    targetLaneOffset = rightClear ? 0f : -overtakeLaneOffset;
                    TransitionTo(rightClear ? OvertakePhase.Aborting : OvertakePhase.Blocked);
                }
                break;

            case OvertakePhase.Passing:
                HoldBlockedTimer(IsDangerousFrontObservation(sensors.leftLaneFront, overtakingActor), deltaTime);
                if (blockedLaneTimer >= blockedLaneHoldSeconds)
                {
                    targetLaneOffset = -overtakeLaneOffset;
                    TransitionTo(OvertakePhase.Blocked);
                    break;
                }
                if (HasPassedOvertakeActor(routeForward))
                {
                    if (remainLeftAfterCurrentPass)
                    {
                        targetLaneOffset = -overtakeLaneOffset;
                        TransitionTo(OvertakePhase.CruisingLeft);
                    }
                    else if (rightClear)
                    {
                        returnWasRequestedFromLeftCruise = false;
                        targetLaneOffset = 0f;
                        TransitionTo(OvertakePhase.Returning);
                    }
                }
                break;

            case OvertakePhase.CruisingLeft:
                targetLaneOffset = -overtakeLaneOffset;
                SimulatedActorObservation leftObstacle = sensors.leftLaneFront;
                bool stoppedLeftVehicle = IsStoppedVehicle(leftObstacle) &&
                    leftObstacle.gapDistance <= overtakeStartDistance;
                bool modelRequestsRight = activeMlDecision != null &&
                    activeMlDecision.Action == TestLabMlAction.ChangeRight;
                HoldClearTimer((stoppedLeftVehicle || modelRequestsRight) && rightClear, deltaTime);
                if (clearLaneTimer >= clearLaneHoldSeconds)
                {
                    overtakingActor = leftObstacle != null ? leftObstacle.actor : null;
                    returnWasRequestedFromLeftCruise = true;
                    targetLaneOffset = 0f;
                    TransitionTo(OvertakePhase.Returning);
                }
                break;

            case OvertakePhase.Returning:
                HoldBlockedTimer(!rightClear, deltaTime);
                if (blockedLaneTimer >= blockedLaneHoldSeconds && physicalLaneOffset < -1.5f)
                {
                    targetLaneOffset = -overtakeLaneOffset;
                    TransitionTo(returnWasRequestedFromLeftCruise
                        ? OvertakePhase.CruisingLeft
                        : OvertakePhase.Blocked);
                }
                else if (IsCenteredInRightLane(routeForward))
                {
                    retryAllowedTime = Time.time + retryCooldownSeconds;
                    ResetOvertake();
                }
                else if (phaseElapsed >= laneChangeTimeoutSeconds)
                {
                    if (rightClear)
                    {
                        retryAllowedTime = Time.time + retryCooldownSeconds;
                        ResetOvertake();
                    }
                    else
                    {
                        targetLaneOffset = -overtakeLaneOffset;
                        TransitionTo(OvertakePhase.Blocked);
                    }
                }
                break;

            case OvertakePhase.Aborting:
                targetLaneOffset = 0f;
                HoldBlockedTimer(!rightClear, deltaTime);
                if (blockedLaneTimer >= blockedLaneHoldSeconds && physicalLaneOffset < -1.5f)
                {
                    targetLaneOffset = -overtakeLaneOffset;
                    TransitionTo(OvertakePhase.Blocked);
                }
                else if (IsCenteredInRightLane(routeForward) || phaseElapsed >= laneChangeTimeoutSeconds)
                {
                    retryAllowedTime = Time.time + retryCooldownSeconds;
                    ResetOvertake();
                }
                break;

            case OvertakePhase.Blocked:
                if (phaseElapsed < blockedRecoveryDelaySeconds)
                    break;
                bool currentlyLeft = physicalLaneOffset < -overtakeLaneOffset * 0.55f;
                HoldClearTimer(leftClear, deltaTime);
                if (currentlyLeft && IsCenteredInLeftLane(routeForward) && !IsDangerousFrontObservation(sensors.leftLaneFront, overtakingActor))
                {
                    targetLaneOffset = -overtakeLaneOffset;
                    TransitionTo(OvertakePhase.Passing);
                }
                else if (clearLaneTimer >= clearLaneHoldSeconds && !currentlyLeft)
                {
                    targetLaneOffset = -overtakeLaneOffset;
                    TransitionTo(OvertakePhase.MovingOut);
                }
                else if (!leftClear && rightClear && physicalLaneOffset < -1.5f)
                {
                    targetLaneOffset = 0f;
                    TransitionTo(OvertakePhase.Aborting);
                }
                break;
        }
    }

    private bool ApplySensorDecision(SimulatedSensorSnapshot sensors, ref float desiredSpeedMps)
    {
        SimulatedActorObservation relevant = null;
        bool obstacleLimited = false;

        if (sensors.pedestrianHazard != null)
        {
            relevant = sensors.pedestrianHazard;
            obstacleLimited = ApplyFollowingSpeed(relevant, ref desiredSpeedMps, 1.4f);
            UpdateDetectedObstacle(relevant.gapDistance);
            return obstacleLimited;
        }

        switch (overtakePhase)
        {
            case OvertakePhase.MovingOut:
                relevant = physicalLaneOffset > -overtakeLaneOffset * 0.55f
                    ? sensors.rightLaneFront
                    : sensors.leftLaneFront;
                desiredSpeedMps = Mathf.Min(desiredSpeedMps, GetLaneChangeSpeedMps());
                break;
            case OvertakePhase.Passing:
            case OvertakePhase.CruisingLeft:
                relevant = sensors.leftLaneFront;
                desiredSpeedMps = Mathf.Min(desiredSpeedMps, GetOvertakeCruiseSpeedMps());
                break;
            case OvertakePhase.Returning:
                relevant = sensors.rightLaneFront;
                desiredSpeedMps = Mathf.Min(desiredSpeedMps, GetLaneChangeSpeedMps());
                break;
            case OvertakePhase.Blocked:
                relevant = physicalLaneOffset < -overtakeLaneOffset * 0.5f
                    ? sensors.leftLaneFront
                    : sensors.rightLaneFront;
                if (relevant != null && relevant.actor == overtakingActor &&
                    physicalLaneOffset < -overtakeLaneOffset * 0.25f)
                    relevant = null;
                desiredSpeedMps = Mathf.Min(desiredSpeedMps, GetLaneChangeSpeedMps());
                break;
            case OvertakePhase.WaitingForGap:
            case OvertakePhase.Aborting:
            case OvertakePhase.None:
                relevant = sensors.rightLaneFront;
                break;
        }

        if (relevant != null && relevant.actor != overtakingActor)
            obstacleLimited = ApplyFollowingSpeed(relevant, ref desiredSpeedMps, 0.9f);
        else if (relevant != null &&
                 (overtakePhase == OvertakePhase.None ||
                  overtakePhase == OvertakePhase.WaitingForGap ||
                  (overtakePhase == OvertakePhase.Blocked &&
                   physicalLaneOffset >= -overtakeLaneOffset * 0.25f)))
            obstacleLimited = ApplyFollowingSpeed(relevant, ref desiredSpeedMps, 0.9f);

        UpdateDetectedObstacle(relevant != null ? relevant.gapDistance : float.PositiveInfinity);
        return obstacleLimited;
    }

    private bool ApplyFollowingSpeed(
        SimulatedActorObservation observation,
        ref float desiredSpeedMps,
        float extraTimeGap)
    {
        if (observation == null)
            return false;

        float safeGap = stoppedDistance + currentSpeedMps * extraTimeGap;
        if (observation.gapDistance <= safeGap || observation.timeToCollision <= 1.5f)
        {
            desiredSpeedMps = 0f;
            return true;
        }

        float fullSpeedGap = safeGap + 16f;
        float gapFactor = Mathf.InverseLerp(safeGap, fullSpeedGap, observation.gapDistance);
        float actorSpeed = Mathf.Max(0f, observation.actorLongitudinalSpeed);
        float gapSpeed = Mathf.Lerp(actorSpeed, maximumSpeedKph * speedFactor / 3.6f, gapFactor);
        if (observation.timeToCollision < 4f)
            gapSpeed = Mathf.Min(gapSpeed, currentSpeedMps * Mathf.InverseLerp(1.5f, 4f, observation.timeToCollision));
        float previousDesired = desiredSpeedMps;
        desiredSpeedMps = Mathf.Min(desiredSpeedMps, gapSpeed);
        return desiredSpeedMps < previousDesired - 0.01f;
    }

    private bool IsOvertakeCandidate(SimulatedActorObservation observation)
    {
        bool reliabilityAllowsLaneChange = activeMlDecision == null ||
            activeMlDecision.sensorReliability == null ||
            activeMlDecision.sensorReliability.Length == 0 ||
            string.Equals(activeMlDecision.sensorSafetyMode, "NORMAL", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(activeMlDecision.sensorSafetyMode, "CAUTIOUS", StringComparison.OrdinalIgnoreCase);
        if (!enableOvertaking || !reliabilityAllowsLaneChange ||
            observation == null || observation.actor == null)
            return false;
        ScenarioActorMarker actor = observation.actor;
        if (!actor.CanOvertake || actor.SemanticClass == ScenarioActorClass.Pedestrian)
            return false;
        return actor.MotionState == ScenarioActorMotionState.Stopped ||
               actor.MotionState == ScenarioActorMotionState.Slow ||
               observation.relativeClosingSpeed > 0.5f;
    }

    private float GetOvertakeTriggerDistance(SimulatedActorObservation observation)
    {
        if (observation?.actor == null)
            return overtakeStartDistance;
        return observation.actor.SemanticClass == ScenarioActorClass.Truck ||
               observation.actor.SemanticClass == ScenarioActorClass.Bus
            ? overtakeStartDistance + 10f
            : overtakeStartDistance;
    }

    private static bool IsLaneClearIgnoring(
        SimulatedActorObservation front,
        SimulatedActorObservation rear,
        ScenarioActorMarker ignoredActor,
        float requiredFrontGap,
        float requiredRearGap)
    {
        bool frontClear = front == null || front.actor == ignoredActor ||
            (front.gapDistance >= requiredFrontGap && front.timeToCollision >= 4f);
        bool rearClear = rear == null || rear.actor == ignoredActor ||
            (rear.gapDistance >= requiredRearGap && rear.timeToCollision >= 4.5f);
        return frontClear && rearClear;
    }

    private static bool IsStoppedVehicle(SimulatedActorObservation observation)
    {
        return observation != null && observation.actor != null &&
               observation.actor.SemanticClass != ScenarioActorClass.Pedestrian &&
               observation.actor.MotionState == ScenarioActorMotionState.Stopped;
    }

    private bool IsDangerousFrontObservation(
        SimulatedActorObservation observation,
        ScenarioActorMarker ignoredActor)
    {
        if (observation == null || observation.actor == ignoredActor)
            return false;
        return observation.gapDistance < 15f || observation.timeToCollision < 4f;
    }

    private bool HasPassedOvertakeActor(Vector3 routeForward)
    {
        if (overtakingActor == null)
            return true;
        Vector3 difference = overtakingActor.transform.position - physicsBody.position;
        difference.y = 0f;
        return Vector3.Dot(difference, routeForward) <= -distancePastActorBeforeReturn;
    }

    private bool IsCenteredInLeftLane(Vector3 routeForward)
    {
        return Mathf.Abs(physicalLaneOffset + overtakeLaneOffset) <= laneCenterTolerance &&
               Vector3.Angle(MovementForward, routeForward) <= laneHeadingToleranceDegrees;
    }

    private bool IsCenteredInRightLane(Vector3 routeForward)
    {
        return Mathf.Abs(physicalLaneOffset) <= laneCenterTolerance &&
               Vector3.Angle(MovementForward, routeForward) <= laneHeadingToleranceDegrees;
    }

    private void HoldClearTimer(bool clear, float deltaTime)
    {
        clearLaneTimer = UpdateHysteresisTimer(clearLaneTimer, clear, deltaTime);
    }

    private void HoldBlockedTimer(bool blocked, float deltaTime)
    {
        blockedLaneTimer = UpdateHysteresisTimer(blockedLaneTimer, blocked, deltaTime);
    }

    private static float UpdateHysteresisTimer(float current, bool condition, float deltaTime)
    {
        float safeDelta = Mathf.Max(0f, deltaTime);
        return condition
            ? current + safeDelta
            : Mathf.Max(0f, current - safeDelta * 2f);
    }

    private void TransitionTo(OvertakePhase next)
    {
        if (overtakePhase == next)
            return;
        overtakePhase = next;
        phaseElapsed = 0f;
        clearLaneTimer = 0f;
        blockedLaneTimer = 0f;
    }

    private bool IsLaneChangePhase()
    {
        return overtakePhase == OvertakePhase.MovingOut ||
               overtakePhase == OvertakePhase.Returning ||
               overtakePhase == OvertakePhase.Aborting;
    }

    private float GetLaneChangeSpeedMps()
    {
        float normalMaximumKph = maximumSpeedKph * speedFactor;
        float targetKph = Mathf.Max(laneChangeSpeedKph, 22f * speedFactor);
        return Mathf.Min(normalMaximumKph, targetKph) / 3.6f;
    }

    private float GetOvertakeCruiseSpeedMps()
    {
        float targetKph = Mathf.Max(30f, maximumSpeedKph, passingSpeedKph);
        return targetKph * speedFactor / 3.6f;
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
            $"{DecisionSourceSummary}\n" +
            $"Range: {sensorSuite.EffectiveForwardRange:F0} m front / {sensorSuite.EffectiveRearRange:F0} m rear\n" +
            $"Right front: {DescribeObservation(sensors.rightLaneFront)}\n" +
            $"Right rear: {DescribeObservation(sensors.rightLaneRear)}\n" +
            $"Left front: {DescribeObservation(sensors.leftLaneFront)}\n" +
            $"Left rear: {DescribeObservation(sensors.leftLaneRear)}\n" +
            $"Pedestrian risk: {DescribeObservation(sensors.pedestrianHazard)}\n" +
            $"Virtual rig: {(virtualSensorRig != null ? virtualSensorRig.FaultSummary : "unavailable")}";
    }

    private float GetMlCruiseSpeedMps()
    {
        bool passing = overtakePhase == OvertakePhase.MovingOut ||
                       overtakePhase == OvertakePhase.Passing ||
                       overtakePhase == OvertakePhase.CruisingLeft;
        float targetKph = passing
            ? Mathf.Max(maximumSpeedKph, passingSpeedKph, 30f)
            : maximumSpeedKph;
        // Send the unmodified reference to Python. Unity applies either the
        // learned weather factor or the fallback exactly once after the reply.
        return targetKph * speedFactor / 3.6f;
    }

    private static string DescribeObservation(SimulatedActorObservation observation)
    {
        if (observation == null)
            return "Clear";
        string actorName = observation.actor != null
            ? $"{observation.actor.SemanticClass}/{observation.actor.MotionState}"
            : "Unregistered object";
        string ttc = float.IsInfinity(observation.timeToCollision)
            ? "not closing"
            : $"TTC {observation.timeToCollision:F1} s";
        return $"{actorName}, {observation.gapDistance:F1} m, closing {observation.relativeClosingSpeed:F1} m/s, {ttc}";
    }

    private bool TryGetConfirmedActorOverlap(out Collider collision)
    {
        Vector3 halfExtents = GetCollisionHalfExtents();
        Vector3 center = physicsBody.position + physicsBody.rotation * egoLocalCollisionCenter;
        Collider[] candidates = Physics.OverlapBox(
            center,
            halfExtents,
            physicsBody.rotation,
            obstacleLayers,
            QueryTriggerInteraction.Ignore
        );
        foreach (Collider candidate in candidates)
        {
            if (ShouldIgnoreCollider(candidate) || !HasVerifiedPenetration(candidate))
                continue;
            collision = candidate;
            return true;
        }
        collision = null;
        return false;
    }

    private bool PredictCollision(
        Quaternion nextRotation,
        Vector3 movementForward,
        float movementDistance,
        out RaycastHit collision)
    {
        // Keep the confirmed-collision envelope full-sized, but make the
        // predictive envelope slightly narrower so an adjacent vehicle in the
        // other lane does not cause repeated false emergency stops.
        Vector3 halfExtents = Vector3.Scale(
            GetCollisionHalfExtents(),
            new Vector3(0.84f, 0.92f, 0.9f)
        );
        Vector3 center = physicsBody.position + physicsBody.rotation * egoLocalCollisionCenter;
        RaycastHit[] hits = Physics.BoxCastAll(
            center,
            halfExtents,
            movementForward,
            nextRotation,
            movementDistance + 0.2f,
            obstacleLayers,
            QueryTriggerInteraction.Ignore
        );
        foreach (RaycastHit hit in hits)
        {
            if (ShouldIgnoreCollider(hit.collider))
                continue;
            collision = hit;
            return true;
        }
        collision = default;
        return false;
    }

    private bool CanSafelyContinueCurrentLaneChange(Collider predictedCollider)
    {
        if (predictedCollider == null || overtakingActor == null || targetLaneOffset >= -0.5f)
            return false;

        ScenarioActorMarker predictedActor = predictedCollider.GetComponentInParent<ScenarioActorMarker>();
        if (predictedActor != overtakingActor)
            return false;

        return overtakePhase == OvertakePhase.MovingOut ||
               overtakePhase == OvertakePhase.Passing ||
               overtakePhase == OvertakePhase.Blocked;
    }

    private Vector3 GetCollisionHalfExtents()
    {
        return Vector3.Max(
            egoLocalCollisionHalfExtents * 0.94f,
            new Vector3(0.25f, 0.25f, 0.5f)
        );
    }

    private bool HasVerifiedPenetration(Collider actorCollider)
    {
        Collider[] egoColliders = GetComponentsInChildren<Collider>(false);
        foreach (Collider egoCollider in egoColliders)
        {
            if (egoCollider == null || !egoCollider.enabled || egoCollider.isTrigger ||
                egoCollider == actorCollider)
                continue;

            if (Physics.ComputePenetration(
                egoCollider,
                egoCollider.transform.position,
                egoCollider.transform.rotation,
                actorCollider,
                actorCollider.transform.position,
                actorCollider.transform.rotation,
                out _,
                out _))
                return true;
        }
        return false;
    }

    private void ApplyPredictiveEmergencyStop(float predictedDistance)
    {
        bool wasMoving = currentSpeedMps > 0.05f;
        currentSpeedMps = 0f;
        IsPredictiveEmergencyStopActive = true;
        SetDecelerationReason(VehicleDecelerationReason.Obstacle);
        UpdateDetectedObstacle(Mathf.Max(0f, predictedDistance));
        if (wasMoving && !wasObstacleBraking)
            BrakingStarted?.Invoke();
        wasObstacleBraking = true;
    }

    private VehicleDecelerationReason GetEnvironmentDecelerationReason()
    {
        return speedFactor < 0.999f || weatherCruisingFactor < 0.999f
            ? VehicleDecelerationReason.Weather
            : VehicleDecelerationReason.None;
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
        return candidate.GetComponentInParent<ScenarioActorMarker>() == null;
    }

    private void RefreshEgoCollisionEnvelope()
    {
        Collider[] colliders = GetComponentsInChildren<Collider>(false);
        bool found = false;
        Bounds localBounds = default;
        foreach (Collider egoCollider in colliders)
        {
            if (egoCollider == null || !egoCollider.enabled || egoCollider.isTrigger)
                continue;
            Bounds worldBounds = egoCollider.bounds;
            Vector3 min = worldBounds.min;
            Vector3 max = worldBounds.max;
            for (int corner = 0; corner < 8; corner++)
            {
                Vector3 worldCorner = new Vector3(
                    (corner & 1) == 0 ? min.x : max.x,
                    (corner & 2) == 0 ? min.y : max.y,
                    (corner & 4) == 0 ? min.z : max.z
                );
                Vector3 localCorner = transform.InverseTransformPoint(worldCorner);
                if (!found)
                {
                    localBounds = new Bounds(localCorner, Vector3.zero);
                    found = true;
                }
                else
                {
                    localBounds.Encapsulate(localCorner);
                }
            }
        }
        if (found)
        {
            egoLocalCollisionCenter = localBounds.center;
            egoLocalCollisionHalfExtents = localBounds.extents;
        }
    }

    private Vector3 GetMovementForward(Quaternion visualRotation)
    {
        Vector3 localForward = Quaternion.Euler(0f, modelForwardYawOffset, 0f) * Vector3.forward;
        return (visualRotation * localForward).normalized;
    }

    private void ResetOvertake()
    {
        overtakingActor = null;
        overtakePhase = OvertakePhase.None;
        remainLeftAfterCurrentPass = false;
        returnWasRequestedFromLeftCruise = false;
        currentLaneOffset = 0f;
        targetLaneOffset = 0f;
        phaseElapsed = 0f;
        clearLaneTimer = 0f;
        blockedLaneTimer = 0f;
    }

    private static bool ShouldRemainLeftAfterPassing(ScenarioActorMarker actor)
    {
        if (actor == null || actor.MotionState == ScenarioActorMotionState.Stopped)
            return false;
        return actor.SemanticClass == ScenarioActorClass.Truck ||
               actor.SemanticClass == ScenarioActorClass.Bus ||
               actor.MotionState == ScenarioActorMotionState.Slow;
    }

    private void ApplyDestinationBraking(float deltaTime)
    {
        SetDecelerationReason(VehicleDecelerationReason.Destination);
        wasObstacleBraking = false;
        currentSpeedMps = Mathf.MoveTowards(
            currentSpeedMps,
            0f,
            brakingMetersPerSecondSquared * brakingFactor * deltaTime
        );
        if (currentSpeedMps <= 0.05f)
            FinishRoute();
    }

    private void SetDecelerationReason(VehicleDecelerationReason reason)
    {
        if (CurrentDecelerationReason == reason)
            return;
        CurrentDecelerationReason = reason;
        if (reason != VehicleDecelerationReason.None)
            DecelerationStarted?.Invoke(reason);
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

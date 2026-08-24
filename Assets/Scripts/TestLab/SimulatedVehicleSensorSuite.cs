using System.Collections.Generic;
using UnityEngine;

public sealed class SimulatedActorObservation
{
    public ScenarioActorMarker actor;
    public Collider collider;
    public Bounds combinedBounds;
    public float longitudinalDistance;
    public float laneOffset;
    public float gapDistance;
    public float actorLongitudinalSpeed;
    public float relativeClosingSpeed;
    public float timeToCollision = float.PositiveInfinity;

    public bool IsValid => actor != null || collider != null;
    public bool IsClosing => relativeClosingSpeed > 0.05f;
}

public sealed class SimulatedSensorSnapshot
{
    public SimulatedActorObservation pedestrianHazard;
    public SimulatedActorObservation rightLaneFront;
    public SimulatedActorObservation rightLaneRear;
    public SimulatedActorObservation leftLaneFront;
    public SimulatedActorObservation leftLaneRear;

    public bool IsLeftLaneClear(float requiredFrontGap, float requiredRearGap)
    {
        return IsLaneClear(leftLaneFront, leftLaneRear, requiredFrontGap, requiredRearGap);
    }

    public bool IsRightLaneClear(float requiredFrontGap, float requiredRearGap)
    {
        return IsLaneClear(rightLaneFront, rightLaneRear, requiredFrontGap, requiredRearGap);
    }

    private static bool IsLaneClear(
        SimulatedActorObservation front,
        SimulatedActorObservation rear,
        float requiredFrontGap,
        float requiredRearGap)
    {
        bool frontClear = front == null ||
            (front.gapDistance >= requiredFrontGap && front.timeToCollision >= 4f);
        bool rearClear = rear == null ||
            (rear.gapDistance >= requiredRearGap && rear.timeToCollision >= 4.5f);
        return frontClear && rearClear;
    }
}

public class SimulatedVehicleSensorSuite : MonoBehaviour
{
    [Header("Sensor Coverage")]
    [SerializeField, Min(10f)] private float forwardRange = 45f;
    [SerializeField, Min(5f)] private float rearRange = 15f;
    [SerializeField, Min(3f)] private float surroundingRadius = 48f;
    [Tooltip("Normally disabled so decorative city colliders are not mistaken for road traffic.")]
    [SerializeField] private bool detectUnregisteredEnvironment;

    [Header("Lane Model")]
    [SerializeField, Min(1f)] private float laneHalfWidth = 2.25f;
    [SerializeField] private float rightLaneCenterOffset;
    [SerializeField] private float leftLaneCenterOffset = -5.5f;
    [SerializeField] private float roadLeftEdgeOffset = -8.75f;
    [SerializeField] private float roadRightEdgeOffset = 3.25f;
    [SerializeField, Min(0.5f)] private float pedestrianPathHalfWidth = 1.8f;

    [Header("Vehicle Envelope")]
    [SerializeField, Min(0.5f)] private float egoHalfLength = 2.1f;

    private readonly HashSet<Collider> processedColliders = new HashSet<Collider>();
    private float rangeFactor = 1f;

    public float EffectiveForwardRange => forwardRange * rangeFactor;
    public float EffectiveRearRange => rearRange * rangeFactor;

    public void SetRangeFactor(float factor)
    {
        rangeFactor = Mathf.Clamp(factor, 0.1f, 1.5f);
    }

    public SimulatedSensorSnapshot Scan(
        Vector3 sensorPosition,
        Vector3 routeForward,
        Vector3 routeRight,
        float egoLaneOffset,
        Transform egoRoot)
    {
        return Scan(sensorPosition, routeForward, routeRight, egoLaneOffset, egoRoot, Vector3.zero);
    }

    public SimulatedSensorSnapshot Scan(
        Vector3 sensorPosition,
        Vector3 routeForward,
        Vector3 routeRight,
        float egoLaneOffset,
        Transform egoRoot,
        Vector3 egoVelocity)
    {
        SimulatedSensorSnapshot snapshot = new SimulatedSensorSnapshot();
        float activeForwardRange = EffectiveForwardRange;
        float activeRearRange = EffectiveRearRange;

        foreach (ScenarioActorMarker marker in ScenarioActorMarker.ActiveActors)
        {
            if (marker == null || !marker.gameObject.activeInHierarchy || marker.transform.IsChildOf(egoRoot))
                continue;

            Collider representativeCollider;
            Bounds actorBounds;
            bool hasBounds = TryGetCombinedColliderBounds(marker.gameObject, out actorBounds, out representativeCollider);
            Vector3 actorPosition = hasBounds ? actorBounds.center : marker.transform.position;
            Observe(snapshot, actorPosition, marker, representativeCollider, actorBounds, hasBounds,
                marker.Velocity, sensorPosition, routeForward, routeRight, egoLaneOffset, egoVelocity,
                activeForwardRange, activeRearRange);
        }

        if (detectUnregisteredEnvironment)
        {
            processedColliders.Clear();
            Collider[] nearby = Physics.OverlapSphere(
                sensorPosition,
                surroundingRadius * rangeFactor,
                ~0,
                QueryTriggerInteraction.Ignore
            );
            foreach (Collider candidate in nearby)
            {
                if (candidate == null || processedColliders.Contains(candidate))
                    continue;
                processedColliders.Add(candidate);
                if (candidate.transform == egoRoot || candidate.transform.IsChildOf(egoRoot))
                    continue;
                if (candidate.GetComponentInParent<RoadSurfaceMarker>() != null)
                    continue;
                if (candidate.GetComponentInParent<ScenarioActorMarker>() != null)
                    continue;

                Observe(snapshot, candidate.bounds.center, null, candidate, candidate.bounds, true,
                    Vector3.zero, sensorPosition, routeForward, routeRight, egoLaneOffset, egoVelocity,
                    activeForwardRange, activeRearRange);
            }
        }

        return snapshot;
    }

    private void Observe(
        SimulatedSensorSnapshot snapshot,
        Vector3 actorPosition,
        ScenarioActorMarker marker,
        Collider actorCollider,
        Bounds actorBounds,
        bool hasBounds,
        Vector3 actorVelocity,
        Vector3 sensorPosition,
        Vector3 routeForward,
        Vector3 routeRight,
        float egoLaneOffset,
        Vector3 egoVelocity,
        float activeForwardRange,
        float activeRearRange)
    {
        Vector3 difference = actorPosition - sensorPosition;
        difference.y = 0f;
        float longitudinal = Vector3.Dot(difference, routeForward);
        if (longitudinal > activeForwardRange || longitudinal < -activeRearRange)
            return;

        float absoluteLaneOffset = egoLaneOffset + Vector3.Dot(difference, routeRight);
        float extent = hasBounds ? ProjectBoundsExtent(actorBounds.extents, routeForward) : 0.8f;
        float gap = Mathf.Max(0f, Mathf.Abs(longitudinal) - extent - egoHalfLength);
        float egoLongitudinalSpeed = Vector3.Dot(egoVelocity, routeForward);
        float actorLongitudinalSpeed = Vector3.Dot(actorVelocity, routeForward);
        float closingSpeed = longitudinal >= 0f
            ? egoLongitudinalSpeed - actorLongitudinalSpeed
            : actorLongitudinalSpeed - egoLongitudinalSpeed;
        float ttc = closingSpeed > 0.05f ? gap / closingSpeed : float.PositiveInfinity;

        SimulatedActorObservation observation = new SimulatedActorObservation
        {
            actor = marker,
            collider = actorCollider,
            combinedBounds = actorBounds,
            longitudinalDistance = longitudinal,
            laneOffset = absoluteLaneOffset,
            gapDistance = gap,
            actorLongitudinalSpeed = actorLongitudinalSpeed,
            relativeClosingSpeed = closingSpeed,
            timeToCollision = ttc
        };

        if (marker != null && marker.SemanticClass == ScenarioActorClass.Pedestrian)
        {
            bool movingCrossingRisk = marker.HasCrossingMotion && !marker.CrossingComplete;
            bool datasetPedestrianOnRoad = !marker.HasCrossingMotion
                && absoluteLaneOffset >= roadLeftEdgeOffset
                && absoluteLaneOffset <= roadRightEdgeOffset
                && Mathf.Abs(absoluteLaneOffset - egoLaneOffset) <= pedestrianPathHalfWidth;
            if (longitudinal >= -1.5f && (movingCrossingRisk || datasetPedestrianOnRoad))
                StoreClosestFront(ref snapshot.pedestrianHazard, observation);
            return;
        }

        if (Mathf.Abs(absoluteLaneOffset - rightLaneCenterOffset) <= laneHalfWidth)
        {
            if (longitudinal >= 0f)
                StoreClosestFront(ref snapshot.rightLaneFront, observation);
            else
                StoreClosestRear(ref snapshot.rightLaneRear, observation);
            return;
        }

        if (Mathf.Abs(absoluteLaneOffset - leftLaneCenterOffset) <= laneHalfWidth)
        {
            if (longitudinal >= 0f)
                StoreClosestFront(ref snapshot.leftLaneFront, observation);
            else
                StoreClosestRear(ref snapshot.leftLaneRear, observation);
        }
    }

    public static bool TryGetCombinedColliderBounds(
        GameObject target,
        out Bounds combinedBounds,
        out Collider representativeCollider)
    {
        combinedBounds = default;
        representativeCollider = null;
        if (target == null)
            return false;

        Collider[] colliders = target.GetComponentsInChildren<Collider>(false);
        bool found = false;
        foreach (Collider actorCollider in colliders)
        {
            if (actorCollider == null || !actorCollider.enabled || actorCollider.isTrigger)
                continue;
            if (!found)
            {
                combinedBounds = actorCollider.bounds;
                representativeCollider = actorCollider;
                found = true;
            }
            else
            {
                combinedBounds.Encapsulate(actorCollider.bounds);
            }
        }
        return found;
    }

    private static float ProjectBoundsExtent(Vector3 worldExtents, Vector3 direction)
    {
        return Mathf.Abs(direction.x) * worldExtents.x
            + Mathf.Abs(direction.y) * worldExtents.y
            + Mathf.Abs(direction.z) * worldExtents.z;
    }

    private static void StoreClosestFront(ref SimulatedActorObservation current, SimulatedActorObservation candidate)
    {
        if (current == null || candidate.gapDistance < current.gapDistance)
            current = candidate;
    }

    private static void StoreClosestRear(ref SimulatedActorObservation current, SimulatedActorObservation candidate)
    {
        if (current == null || candidate.gapDistance < current.gapDistance)
            current = candidate;
    }
}

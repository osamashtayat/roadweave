using System.Collections.Generic;
using UnityEngine;

public sealed class SimulatedActorObservation
{
    public ScenarioActorMarker actor;
    public Collider collider;
    public float longitudinalDistance;
    public float laneOffset;
    public float gapDistance;

    public bool IsValid => actor != null || collider != null;
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
        bool frontClear = leftLaneFront == null || leftLaneFront.gapDistance >= requiredFrontGap;
        bool rearClear = leftLaneRear == null || leftLaneRear.gapDistance >= requiredRearGap;
        return frontClear && rearClear;
    }

    public bool IsRightLaneClear(float requiredFrontGap, float requiredRearGap)
    {
        bool frontClear = rightLaneFront == null || rightLaneFront.gapDistance >= requiredFrontGap;
        bool rearClear = rightLaneRear == null || rightLaneRear.gapDistance >= requiredRearGap;
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

    public SimulatedSensorSnapshot Scan(
        Vector3 sensorPosition,
        Vector3 routeForward,
        Vector3 routeRight,
        float egoLaneOffset,
        Transform egoRoot)
    {
        SimulatedSensorSnapshot snapshot = new SimulatedSensorSnapshot();

        foreach (ScenarioActorMarker marker in ScenarioActorMarker.ActiveActors)
        {
            if (marker == null || !marker.gameObject.activeInHierarchy)
                continue;
            Collider actorCollider = marker.GetComponentInChildren<Collider>();
            Observe(
                snapshot,
                marker.transform.position,
                marker,
                actorCollider,
                sensorPosition,
                routeForward,
                routeRight,
                egoLaneOffset
            );
        }

        if (detectUnregisteredEnvironment)
        {
            processedColliders.Clear();
            Collider[] nearby = Physics.OverlapSphere(
                sensorPosition,
                surroundingRadius,
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

                Observe(
                    snapshot,
                    candidate.bounds.center,
                    null,
                    candidate,
                    sensorPosition,
                    routeForward,
                    routeRight,
                    egoLaneOffset
                );
            }
        }

        return snapshot;
    }

    private void Observe(
        SimulatedSensorSnapshot snapshot,
        Vector3 actorPosition,
        ScenarioActorMarker marker,
        Collider actorCollider,
        Vector3 sensorPosition,
        Vector3 routeForward,
        Vector3 routeRight,
        float egoLaneOffset)
    {
        Vector3 difference = actorPosition - sensorPosition;
        difference.y = 0f;
        float longitudinal = Vector3.Dot(difference, routeForward);
        if (longitudinal > forwardRange || longitudinal < -rearRange)
            return;

        float absoluteLaneOffset = egoLaneOffset + Vector3.Dot(difference, routeRight);
        float extent = actorCollider != null
            ? ProjectBoundsExtent(actorCollider.bounds.extents, routeForward)
            : 0.8f;
        float gap = Mathf.Max(0f, Mathf.Abs(longitudinal) - extent - egoHalfLength);

        SimulatedActorObservation observation = new SimulatedActorObservation
        {
            actor = marker,
            collider = actorCollider,
            longitudinalDistance = longitudinal,
            laneOffset = absoluteLaneOffset,
            gapDistance = gap
        };

        if (marker != null && marker.ActorType == ScenarioActorType.Pedestrian)
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

    private static float ProjectBoundsExtent(Vector3 worldExtents, Vector3 direction)
    {
        return Mathf.Abs(direction.x) * worldExtents.x
            + Mathf.Abs(direction.y) * worldExtents.y
            + Mathf.Abs(direction.z) * worldExtents.z;
    }

    private static void StoreClosestFront(
        ref SimulatedActorObservation current,
        SimulatedActorObservation candidate)
    {
        if (current == null || candidate.gapDistance < current.gapDistance)
            current = candidate;
    }

    private static void StoreClosestRear(
        ref SimulatedActorObservation current,
        SimulatedActorObservation candidate)
    {
        if (current == null || candidate.gapDistance < current.gapDistance)
            current = candidate;
    }
}

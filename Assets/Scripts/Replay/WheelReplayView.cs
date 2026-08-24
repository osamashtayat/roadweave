using UnityEngine;

public class WheelReplayView : MonoBehaviour
{
    [SerializeField] private DigitalTwinStateManager stateManager;

    [Header("Wheel Mesh Transforms")]
    [SerializeField] private Transform frontLeftWheel;
    [SerializeField] private Transform frontRightWheel;
    [SerializeField] private Transform rearLeftWheel;
    [SerializeField] private Transform rearRightWheel;

    [Header("Rotation Setup")]
    [Tooltip("Most car models use X. Change this in the Inspector if your wheels spin around another axis.")]
    [SerializeField] private Vector3 localRotationAxis = Vector3.right;
    [SerializeField] private float frontLeftDirection = 1f;
    [SerializeField] private float frontRightDirection = 1f;
    [SerializeField] private float rearLeftDirection = 1f;
    [SerializeField] private float rearRightDirection = 1f;

    private Quaternion frontLeftStart;
    private Quaternion frontRightStart;
    private Quaternion rearLeftStart;
    private Quaternion rearRightStart;
    private float frontLeftAngle;
    private float frontRightAngle;
    private float rearLeftAngle;
    private float rearRightAngle;
    private double previousSourceTime;
    private string previousSessionId;

    private void Awake()
    {
        if (stateManager == null)
            stateManager = FindAnyObjectByType<DigitalTwinStateManager>();

        if (frontLeftWheel != null) frontLeftStart = frontLeftWheel.localRotation;
        if (frontRightWheel != null) frontRightStart = frontRightWheel.localRotation;
        if (rearLeftWheel != null) rearLeftStart = rearLeftWheel.localRotation;
        if (rearRightWheel != null) rearRightStart = rearRightWheel.localRotation;
    }

    private void LateUpdate()
    {
        if (stateManager == null || !stateManager.IsReady || !stateManager.IsFresh ||
            (stateManager.Wheels.validity != TwinDataValidity.Valid &&
             stateManager.Wheels.validity != TwinDataValidity.Partial))
            return;

        string sessionId = stateManager.Session?.sessionId;
        double sourceTime = stateManager.CurrentSourceTimestampSeconds;
        if (sessionId != previousSessionId || sourceTime < previousSourceTime)
            ResetAngles(sourceTime);

        if (stateManager.IsPlaying)
        {
            // Subtract in double precision before converting the small delta to
            // float. Casting Unix timestamps to float first loses sub-second time.
            float seconds = (float)System.Math.Max(0d, sourceTime - previousSourceTime);
            frontLeftAngle += RpmToDegrees(stateManager.Wheels.frontLeftRpm, seconds) * frontLeftDirection;
            frontRightAngle += RpmToDegrees(stateManager.Wheels.frontRightRpm, seconds) * frontRightDirection;
            rearLeftAngle += RpmToDegrees(stateManager.Wheels.rearLeftRpm, seconds) * rearLeftDirection;
            rearRightAngle += RpmToDegrees(stateManager.Wheels.rearRightRpm, seconds) * rearRightDirection;
        }

        ApplyWheel(frontLeftWheel, frontLeftStart, frontLeftAngle);
        ApplyWheel(frontRightWheel, frontRightStart, frontRightAngle);
        ApplyWheel(rearLeftWheel, rearLeftStart, rearLeftAngle);
        ApplyWheel(rearRightWheel, rearRightStart, rearRightAngle);
        previousSourceTime = sourceTime;
        previousSessionId = sessionId;
    }

    private static float RpmToDegrees(float rpm, float seconds)
    {
        return rpm * 6f * seconds;
    }

    private void ApplyWheel(Transform wheel, Quaternion startingRotation, float angle)
    {
        if (wheel == null)
            return;

        // Some imported models place the mesh away from the wheel transform's
        // pivot. Rotating that transform makes the wheel orbit out of the body.
        // A zero axis intentionally keeps those wheels securely attached.
        if (localRotationAxis.sqrMagnitude <= 0.0001f)
        {
            wheel.localRotation = startingRotation;
            return;
        }
        wheel.localRotation = startingRotation * Quaternion.AngleAxis(angle, localRotationAxis.normalized);
    }

    private void ResetAngles(double sourceTime)
    {
        frontLeftAngle = 0f;
        frontRightAngle = 0f;
        rearLeftAngle = 0f;
        rearRightAngle = 0f;
        previousSourceTime = sourceTime;
    }
}

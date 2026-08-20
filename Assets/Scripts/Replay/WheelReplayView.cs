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
    private float previousReplayTime;

    private void Awake()
    {
        if (stateManager == null)
            stateManager = FindFirstObjectByType<DigitalTwinStateManager>();

        if (frontLeftWheel != null) frontLeftStart = frontLeftWheel.localRotation;
        if (frontRightWheel != null) frontRightStart = frontRightWheel.localRotation;
        if (rearLeftWheel != null) rearLeftStart = rearLeftWheel.localRotation;
        if (rearRightWheel != null) rearRightStart = rearRightWheel.localRotation;
    }

    private void LateUpdate()
    {
        if (stateManager == null || !stateManager.IsReady)
            return;

        if (stateManager.CurrentTime < previousReplayTime)
            ResetAngles();

        if (stateManager.IsPlaying)
        {
            float seconds = Mathf.Max(0f, stateManager.CurrentTime - previousReplayTime);
            frontLeftAngle += RpmToDegrees(stateManager.Wheels.frontLeftRpm, seconds) * frontLeftDirection;
            frontRightAngle += RpmToDegrees(stateManager.Wheels.frontRightRpm, seconds) * frontRightDirection;
            rearLeftAngle += RpmToDegrees(stateManager.Wheels.rearLeftRpm, seconds) * rearLeftDirection;
            rearRightAngle += RpmToDegrees(stateManager.Wheels.rearRightRpm, seconds) * rearRightDirection;
        }

        ApplyWheel(frontLeftWheel, frontLeftStart, frontLeftAngle);
        ApplyWheel(frontRightWheel, frontRightStart, frontRightAngle);
        ApplyWheel(rearLeftWheel, rearLeftStart, rearLeftAngle);
        ApplyWheel(rearRightWheel, rearRightStart, rearRightAngle);
        previousReplayTime = stateManager.CurrentTime;
    }

    private static float RpmToDegrees(float rpm, float seconds)
    {
        return rpm * 6f * seconds;
    }

    private void ApplyWheel(Transform wheel, Quaternion startingRotation, float angle)
    {
        if (wheel == null)
            return;
        wheel.localRotation = startingRotation * Quaternion.AngleAxis(angle, localRotationAxis.normalized);
    }

    private void ResetAngles()
    {
        frontLeftAngle = 0f;
        frontRightAngle = 0f;
        rearLeftAngle = 0f;
        rearRightAngle = 0f;
        previousReplayTime = 0f;
    }
}

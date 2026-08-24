using UnityEngine;

[DefaultExecutionOrder(100)]
public class VehicleFollowCamera : MonoBehaviour
{
    [SerializeField] private Transform target;
    [SerializeField] private Vector3 targetOffset = new Vector3(0f, 1.4f, 0f);

    [Header("Follow")]
    [SerializeField, Min(0.1f)] private float distance = 8f;
    [SerializeField, Min(0f)] private float followSmoothness = 10f;
    [SerializeField] private bool rotateWithVehicle = true;

    [Header("Mouse Control")]
    [SerializeField] private float yawSensitivity = 4f;
    [SerializeField] private float pitchSensitivity = 3f;
    [SerializeField] private float zoomSensitivity = 2f;
    [SerializeField] private float minimumDistance = 3f;
    [SerializeField] private float maximumDistance = 20f;
    [SerializeField] private float minimumPitch = 5f;
    [SerializeField] private float maximumPitch = 75f;

    private float yawOffset;
    private float pitch = 20f;
    private Camera controlledCamera;

    private void Awake()
    {
        controlledCamera = GetComponent<Camera>();
        if (controlledCamera == null)
        {
            Debug.LogWarning("VehicleFollowCamera must be placed on the Camera it controls.", this);
            enabled = false;
            return;
        }

        OrbitCamera legacyOrbit = GetComponent<OrbitCamera>();
        if (legacyOrbit != null && legacyOrbit.enabled)
        {
            legacyOrbit.enabled = false;
            Debug.LogWarning("OrbitCamera was disabled because VehicleFollowCamera owns this Camera.", this);
        }
    }

    private void Start()
    {
        if (target == null)
        {
            EgoVehicleReplayView vehicle = FindFirstObjectByType<EgoVehicleReplayView>();
            if (vehicle != null)
                target = vehicle.transform;
        }

        if (target == null)
        {
            Debug.LogWarning("VehicleFollowCamera has no target. Drag EgoVehicleRoot into its Target field.");
            enabled = false;
            return;
        }

        Vector3 direction = transform.position - (target.position + targetOffset);
        if (direction.sqrMagnitude > 0.01f)
        {
            distance = Mathf.Clamp(direction.magnitude, minimumDistance, maximumDistance);
            pitch = Mathf.Clamp(Mathf.Asin(direction.normalized.y) * Mathf.Rad2Deg, minimumPitch, maximumPitch);
            float worldYaw = Mathf.Atan2(-direction.x, -direction.z) * Mathf.Rad2Deg;
            yawOffset = rotateWithVehicle ? Mathf.DeltaAngle(target.eulerAngles.y, worldYaw) : worldYaw;
        }
    }

    private void LateUpdate()
    {
        if (target == null)
            return;

        if (Input.GetMouseButton(1))
        {
            yawOffset += Input.GetAxis("Mouse X") * yawSensitivity;
            pitch -= Input.GetAxis("Mouse Y") * pitchSensitivity;
            pitch = Mathf.Clamp(pitch, minimumPitch, maximumPitch);
        }

        float scroll = Input.mouseScrollDelta.y;
        if (Mathf.Abs(scroll) > 0.001f)
            distance = Mathf.Clamp(distance - scroll * zoomSensitivity, minimumDistance, maximumDistance);

        float baseYaw = rotateWithVehicle ? target.eulerAngles.y : 0f;
        Quaternion orbitRotation = Quaternion.Euler(pitch, baseYaw + yawOffset, 0f);
        Vector3 focusPoint = target.position + targetOffset;
        Vector3 desiredPosition = focusPoint - orbitRotation * Vector3.forward * distance;
        float blend = 1f - Mathf.Exp(-followSmoothness * Time.deltaTime);

        transform.position = Vector3.Lerp(transform.position, desiredPosition, blend);
        transform.rotation = Quaternion.LookRotation(focusPoint - transform.position, Vector3.up);
    }

    public void SetTarget(Transform newTarget)
    {
        target = newTarget;
        enabled = target != null && controlledCamera != null;
    }
}

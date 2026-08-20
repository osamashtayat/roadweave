using UnityEngine;

public class EgoVehicleReplayView : MonoBehaviour
{
    [SerializeField] private DigitalTwinStateManager stateManager;
    [Tooltip("Usually this is EgoVehicleRoot. Leave empty to move this GameObject.")]
    [SerializeField] private Transform vehicleRoot;
    [SerializeField] private Vector3 localPositionOffset;
    [SerializeField] private float modelYawOffset;
    [SerializeField] private bool applyPosition = true;
    [SerializeField] private bool applyRotation = true;

    private void Awake()
    {
        if (vehicleRoot == null)
            vehicleRoot = transform;

        if (stateManager == null)
            stateManager = FindFirstObjectByType<DigitalTwinStateManager>();
    }

    private void LateUpdate()
    {
        if (stateManager == null || !stateManager.IsReady)
            return;

        if (applyPosition)
            vehicleRoot.localPosition = stateManager.Ego.position + localPositionOffset;

        if (applyRotation)
            vehicleRoot.localRotation = Quaternion.Euler(0f, stateManager.Ego.yawDegrees + modelYawOffset, 0f);
    }
}

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

    [Header("Streaming Presentation")]
    [Tooltip("Replay already publishes interpolated poses. This smooths only simulated/live streams.")]
    [SerializeField] private bool smoothStreamingSources = true;
    [SerializeField, Min(0f)] private float positionSmoothness = 18f;
    [SerializeField, Min(0f)] private float rotationSmoothness = 14f;

    private Renderer[] presentationRenderers;
    private bool[] rendererEnabledDefaults;
    private bool presentationAvailable = true;
    private bool streamingPoseInitialized;
    private string presentedSessionId;

    private void Awake()
    {
        if (vehicleRoot == null)
            vehicleRoot = transform;

        if (stateManager == null)
            stateManager = FindAnyObjectByType<DigitalTwinStateManager>();

        presentationRenderers = vehicleRoot.GetComponentsInChildren<Renderer>(true);
        rendererEnabledDefaults = new bool[presentationRenderers.Length];
        for (int index = 0; index < presentationRenderers.Length; index++)
            rendererEnabledDefaults[index] = presentationRenderers[index] != null && presentationRenderers[index].enabled;
    }

    private void LateUpdate()
    {
        bool available = stateManager != null && stateManager.IsReady && stateManager.IsFresh &&
                         (stateManager.Ego.validity == TwinDataValidity.Valid ||
                          stateManager.Ego.validity == TwinDataValidity.Partial);
        SetPresentationAvailable(available);
        if (!available)
        {
            streamingPoseInitialized = false;
            return;
        }

        TwinSnapshotMetadata metadata = stateManager.Metadata;
        bool streaming = smoothStreamingSources &&
                         metadata?.session != null &&
                         metadata.session.sourceKind != TwinSourceKind.Replay;

        Vector3 targetPosition = stateManager.Ego.position + localPositionOffset;
        Quaternion targetRotation = Quaternion.Euler(
            0f,
            stateManager.Ego.yawDegrees + modelYawOffset,
            0f);

        string sessionId = metadata?.session?.sessionId;
        bool newSession = !string.Equals(presentedSessionId, sessionId, System.StringComparison.Ordinal);
        if (!streaming || !streamingPoseInitialized || newSession)
        {
            if (applyPosition)
                vehicleRoot.localPosition = targetPosition;
            if (applyRotation)
                vehicleRoot.localRotation = targetRotation;
            streamingPoseInitialized = streaming;
            presentedSessionId = sessionId;
            return;
        }

        if (applyPosition)
        {
            float blend = ExponentialBlend(positionSmoothness, Time.deltaTime);
            vehicleRoot.localPosition = Vector3.Lerp(vehicleRoot.localPosition, targetPosition, blend);
        }

        if (applyRotation)
        {
            float blend = ExponentialBlend(rotationSmoothness, Time.deltaTime);
            vehicleRoot.localRotation = Quaternion.Slerp(vehicleRoot.localRotation, targetRotation, blend);
        }
    }

    private void SetPresentationAvailable(bool available)
    {
        if (presentationAvailable == available)
            return;
        presentationAvailable = available;
        for (int index = 0; index < presentationRenderers.Length; index++)
        {
            Renderer renderer = presentationRenderers[index];
            if (renderer != null)
                renderer.enabled = available && rendererEnabledDefaults[index];
        }
    }

    private static float ExponentialBlend(float smoothness, float deltaTime)
    {
        if (smoothness <= 0f)
            return 1f;
        return 1f - Mathf.Exp(-smoothness * Mathf.Max(0f, deltaTime));
    }
}

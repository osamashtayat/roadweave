using System;
using System.Collections.Generic;
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
    [Tooltip("Replay already publishes interpolated poses. Buffer only simulated/live streams.")]
    [SerializeField] private bool smoothStreamingSources = true;
    [Tooltip("Presentation delay used to interpolate between two received poses instead of chasing UDP steps.")]
    [SerializeField, Min(0.03f)] private float streamingInterpolationDelaySeconds = 0.16f;
    [SerializeField, Range(4, 64)] private int maximumBufferedPoses = 24;

    private struct StreamingPoseSample
    {
        public long sequenceNumber;
        public double sourceTimestampSeconds;
        public double receiptTimestampSeconds;
        public Vector3 localPosition;
        public Quaternion localRotation;
    }

    private Renderer[] presentationRenderers;
    private bool[] rendererEnabledDefaults;
    private readonly List<StreamingPoseSample> streamingPoseBuffer =
        new List<StreamingPoseSample>(16);
    private bool presentationAvailable = true;
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

    private void OnEnable()
    {
        if (stateManager == null)
            stateManager = FindAnyObjectByType<DigitalTwinStateManager>();
        if (stateManager != null)
            stateManager.SnapshotUpdated += BufferStreamingSnapshot;
    }

    private void Start()
    {
        if (stateManager?.CurrentSnapshot != null)
            BufferStreamingSnapshot(stateManager.CurrentSnapshot);
    }

    private void OnDisable()
    {
        if (stateManager != null)
            stateManager.SnapshotUpdated -= BufferStreamingSnapshot;
        ResetStreamingPresentation();
    }

    private void LateUpdate()
    {
        bool available = stateManager != null && stateManager.IsReady && stateManager.IsFresh &&
                         (stateManager.Ego.validity == TwinDataValidity.Valid ||
                          stateManager.Ego.validity == TwinDataValidity.Partial);
        SetPresentationAvailable(available);
        if (!available)
        {
            ResetStreamingPresentation();
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

        string sessionId = metadata?.session?.sessionId ?? string.Empty;
        bool newSession = !string.Equals(presentedSessionId, sessionId, StringComparison.Ordinal);
        if (!streaming)
        {
            if (applyPosition)
                vehicleRoot.localPosition = targetPosition;
            if (applyRotation)
                vehicleRoot.localRotation = targetRotation;
            presentedSessionId = sessionId;
            return;
        }

        if (newSession)
            presentedSessionId = sessionId;

        if (!TrySampleBufferedPose(
                Time.realtimeSinceStartupAsDouble,
                out Vector3 presentedPosition,
                out Quaternion presentedRotation))
        {
            presentedPosition = targetPosition;
            presentedRotation = targetRotation;
        }

        if (applyPosition)
            vehicleRoot.localPosition = presentedPosition;
        if (applyRotation)
            vehicleRoot.localRotation = presentedRotation;
        presentedSessionId = sessionId;
    }

    private void BufferStreamingSnapshot(TwinSnapshot snapshot)
    {
        TwinSnapshotMetadata metadata = snapshot?.metadata;
        TwinSessionInfo session = metadata?.session;
        TwinEgoState ego = snapshot?.ego;
        if (metadata == null || session == null || ego == null ||
            session.sourceKind == TwinSourceKind.Replay ||
            metadata.freshness == TwinDataFreshness.Stale ||
            (ego.validity != TwinDataValidity.Valid && ego.validity != TwinDataValidity.Partial))
            return;

        string sessionId = session.sessionId ?? string.Empty;
        if (!string.Equals(presentedSessionId, sessionId, StringComparison.Ordinal))
        {
            streamingPoseBuffer.Clear();
            presentedSessionId = sessionId;
        }

        StreamingPoseSample sample = new StreamingPoseSample
        {
            sequenceNumber = metadata.sequenceNumber,
            sourceTimestampSeconds = metadata.sourceTimestampSeconds,
            receiptTimestampSeconds = metadata.receiptTimestampSeconds,
            localPosition = ego.position + localPositionOffset,
            localRotation = Quaternion.Euler(0f, ego.yawDegrees + modelYawOffset, 0f)
        };

        if (streamingPoseBuffer.Count > 0)
        {
            StreamingPoseSample last = streamingPoseBuffer[streamingPoseBuffer.Count - 1];
            if (sample.sourceTimestampSeconds < last.sourceTimestampSeconds - 0.000001d)
            {
                // A restarted stream may reset both its source clock and its
                // sequence number while retaining the same session id.
                streamingPoseBuffer.Clear();
            }
            else if (sample.sequenceNumber <= last.sequenceNumber)
            {
                return;
            }
            else if (Math.Abs(sample.sourceTimestampSeconds - last.sourceTimestampSeconds) <= 0.000001d)
            {
                // Paused streaming sources send heartbeat snapshots with the
                // same source time. Replace the heartbeat instead of filling
                // the interpolation buffer with duplicate poses.
                streamingPoseBuffer[streamingPoseBuffer.Count - 1] = sample;
                return;
            }
        }

        streamingPoseBuffer.Add(sample);
        int maximum = Mathf.Clamp(maximumBufferedPoses, 4, 64);
        while (streamingPoseBuffer.Count > maximum)
            streamingPoseBuffer.RemoveAt(0);
    }

    private bool TrySampleBufferedPose(
        double realtimeNow,
        out Vector3 localPosition,
        out Quaternion localRotation)
    {
        if (streamingPoseBuffer.Count == 0)
        {
            localPosition = default;
            localRotation = Quaternion.identity;
            return false;
        }

        StreamingPoseSample newest = streamingPoseBuffer[streamingPoseBuffer.Count - 1];
        bool sourceIsRunning = stateManager?.Session?.status == TwinSessionStatus.Running;
        double estimatedSourceNow = sourceIsRunning
            ? newest.sourceTimestampSeconds + Math.Max(0d, realtimeNow - newest.receiptTimestampSeconds)
            : newest.sourceTimestampSeconds;
        double renderSourceTime = estimatedSourceNow -
                                  Math.Max(0.03d, streamingInterpolationDelaySeconds);

        while (streamingPoseBuffer.Count > 2 &&
               streamingPoseBuffer[1].sourceTimestampSeconds <= renderSourceTime)
            streamingPoseBuffer.RemoveAt(0);

        StreamingPoseSample first = streamingPoseBuffer[0];
        if (streamingPoseBuffer.Count == 1 || renderSourceTime <= first.sourceTimestampSeconds)
        {
            localPosition = first.localPosition;
            localRotation = first.localRotation;
            return true;
        }

        StreamingPoseSample second = streamingPoseBuffer[1];
        float interpolation = SourceInterpolationFactor(
            renderSourceTime,
            first.sourceTimestampSeconds,
            second.sourceTimestampSeconds);
        localPosition = Vector3.LerpUnclamped(first.localPosition, second.localPosition, interpolation);
        localRotation = Quaternion.SlerpUnclamped(first.localRotation, second.localRotation, interpolation);
        return true;
    }

    private static float SourceInterpolationFactor(double time, double from, double to)
    {
        double duration = to - from;
        if (duration <= 0.000001d)
            return 1f;
        return Mathf.Clamp01((float)((time - from) / duration));
    }

    private void ResetStreamingPresentation()
    {
        streamingPoseBuffer.Clear();
        presentedSessionId = null;
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

}

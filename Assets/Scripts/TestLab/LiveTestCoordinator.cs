using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class LiveTestCoordinator : MonoBehaviour
{
    [Header("Live Twin")]
    [SerializeField] private DigitalTwinStateManager stateManager;
    [SerializeField] private NuScenesReplayController replayController;
    [SerializeField] private ReplayRoadGenerator roadGenerator;
    [SerializeField] private Transform replayRoot;
    [SerializeField] private Transform liveVehicleRoot;

    [Header("Test Lab")]
    [SerializeField] private Transform testWorldRoot;
    [Tooltip("Optional. If empty, the live vehicle hierarchy is cloned.")]
    [SerializeField] private GameObject testVehiclePrefab;
    [SerializeField] private float testVehicleModelYawOffset;
    [SerializeField, Min(0.25f)] private float routePointSpacing = 1f;
    [SerializeField, Min(25f)] private float testRouteContinuationLength = 250f;
    [SerializeField, Min(0.5f)] private float testRouteContinuationSpacing = 2f;
    [SerializeField] private TestScenarioManager scenarioManager;
    [SerializeField] private TestWeatherController weatherController;
    [SerializeField] private TestMetricsRecorder metricsRecorder;

    [Header("Camera and UI")]
    [SerializeField] private VehicleFollowCamera followCamera;
    [SerializeField] private GameObject liveTwinPanel;
    [SerializeField] private GameObject testLabPanel;
    [SerializeField] private TMP_Text modeStatusText;
    [SerializeField] private bool resumeReplayWhenReturning = true;

    public AutonomousTestVehicleController TestVehicle { get; private set; }
    public bool IsInTestLab { get; private set; }

    private float capturedReplayTime;
    private float capturedSpeedKph;
    private Vector3 capturedPosition;
    private Quaternion capturedRotation;

    private void Awake()
    {
        if (stateManager == null) stateManager = FindFirstObjectByType<DigitalTwinStateManager>();
        if (replayController == null) replayController = FindFirstObjectByType<NuScenesReplayController>();
        if (roadGenerator == null) roadGenerator = FindFirstObjectByType<ReplayRoadGenerator>();
        if (followCamera == null) followCamera = FindFirstObjectByType<VehicleFollowCamera>();

        if (replayRoot == null && liveVehicleRoot != null)
            replayRoot = liveVehicleRoot.parent;

        if (testWorldRoot == null)
        {
            GameObject root = new GameObject("TestWorldRoot");
            root.transform.SetParent(replayRoot != null ? replayRoot : transform, false);
            testWorldRoot = root.transform;
        }
    }

    private void Start()
    {
        ShowLiveTwin(false);
    }

    public void CreateTest()
    {
        if (stateManager == null || !stateManager.IsReady || liveVehicleRoot == null)
        {
            Debug.LogWarning("LiveTestCoordinator cannot create a test until the replay and live vehicle are ready.");
            return;
        }

        bool replayWasMoving = stateManager.IsPlaying;
        replayController?.PauseReplay();
        capturedReplayTime = stateManager.CurrentTime;
        // If Create Test is pressed before Drive, begin the autonomous test from
        // rest instead of inheriting a non-zero speed stored in the first frame.
        capturedSpeedKph = replayWasMoving
            ? stateManager.Vehicle.speedKilometersPerHour
            : 0f;
        capturedPosition = liveVehicleRoot.position;
        capturedRotation = liveVehicleRoot.rotation;
        BuildTestFromCapturedState();
    }

    public void RunTest()
    {
        if (TestVehicle == null)
        {
            Debug.LogWarning("Press Create Test before Run Test.");
            return;
        }

        if (metricsRecorder != null && !metricsRecorder.IsRecording)
        {
            string scenario = scenarioManager != null ? scenarioManager.CurrentScenarioName : "Clear Route";
            string weather = weatherController != null ? weatherController.CurrentWeather.ToString() : "Dry";
            metricsRecorder.BeginRecording(scenario, weather);
        }
        TestVehicle.SetRunning(true);
        UpdateModeLabel("TEST LAB — RUNNING");
    }

    public void PauseTest()
    {
        if (TestVehicle != null)
            TestVehicle.SetRunning(false);
        UpdateModeLabel("TEST LAB — PAUSED");
    }

    public void ResetTest()
    {
        if (!IsInTestLab)
        {
            Debug.LogWarning("Create a test before resetting it.");
            return;
        }
        BuildTestFromCapturedState();
    }

    public void ReturnToLiveTwin()
    {
        if (TestVehicle != null)
            TestVehicle.SetRunning(false);

        scenarioManager?.ClearScenarios();
        weatherController?.SetDry();
        weatherController?.SetTestVehicle(null);
        metricsRecorder?.StopRecording();
        ClearTestVehicle();

        if (testWorldRoot != null) testWorldRoot.gameObject.SetActive(false);
        if (liveVehicleRoot != null) liveVehicleRoot.gameObject.SetActive(true);
        if (roadGenerator != null) roadGenerator.ResumeReplayReveal();
        if (followCamera != null && liveVehicleRoot != null) followCamera.SetTarget(liveVehicleRoot);
        if (liveTwinPanel != null) liveTwinPanel.SetActive(true);
        if (testLabPanel != null) testLabPanel.SetActive(false);

        IsInTestLab = false;
        UpdateModeLabel("LIVE TWIN");
        if (resumeReplayWhenReturning)
            replayController?.StartDrive();
    }

    private void BuildTestFromCapturedState()
    {
        if (TestVehicle != null)
            TestVehicle.SetRunning(false);

        scenarioManager?.ClearScenarios();
        weatherController?.SetDry();
        metricsRecorder?.ResetResults();
        ClearTestVehicle();

        if (testWorldRoot != null)
            testWorldRoot.gameObject.SetActive(true);

        List<Vector3> route = BuildRemainingWorldRoute();
        Vector3 testStartPosition = route.Count > 0 ? route[0] : capturedPosition;
        GameObject source = testVehiclePrefab != null ? testVehiclePrefab : liveVehicleRoot.gameObject;
        GameObject testVehicleObject = Instantiate(
            source,
            testStartPosition,
            capturedRotation,
            testWorldRoot
        );
        testVehicleObject.name = "TestVehicle";

        foreach (EgoVehicleReplayView replayView in testVehicleObject.GetComponentsInChildren<EgoVehicleReplayView>(true))
            replayView.enabled = false;
        foreach (WheelReplayView wheelView in testVehicleObject.GetComponentsInChildren<WheelReplayView>(true))
            wheelView.enabled = false;

        // After the first test is created, the live vehicle is hidden. Unity copies
        // that inactive state when Reset clones the live vehicle again, so Reset
        // must explicitly activate the new test copy.
        testVehicleObject.SetActive(true);

        TestVehicle = testVehicleObject.GetComponent<AutonomousTestVehicleController>();
        if (TestVehicle == null)
            TestVehicle = testVehicleObject.AddComponent<AutonomousTestVehicleController>();
        TestVehicle.ConfigureModelForwardYawOffset(testVehicleModelYawOffset);

        TestVehicle.SetRoute(route, capturedSpeedKph);
        TestVehicle.SetRunning(false);

        if (liveVehicleRoot != null) liveVehicleRoot.gameObject.SetActive(false);
        if (roadGenerator != null) roadGenerator.ShowCompleteRoad();
        if (scenarioManager != null) scenarioManager.SetTestVehicle(TestVehicle);
        if (weatherController != null) weatherController.SetTestVehicle(TestVehicle);
        if (metricsRecorder != null) metricsRecorder.AttachVehicle(TestVehicle);
        if (followCamera != null) followCamera.SetTarget(TestVehicle.transform);
        if (liveTwinPanel != null) liveTwinPanel.SetActive(false);
        if (testLabPanel != null) testLabPanel.SetActive(true);

        IsInTestLab = true;
        UpdateModeLabel("TEST LAB — READY");
    }

    private List<Vector3> BuildRemainingWorldRoute()
    {
        List<Vector3> route = new List<Vector3> { capturedPosition };
        if (stateManager.Package == null || stateManager.Package.egoFrames == null)
            return route;

        Vector3 lastPoint = capturedPosition;
        foreach (EgoReplayFrame frame in stateManager.Package.egoFrames)
        {
            if (frame.time < capturedReplayTime)
                continue;

            Vector3 localPoint = frame.position.ToVector3();
            Vector3 worldPoint = replayRoot != null ? replayRoot.TransformPoint(localPoint) : localPoint;
            if (Vector3.Distance(lastPoint, worldPoint) < routePointSpacing)
                continue;

            route.Add(worldPoint);
            lastPoint = worldPoint;
        }

        EgoReplayFrame[] frames = stateManager.Package.egoFrames;
        if (frames.Length > 0)
        {
            Vector3 lastLocal = frames[frames.Length - 1].position.ToVector3();
            Vector3 finalPoint = replayRoot != null ? replayRoot.TransformPoint(lastLocal) : lastLocal;
            if (Vector3.Distance(lastPoint, finalPoint) > 0.1f)
                route.Add(finalPoint);
        }

        AppendTestRouteContinuation(route, frames);

        if (route.Count < 2)
            Debug.LogWarning("Create Test could not build a usable autonomous route.");

        return route;
    }

    private void AppendTestRouteContinuation(List<Vector3> route, EgoReplayFrame[] frames)
    {
        if (route == null || route.Count == 0 || testRouteContinuationLength <= 0f)
            return;

        Vector3 direction = Vector3.zero;
        if (route.Count >= 2)
        {
            direction = route[route.Count - 1] - route[route.Count - 2];
        }
        else if (frames != null && frames.Length >= 2)
        {
            Vector3 previousLocal = frames[frames.Length - 2].position.ToVector3();
            Vector3 finalLocal = frames[frames.Length - 1].position.ToVector3();
            Vector3 previousWorld = replayRoot != null ? replayRoot.TransformPoint(previousLocal) : previousLocal;
            Vector3 finalWorld = replayRoot != null ? replayRoot.TransformPoint(finalLocal) : finalLocal;
            direction = finalWorld - previousWorld;
        }

        direction.y = 0f;
        if (direction.sqrMagnitude <= 0.001f)
            return;
        direction.Normalize();

        Vector3 startingPoint = route[route.Count - 1];
        int stepCount = Mathf.CeilToInt(testRouteContinuationLength / testRouteContinuationSpacing);
        for (int step = 1; step <= stepCount; step++)
        {
            float distance = Mathf.Min(step * testRouteContinuationSpacing, testRouteContinuationLength);
            route.Add(startingPoint + direction * distance);
        }
    }

    private void ClearTestVehicle()
    {
        if (TestVehicle != null)
            Destroy(TestVehicle.gameObject);
        TestVehicle = null;
    }

    private void ShowLiveTwin(bool resumeReplay)
    {
        if (testWorldRoot != null) testWorldRoot.gameObject.SetActive(false);
        if (liveVehicleRoot != null) liveVehicleRoot.gameObject.SetActive(true);
        if (liveTwinPanel != null) liveTwinPanel.SetActive(true);
        if (testLabPanel != null) testLabPanel.SetActive(false);
        if (followCamera != null && liveVehicleRoot != null) followCamera.SetTarget(liveVehicleRoot);
        IsInTestLab = false;
        UpdateModeLabel("LIVE TWIN");
        if (resumeReplay) replayController?.StartDrive();
    }

    private void UpdateModeLabel(string message)
    {
        if (modeStatusText != null)
            modeStatusText.text = message;
    }
}

using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class LiveTestCoordinator : MonoBehaviour
{
    [Header("Live Twin")]
    [SerializeField] private DigitalTwinStateManager stateManager;
    [SerializeField] private NuScenesReplayController replayController;
    [SerializeField] private ReplayRoadGenerator roadGenerator;
    [SerializeField] private ReplayActorManager replayActorManager;
    [Tooltip("Legacy replay route-provider fallback. ReplayRoadGenerator is preferred.")]
    [SerializeField] private MonoBehaviour routeProviderComponent;
    [Tooltip("Route provider used by simulated and live-stream sources.")]
    [SerializeField] private MonoBehaviour streamingRouteProviderComponent;
    [SerializeField] private Transform replayRoot;
    [SerializeField] private Transform liveVehicleRoot;

    [Header("Test Lab")]
    [SerializeField] private Transform testWorldRoot;
    [Tooltip("Optional. If empty, the live vehicle hierarchy is cloned.")]
    [SerializeField] private GameObject testVehiclePrefab;
    [SerializeField] private float testVehicleModelYawOffset;
    [SerializeField, Min(0.25f)] private float routePointSpacing = 1f;
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

    private float capturedSourceTime;
    private float capturedSpeedKph;
    private Vector3 capturedPosition;
    private Quaternion capturedRotation;
    private bool sourceWasRunningBeforeTest;
    private bool sourceWasPausedByTest;
    private bool replayActorsWereVisible = true;
    private ITestWorldRouteProvider activeRouteProvider;
    private Button runtimeRiskDrivingButton;
    private TMP_Text runtimeRiskDrivingButtonLabel;

    private void Update()
    {
        if (!IsInTestLab || TestVehicle == null || !TestVehicle.IsRiskDrivingActive)
            return;

        if (TestVehicle.IsMlRiskTakeoverActive)
        {
            SetRiskButtonLabel("RISK: ML");
            UpdateModeLabel("TEST LAB — ML SAFETY TAKEOVER");
        }
        else
        {
            SetRiskButtonLabel("RISK: DRIVER");
            UpdateModeLabel(
                $"TEST LAB — RECKLESS DRIVER {TestVehicle.CurrentSpeedKph:F0} / " +
                $"{TestVehicle.CurrentDriverRequestedSpeedKph:F0} KM/H"
            );
        }
    }

    private void Awake()
    {
        if (stateManager == null) stateManager = FindFirstObjectByType<DigitalTwinStateManager>();
        if (replayController == null) replayController = FindFirstObjectByType<NuScenesReplayController>();
        if (roadGenerator == null) roadGenerator = FindFirstObjectByType<ReplayRoadGenerator>();
        if (replayActorManager == null) replayActorManager = FindFirstObjectByType<ReplayActorManager>();
        if (streamingRouteProviderComponent == null)
            streamingRouteProviderComponent = FindAnyObjectByType<StreamingRouteProvider>();
        if (followCamera == null) followCamera = FindFirstObjectByType<VehicleFollowCamera>();

        if (routeProviderComponent != null && !(routeProviderComponent is ITestWorldRouteProvider))
        {
            Debug.LogError("LiveTestCoordinator Replay Route Provider does not implement ITestWorldRouteProvider.");
        }

        if (streamingRouteProviderComponent != null &&
            !(streamingRouteProviderComponent is ITestWorldRouteProvider))
        {
            Debug.LogError("LiveTestCoordinator Streaming Route Provider does not implement ITestWorldRouteProvider.");
        }

        if (replayRoot == null && liveVehicleRoot != null)
            replayRoot = liveVehicleRoot.parent;

        if (testWorldRoot == null)
        {
            GameObject root = new GameObject("TestWorldRoot");
            root.transform.SetParent(replayRoot != null ? replayRoot : transform, false);
            testWorldRoot = root.transform;
        }
    }

    private void OnEnable()
    {
        // Runtime-only UI installation keeps the saved Canvas untouched and
        // also survives Unity play-mode/domain reload ordering.
        EnsureRuntimeRiskDrivingButton();
    }

    private void Start()
    {
        EnsureRuntimeRiskDrivingButton();
        ShowLiveTwin(false);
    }

    public void CreateTest()
    {
        if (stateManager == null || !stateManager.IsReady || liveVehicleRoot == null)
        {
            Debug.LogWarning("LiveTestCoordinator cannot create a test until the digital-twin state and vehicle are ready.");
            return;
        }

        sourceWasRunningBeforeTest = stateManager.IsPlaying;
        replayActorsWereVisible = replayActorManager == null || replayActorManager.RuntimeActorsVisible;
        PauseActiveSourceIfSupported();
        capturedSourceTime = stateManager.CurrentTime;
        // If Create Test is pressed before Drive, begin the autonomous test from
        // rest instead of inheriting a non-zero speed stored in the first frame.
        capturedSpeedKph = sourceWasRunningBeforeTest
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

    public void StartRiskDriving()
    {
        if (TestVehicle == null)
        {
            Debug.LogWarning("Press Create Test before Risk Driving.");
            return;
        }

        TestVehicle.SetRiskDriving(true);
        if (metricsRecorder != null && !metricsRecorder.IsRecording)
        {
            string weather = weatherController != null ? weatherController.CurrentWeather.ToString() : "Dry";
            metricsRecorder.BeginRecording("Risky Driver — ML Takeover", weather);
        }
        TestVehicle.SetRunning(true);
        SetRiskButtonLabel("RISK: DRIVER");
        UpdateModeLabel("TEST LAB — RECKLESS DRIVER (ML OBSERVING)");
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
        SetRiskButtonLabel("RISK DRIVING");
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
        if (replayActorManager != null)
            replayActorManager.SetRuntimeActorsVisible(replayActorsWereVisible);
        activeRouteProvider?.ExitTestMode();
        activeRouteProvider = null;
        if (followCamera != null && liveVehicleRoot != null) followCamera.SetTarget(liveVehicleRoot);
        if (liveTwinPanel != null) liveTwinPanel.SetActive(true);
        if (testLabPanel != null) testLabPanel.SetActive(false);

        IsInTestLab = false;
        SetRiskButtonLabel("RISK DRIVING");
        UpdateModeLabel("LIVE TWIN");
        ResumeActiveSourceIfNeeded();
    }

    private void BuildTestFromCapturedState()
    {
        if (!TryBuildRemainingWorldRoute(out List<Vector3> route, out string routeFailure))
        {
            Debug.LogError(routeFailure);
            UpdateModeLabel("TEST LAB — ROUTE UNAVAILABLE");
            activeRouteProvider = null;
            if (!IsInTestLab)
                ResumeActiveSourceIfNeeded();
            return;
        }

        if (TestVehicle != null)
            TestVehicle.SetRunning(false);

        scenarioManager?.ClearScenarios();
        weatherController?.SetDry();
        metricsRecorder?.ResetResults();
        ClearTestVehicle();

        if (testWorldRoot != null)
            testWorldRoot.gameObject.SetActive(true);

        replayActorManager?.SetRuntimeActorsVisible(false);
        activeRouteProvider?.EnterTestMode();
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

        // The bridge is created only on the temporary Test Lab clone. The live
        // twin, scene hierarchy, and Canvas remain unchanged. If the external
        // ML service is unavailable, the controller continues with its safe
        // deterministic fallback instead of blocking the test.
        TestLabMlDecisionBridge mlBridge = testVehicleObject.GetComponent<TestLabMlDecisionBridge>();
        if (mlBridge == null)
            mlBridge = testVehicleObject.AddComponent<TestLabMlDecisionBridge>();
        TestVehicle.SetMlDecisionBridge(mlBridge);

        TestVehicle.SetRoute(route, capturedSpeedKph);
        TestVehicle.SetRunning(false);

        if (liveVehicleRoot != null) liveVehicleRoot.gameObject.SetActive(false);
        if (scenarioManager != null) scenarioManager.SetTestVehicle(TestVehicle);
        if (weatherController != null) weatherController.SetTestVehicle(TestVehicle);
        if (metricsRecorder != null) metricsRecorder.AttachVehicle(TestVehicle);
        if (followCamera != null) followCamera.SetTarget(TestVehicle.transform);
        if (liveTwinPanel != null) liveTwinPanel.SetActive(false);
        if (testLabPanel != null) testLabPanel.SetActive(true);

        IsInTestLab = true;
        SetRiskButtonLabel("RISK DRIVING");
        UpdateModeLabel("TEST LAB — READY");
    }

    private void EnsureRuntimeRiskDrivingButton()
    {
        if (runtimeRiskDrivingButton != null || testLabPanel == null)
            return;

        Button template = null;
        Button[] candidates = testLabPanel.GetComponentsInChildren<Button>(true);
        for (int index = 0; index < candidates.Length; index++)
        {
            if (candidates[index] == null)
                continue;
            if (candidates[index].name == "RiskDrivingButton_Runtime")
            {
                runtimeRiskDrivingButton = candidates[index];
                runtimeRiskDrivingButtonLabel = candidates[index].GetComponentInChildren<TMP_Text>(true);
                runtimeRiskDrivingButton.onClick = new Button.ButtonClickedEvent();
                runtimeRiskDrivingButton.onClick.AddListener(StartRiskDriving);
                SetRiskButtonLabel("RISK DRIVING");
                return;
            }
            if (candidates[index].name == "RunTestButton")
            {
                template = candidates[index];
            }
        }

        if (template == null)
        {
            Debug.LogWarning("Risk Driving button could not be created because RunTestButton was not found.");
            return;
        }

        GameObject buttonObject = Instantiate(template.gameObject, template.transform.parent);
        buttonObject.name = "RiskDrivingButton_Runtime";
        buttonObject.SetActive(true);
        runtimeRiskDrivingButton = buttonObject.GetComponent<Button>();
        runtimeRiskDrivingButton.onClick = new Button.ButtonClickedEvent();
        runtimeRiskDrivingButton.onClick.AddListener(StartRiskDriving);

        RectTransform rect = buttonObject.GetComponent<RectTransform>();
        RectTransform templateRect = template.GetComponent<RectTransform>();
        if (rect != null && templateRect != null)
            rect.anchoredPosition = templateRect.anchoredPosition + new Vector2(180f, 0f);

        runtimeRiskDrivingButtonLabel = buttonObject.GetComponentInChildren<TMP_Text>(true);
        SetRiskButtonLabel("RISK DRIVING");

        ColorBlock colors = runtimeRiskDrivingButton.colors;
        colors.normalColor = new Color(1f, 0.72f, 0.28f, 1f);
        colors.highlightedColor = new Color(1f, 0.82f, 0.45f, 1f);
        runtimeRiskDrivingButton.colors = colors;
    }

    private void SetRiskButtonLabel(string label)
    {
        if (runtimeRiskDrivingButtonLabel != null)
            runtimeRiskDrivingButtonLabel.text = label;
    }

    private bool TryBuildRemainingWorldRoute(out List<Vector3> route, out string failure)
    {
        route = new List<Vector3>();
        failure = null;
        activeRouteProvider = ResolveRouteProvider(out failure);
        if (activeRouteProvider == null)
            return false;

        if (!activeRouteProvider.IsRouteAvailable)
        {
            failure = $"Create Test cannot start: {activeRouteProvider.ProviderName} has no usable route yet.";
            return false;
        }

        if (!activeRouteProvider.TryBuildTestRoute(
                capturedPosition,
                capturedSourceTime,
                routePointSpacing,
                route) || route.Count < 2)
        {
            failure = $"Create Test cannot start: {activeRouteProvider.ProviderName} could not provide at least two forward route points.";
            return false;
        }

        return true;
    }

    private ITestWorldRouteProvider ResolveRouteProvider(out string failure)
    {
        TwinSourceKind sourceKind = stateManager?.Session?.sourceKind ?? TwinSourceKind.Unknown;
        return ResolveRouteProviderForSource(sourceKind, out failure);
    }

    private ITestWorldRouteProvider ResolveRouteProviderForSource(
        TwinSourceKind sourceKind,
        out string failure)
    {
        MonoBehaviour providerComponent;
        string expectedProvider;
        switch (sourceKind)
        {
            case TwinSourceKind.Replay:
                providerComponent = roadGenerator != null
                    ? roadGenerator
                    : routeProviderComponent;
                expectedProvider = "ReplayRoadGenerator";
                break;
            case TwinSourceKind.SimulatedStream:
            case TwinSourceKind.LiveSensor:
                providerComponent = streamingRouteProviderComponent;
                expectedProvider = "StreamingRouteProvider";
                break;
            default:
                failure = $"Create Test cannot start: source kind '{sourceKind}' does not identify a route provider.";
                return null;
        }

        if (providerComponent == null)
        {
            failure = $"Create Test cannot start: source kind '{sourceKind}' requires {expectedProvider}, but none is assigned.";
            return null;
        }

        ITestWorldRouteProvider provider = providerComponent as ITestWorldRouteProvider;
        if (provider != null)
        {
            failure = null;
            return provider;
        }

        failure = $"Create Test cannot start: '{providerComponent.name}' is not a compatible {expectedProvider}.";
        return null;
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
        replayActorManager?.SetRuntimeActorsVisible(true);
        if (liveTwinPanel != null) liveTwinPanel.SetActive(true);
        if (testLabPanel != null) testLabPanel.SetActive(false);
        if (followCamera != null && liveVehicleRoot != null) followCamera.SetTarget(liveVehicleRoot);
        IsInTestLab = false;
        UpdateModeLabel("LIVE TWIN");
        if (resumeReplay) replayController?.StartDrive();
    }

    private void PauseActiveSourceIfSupported()
    {
        sourceWasPausedByTest = false;
        if (!sourceWasRunningBeforeTest || stateManager == null)
            return;

        bool supportsPause = (stateManager.Session.capabilities & TwinSourceCapabilities.Pause) != 0;
        ITwinSessionControl sessionControl = stateManager.SessionControl;
        if (supportsPause && sessionControl != null)
        {
            sessionControl.PauseSession();
            sourceWasPausedByTest = true;
            return;
        }

        if (stateManager.Session.sourceKind == TwinSourceKind.Replay && replayController != null)
        {
            replayController.PauseReplay();
            sourceWasPausedByTest = true;
        }
    }

    private void ResumeActiveSourceIfNeeded()
    {
        if (!resumeReplayWhenReturning || !sourceWasRunningBeforeTest || !sourceWasPausedByTest)
            return;

        ITwinSessionControl sessionControl = stateManager != null ? stateManager.SessionControl : null;
        if (sessionControl != null)
            sessionControl.StartSession();
        else
            replayController?.StartDrive();
        sourceWasPausedByTest = false;
    }

    private void OnDisable()
    {
        if (!IsInTestLab)
            return;

        if (liveVehicleRoot != null)
            liveVehicleRoot.gameObject.SetActive(true);
        if (testWorldRoot != null)
            testWorldRoot.gameObject.SetActive(false);
        if (replayActorManager != null)
            replayActorManager.SetRuntimeActorsVisible(replayActorsWereVisible);
        activeRouteProvider?.ExitTestMode();
    }

    private void UpdateModeLabel(string message)
    {
        if (modeStatusText != null)
            modeStatusText.text = message;
    }
}

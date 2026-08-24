using TMPro;
using UnityEngine;

public class TwinDashboard : MonoBehaviour
{
    public static TwinDashboard Instance { get; private set; }

    [Header("RoadWeave References")]
    [SerializeField] private DigitalTwinStateManager stateManager;
    [Tooltip("Compatibility reference. Any ITwinSessionControl source is used at runtime.")]
    [SerializeField] private NuScenesReplayController replayController;

    [Header("Information Panel")]
    [SerializeField] private GameObject informationPanel;
    [SerializeField] private TMP_Text titleText;
    [SerializeField] private TMP_Text contentText;

    [Header("Drive Button")]
    [SerializeField] private TMP_Text driveButtonText;

    private SelectedComponent selectedComponent = SelectedComponent.None;
    private ITwinSessionControl sessionControl;

    private enum SelectedComponent
    {
        None,
        VehicleBody,
        FrontLeftWheel,
        FrontRightWheel,
        RearLeftWheel,
        RearRightWheel
    }

    private void Awake()
    {
        Instance = this;
        if (stateManager == null) stateManager = FindAnyObjectByType<DigitalTwinStateManager>();
        ResolveSessionControl();
    }

    private void OnEnable()
    {
        if (stateManager == null) stateManager = FindAnyObjectByType<DigitalTwinStateManager>();
        if (stateManager != null) stateManager.SourceChanged += HandleSourceChanged;
        ResolveSessionControl();
    }

    private void OnDisable()
    {
        if (stateManager != null) stateManager.SourceChanged -= HandleSourceChanged;
    }

    private void OnDestroy()
    {
        if (Instance == this) Instance = null;
    }

    private void Start()
    {
        if (informationPanel != null) informationPanel.SetActive(false);
    }

    private void Update()
    {
        UpdateDriveButtonText();
        if (selectedComponent != SelectedComponent.None) RefreshInformation();
    }

    public void OnDriveButtonPressed()
    {
        // The connected source can change at runtime (replay -> simulated -> live).
        // Resolve on every command so an old replay controller is never controlled.
        ResolveSessionControl();
        if (sessionControl == null)
        {
            Debug.LogWarning("No digital-twin session source is available for the Drive button.");
            return;
        }

        switch (stateManager?.Session?.status ?? TwinSessionStatus.Disconnected)
        {
            case TwinSessionStatus.Running:
                sessionControl.PauseSession();
                break;
            case TwinSessionStatus.Finished:
                sessionControl.RestartSession(true);
                break;
            default:
                sessionControl.StartSession();
                break;
        }
    }

    public void SelectVehicleBody() => Select(SelectedComponent.VehicleBody);
    public void SelectFrontLeftWheel() => Select(SelectedComponent.FrontLeftWheel);
    public void SelectFrontRightWheel() => Select(SelectedComponent.FrontRightWheel);
    public void SelectRearLeftWheel() => Select(SelectedComponent.RearLeftWheel);
    public void SelectRearRightWheel() => Select(SelectedComponent.RearRightWheel);

    public bool SelectComponent(string componentId)
    {
        switch (componentId)
        {
            case "vehicle_body": SelectVehicleBody(); return true;
            case "wheel_lf":
            case "wheel_fl": SelectFrontLeftWheel(); return true;
            case "wheel_rf":
            case "wheel_fr": SelectFrontRightWheel(); return true;
            case "wheel_lr":
            case "wheel_rl": SelectRearLeftWheel(); return true;
            case "wheel_rr": SelectRearRightWheel(); return true;
            default: return false;
        }
    }

    public void HideInformation()
    {
        selectedComponent = SelectedComponent.None;
        if (informationPanel != null) informationPanel.SetActive(false);
    }

    private void Select(SelectedComponent component)
    {
        selectedComponent = component;
        if (informationPanel != null) informationPanel.SetActive(true);
        RefreshInformation();
    }

    private void RefreshInformation()
    {
        if (stateManager == null || titleText == null || contentText == null) return;
        if (!stateManager.IsReady)
        {
            titleText.text = "RoadWeave";
            contentText.text = stateManager.Session?.status == TwinSessionStatus.Error
                ? $"Data source error\n{stateManager.Session.statusMessage}"
                : "Waiting for digital-twin data...";
            return;
        }

        switch (selectedComponent)
        {
            case SelectedComponent.VehicleBody: ShowVehicleBody(); break;
            case SelectedComponent.FrontLeftWheel: ShowWheel("Front Left Wheel", stateManager.Wheels.frontLeftRpm); break;
            case SelectedComponent.FrontRightWheel: ShowWheel("Front Right Wheel", stateManager.Wheels.frontRightRpm); break;
            case SelectedComponent.RearLeftWheel: ShowWheel("Rear Left Wheel", stateManager.Wheels.rearLeftRpm); break;
            case SelectedComponent.RearRightWheel: ShowWheel("Rear Right Wheel", stateManager.Wheels.rearRightRpm); break;
        }
    }

    private void ShowVehicleBody()
    {
        TwinVehicleState vehicle = stateManager.Vehicle;
        TwinSnapshotMetadata metadata = stateManager.Metadata;
        double duration = metadata?.timelineDurationSeconds ?? 0d;
        string time = duration > 0d
            ? $"{stateManager.CurrentTime:F1} / {duration:F1} s"
            : $"{stateManager.CurrentTime:F1} s";
        string temperature = vehicle.temperatureIsValid
            ? $"{vehicle.temperatureCelsius:F1} °C"
            : "N/A (source did not provide it)";
        string freshness = metadata?.freshness == TwinDataFreshness.Stale ? "STALE — waiting for update" : "Fresh";

        titleText.text = "Vehicle Body";
        contentText.text =
            $"Source: {stateManager.Session.sourceKind} ({stateManager.Session.sourceId})\n" +
            $"Session: {stateManager.Session.status}\n" +
            $"Data: {freshness} / {metadata?.validity}\n" +
            $"Time: {time}\n" +
            $"Speed: {vehicle.speedKilometersPerHour:F2} km/h\n" +
            $"Battery: {vehicle.batteryPercent:F0}%\n" +
            $"Remaining Distance: {vehicle.availableDistanceKilometers:F0} km\n" +
            $"Gear Position: {vehicle.gearPosition}\n" +
            $"Throttle: {vehicle.throttlePercent:F0}%\n" +
            $"Brake: {vehicle.brake:F1}\n" +
            $"Steering: {vehicle.steeringDegrees:F1}°\n" +
            $"Acceleration: {stateManager.Ego.longitudinalAcceleration:F2} m/s²\n" +
            $"Temperature: {temperature}";
    }

    private void ShowWheel(string displayName, float rpm)
    {
        string wheelStatus = !stateManager.IsFresh
            ? "Data stale"
            : !stateManager.IsPlaying ? "Paused"
            : Mathf.Abs(rpm) > 0.1f ? "Rotating" : "Stopped";
        titleText.text = displayName;
        contentText.text =
            $"Source Time: {stateManager.CurrentTime:F1} s\n" +
            $"Wheel Speed: {rpm:F2} RPM\n" +
            $"Status: {wheelStatus}";
    }

    private void UpdateDriveButtonText()
    {
        if (driveButtonText == null || stateManager == null) return;
        switch (stateManager.Session.status)
        {
            case TwinSessionStatus.Connecting: driveButtonText.text = "Loading..."; break;
            case TwinSessionStatus.Running: driveButtonText.text = "Pause"; break;
            case TwinSessionStatus.Finished: driveButtonText.text = "Restart"; break;
            case TwinSessionStatus.Error: driveButtonText.text = "Data Error"; break;
            case TwinSessionStatus.Disconnected: driveButtonText.text = "No Source"; break;
            default: driveButtonText.text = "Drive"; break;
        }
    }

    private void ResolveSessionControl()
    {
        sessionControl = stateManager?.SessionControl;
        if (sessionControl != null) return;
        // Compatibility for existing scenes during Awake ordering. Once a source
        // connects, HandleSourceChanged replaces this fallback immediately.
        if (replayController == null) replayController = FindAnyObjectByType<NuScenesReplayController>();
        if (replayController != null) { sessionControl = replayController; return; }
        sessionControl = null;
    }

    private void HandleSourceChanged(ITwinStateSource source)
    {
        sessionControl = source as ITwinSessionControl;
    }
}

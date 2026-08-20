using TMPro;
using UnityEngine;

public class TwinDashboard : MonoBehaviour
{
    public static TwinDashboard Instance { get; private set; }

    [Header("RoadWeave References")]
    [SerializeField] private DigitalTwinStateManager stateManager;
    [SerializeField] private NuScenesReplayController replayController;

    [Header("Information Panel")]
    [SerializeField] private GameObject informationPanel;
    [SerializeField] private TMP_Text titleText;
    [SerializeField] private TMP_Text contentText;

    [Header("Drive Button")]
    [SerializeField] private TMP_Text driveButtonText;

    private SelectedComponent selectedComponent = SelectedComponent.None;

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

        if (stateManager == null)
            stateManager = FindFirstObjectByType<DigitalTwinStateManager>();
        if (replayController == null)
            replayController = FindFirstObjectByType<NuScenesReplayController>();
    }

    private void OnDestroy()
    {
        if (Instance == this)
            Instance = null;
    }

    private void Start()
    {
        if (informationPanel != null)
            informationPanel.SetActive(false);
    }

    private void Update()
    {
        UpdateDriveButtonText();

        if (selectedComponent != SelectedComponent.None)
            RefreshInformation();
    }

    public void OnDriveButtonPressed()
    {
        if (replayController != null)
            replayController.HandleDriveButton();
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
        if (informationPanel != null)
            informationPanel.SetActive(false);
    }

    private void Select(SelectedComponent component)
    {
        selectedComponent = component;
        if (informationPanel != null)
            informationPanel.SetActive(true);
        RefreshInformation();
    }

    private void RefreshInformation()
    {
        if (stateManager == null || titleText == null || contentText == null)
            return;

        if (!stateManager.IsReady)
        {
            titleText.text = "RoadWeave";
            contentText.text = stateManager.Status == ReplayStatus.Error
                ? "The replay could not be loaded. Check the Console."
                : "Loading replay data...";
            return;
        }

        switch (selectedComponent)
        {
            case SelectedComponent.VehicleBody:
                ShowVehicleBody();
                break;
            case SelectedComponent.FrontLeftWheel:
                ShowWheel("Front Left Wheel", stateManager.Wheels.frontLeftRpm);
                break;
            case SelectedComponent.FrontRightWheel:
                ShowWheel("Front Right Wheel", stateManager.Wheels.frontRightRpm);
                break;
            case SelectedComponent.RearLeftWheel:
                ShowWheel("Rear Left Wheel", stateManager.Wheels.rearLeftRpm);
                break;
            case SelectedComponent.RearRightWheel:
                ShowWheel("Rear Right Wheel", stateManager.Wheels.rearRightRpm);
                break;
        }
    }

    private void ShowVehicleBody()
    {
        TwinVehicleState vehicle = stateManager.Vehicle;
        titleText.text = "Vehicle Body";
        contentText.text =
            $"Replay: {stateManager.Status}\n" +
            $"Time: {stateManager.CurrentTime:F1} / {stateManager.Package.durationSeconds:F1} s\n" +
            $"Speed: {vehicle.speedKilometersPerHour:F2} km/h\n" +
            $"Battery: {vehicle.batteryPercent:F0}%\n" +
            $"Remaining Distance: {vehicle.availableDistanceKilometers:F0} km\n" +
            $"Gear Position: {vehicle.gearPosition}\n" +
            $"Throttle: {vehicle.throttlePercent:F0}%\n" +
            $"Brake: {vehicle.brake:F1}\n" +
            $"Steering: {vehicle.steeringDegrees:F1}°\n" +
            $"Acceleration: {stateManager.Ego.longitudinalAcceleration:F2} m/s²\n" +
            "Temperature: N/A (not recorded by nuScenes)";
    }

    private void ShowWheel(string displayName, float rpm)
    {
        string wheelStatus = !stateManager.IsPlaying
            ? "Paused"
            : Mathf.Abs(rpm) > 0.1f ? "Rotating" : "Stopped";

        titleText.text = displayName;
        contentText.text =
            $"Replay Time: {stateManager.CurrentTime:F1} s\n" +
            $"Wheel Speed: {rpm:F2} RPM\n" +
            $"Status: {wheelStatus}";
    }

    private void UpdateDriveButtonText()
    {
        if (driveButtonText == null || stateManager == null)
            return;

        switch (stateManager.Status)
        {
            case ReplayStatus.Loading: driveButtonText.text = "Loading..."; break;
            case ReplayStatus.Playing: driveButtonText.text = "Pause"; break;
            case ReplayStatus.Finished: driveButtonText.text = "Restart"; break;
            case ReplayStatus.Error: driveButtonText.text = "Data Error"; break;
            default: driveButtonText.text = "Drive"; break;
        }
    }
}

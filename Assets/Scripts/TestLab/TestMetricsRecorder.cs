using TMPro;
using UnityEngine;

public class TestMetricsRecorder : MonoBehaviour
{
    [SerializeField] private TMP_Text resultsText;
    [SerializeField] private TestWeatherController weatherController;

    public bool IsRecording { get; private set; }

    private AutonomousTestVehicleController vehicle;
    private string scenarioName = "Clear Route";
    private string weatherName = "Dry";
    private float elapsedTime;
    private float startingSpeedKph;
    private Vector3 lastSamplePosition;
    private float travelledPathDistance;
    private float firstDetectionTime = -1f;
    private float firstBrakingTime = -1f;
    private float minimumObstacleDistance = float.PositiveInfinity;
    private bool collision;
    private bool destinationReached;
    private string displayStatus = "READY";
    private int obstacleBrakingEvents;
    private int curveSlowdownEvents;
    private int weatherSlowdownEvents;
    private int maneuverSlowdownEvents;

    private void Awake()
    {
        if (weatherController == null)
            weatherController = FindAnyObjectByType<TestWeatherController>();
        RefreshWeatherName();
    }

    private void Start()
    {
        ShowReadyMessage();
    }

    private void Update()
    {
        if (RefreshWeatherName())
        {
            if (vehicle != null)
                UpdateDisplay(displayStatus);
            else
                ShowReadyMessage();
        }

        if (!IsRecording || vehicle == null || !vehicle.IsRunning)
            return;

        elapsedTime += Time.deltaTime;
        AccumulatePathDistance(vehicle.transform.position);
        if (!float.IsInfinity(vehicle.NearestObstacleDistance))
            minimumObstacleDistance = Mathf.Min(minimumObstacleDistance, vehicle.NearestObstacleDistance);
        displayStatus = "RUNNING";
        UpdateDisplay("RUNNING");
    }

    public void AttachVehicle(AutonomousTestVehicleController newVehicle)
    {
        Unsubscribe();
        vehicle = newVehicle;
        if (vehicle == null)
            return;

        vehicle.ObstacleDetected += OnObstacleDetected;
        vehicle.BrakingStarted += OnBrakingStarted;
        vehicle.DecelerationStarted += OnDecelerationStarted;
        vehicle.DestinationReached += OnDestinationReached;
        vehicle.CollisionOccurred += OnCollisionOccurred;
    }

    public void BeginRecording(string newScenarioName, string newWeatherName)
    {
        if (vehicle == null)
        {
            Debug.LogWarning("TestMetricsRecorder has no test vehicle.");
            return;
        }

        scenarioName = string.IsNullOrEmpty(newScenarioName) ? "Clear Route" : newScenarioName;
        weatherName = string.IsNullOrEmpty(newWeatherName) ? "Dry" : newWeatherName;
        elapsedTime = 0f;
        startingSpeedKph = vehicle.CurrentSpeedKph;
        lastSamplePosition = vehicle.transform.position;
        travelledPathDistance = 0f;
        firstDetectionTime = -1f;
        firstBrakingTime = -1f;
        minimumObstacleDistance = float.PositiveInfinity;
        collision = false;
        destinationReached = false;
        obstacleBrakingEvents = 0;
        curveSlowdownEvents = 0;
        weatherSlowdownEvents = 0;
        maneuverSlowdownEvents = 0;
        IsRecording = true;
        displayStatus = "RUNNING";
        UpdateDisplay("RUNNING");
    }

    public void StopRecording()
    {
        if (!IsRecording)
            return;
        IsRecording = false;
        displayStatus = "STOPPED";
        UpdateDisplay("STOPPED");
    }

    public void ResetResults()
    {
        IsRecording = false;
        displayStatus = "READY";
        RefreshWeatherName();
        ShowReadyMessage();
    }

    private void OnObstacleDetected(float distance)
    {
        if (!IsRecording)
            return;
        if (firstDetectionTime < 0f)
            firstDetectionTime = elapsedTime;
        minimumObstacleDistance = Mathf.Min(minimumObstacleDistance, distance);
    }

    private void AccumulatePathDistance(Vector3 currentPosition)
    {
        travelledPathDistance += Vector3.Distance(lastSamplePosition, currentPosition);
        lastSamplePosition = currentPosition;
    }

    private void OnBrakingStarted()
    {
        if (!IsRecording)
            return;
        obstacleBrakingEvents++;
        if (firstBrakingTime < 0f)
            firstBrakingTime = elapsedTime;
    }

    private void OnDecelerationStarted(VehicleDecelerationReason reason)
    {
        if (!IsRecording)
            return;
        switch (reason)
        {
            case VehicleDecelerationReason.Curve:
                curveSlowdownEvents++;
                break;
            case VehicleDecelerationReason.Weather:
                weatherSlowdownEvents++;
                break;
            case VehicleDecelerationReason.Maneuver:
                maneuverSlowdownEvents++;
                break;
        }
    }

    private void OnDestinationReached()
    {
        destinationReached = true;
        IsRecording = false;
        displayStatus = "COMPLETED";
        UpdateDisplay("COMPLETED");
    }

    private void OnCollisionOccurred()
    {
        collision = true;
        IsRecording = false;
        displayStatus = "COLLISION";
        UpdateDisplay("COLLISION");
    }

    private void UpdateDisplay(string status)
    {
        if (resultsText == null || vehicle == null)
            return;

        string detection = firstDetectionTime < 0f ? "Not detected" : $"{firstDetectionTime:F2} s";
        string braking = firstBrakingTime < 0f ? "Not started" : $"{firstBrakingTime:F2} s";
        string reaction = firstDetectionTime >= 0f && firstBrakingTime >= firstDetectionTime
            ? $"{firstBrakingTime - firstDetectionTime:F2} s"
            : "N/A";
        string minimumDistance = float.IsInfinity(minimumObstacleDistance)
            ? "No obstacle"
            : $"{minimumObstacleDistance:F2} m";

        resultsText.text =
            $"Status: {status}\n" +
            $"Scenario: {scenarioName}\n" +
            $"Weather: {weatherName}\n" +
            $"Maneuver: {vehicle.CurrentManeuver}\n" +
            $"Simulated Sensors:\n{vehicle.SensorSummary}\n" +
            $"Elapsed Time: {elapsedTime:F2} s\n" +
            $"Starting Speed: {startingSpeedKph:F1} km/h\n" +
            $"Current Speed: {vehicle.CurrentSpeedKph:F1} km/h\n" +
            $"Distance Travelled (path): {travelledPathDistance:F1} m\n" +
            $"First Detection: {detection}\n" +
            $"First Obstacle Braking: {braking}\n" +
            $"Reaction Time: {reaction}\n" +
            $"Current Slowdown Reason: {vehicle.CurrentDecelerationReason}\n" +
            $"Slowdown Events: obstacle {obstacleBrakingEvents}, curve {curveSlowdownEvents}, " +
            $"weather {weatherSlowdownEvents}, maneuver {maneuverSlowdownEvents}\n" +
            $"Minimum Obstacle Distance: {minimumDistance}\n" +
            $"Collision: {(collision ? "Yes" : "No")}\n" +
            $"Destination Reached: {(destinationReached ? "Yes" : "No")}";
    }

    private void ShowReadyMessage()
    {
        if (resultsText != null)
            resultsText.text =
                $"Status: READY\n" +
                $"Weather: {weatherName}\n\n" +
                "Add a scenario, choose weather, then press Run Test.";
    }

    private bool RefreshWeatherName()
    {
        if (weatherController == null)
            return false;

        string latestWeather = weatherController.CurrentWeather.ToString();
        if (weatherName == latestWeather)
            return false;

        weatherName = latestWeather;
        return true;
    }

    private void Unsubscribe()
    {
        if (vehicle == null)
            return;
        vehicle.ObstacleDetected -= OnObstacleDetected;
        vehicle.BrakingStarted -= OnBrakingStarted;
        vehicle.DecelerationStarted -= OnDecelerationStarted;
        vehicle.DestinationReached -= OnDestinationReached;
        vehicle.CollisionOccurred -= OnCollisionOccurred;
    }

    private void OnDestroy()
    {
        Unsubscribe();
    }
}

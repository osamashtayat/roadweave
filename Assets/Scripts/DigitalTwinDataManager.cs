using System.IO;
using UnityEngine;

public class DigitalTwinDataManager : MonoBehaviour
{
    public static DigitalTwinDataManager Instance { get; private set; }

    public DigitalTwinData Data { get; private set; }

    private void Awake()
    {
        Instance = this;
        LoadData();
    }

    private void LoadData()
    {
        string path = Path.Combine(
            Application.streamingAssetsPath,
            "digital_twin.json"
        );

        if (!File.Exists(path))
        {
            Debug.LogError("digital_twin.json was not found at: " + path);
            return;
        }

        string json = File.ReadAllText(path);

        Data = JsonUtility.FromJson<DigitalTwinData>(json);

        if (Data == null)
        {
            Debug.LogError("Could not read digital_twin.json");
            return;
        }

        Debug.Log("Digital twin data loaded successfully.");
    }

    public WheelData GetWheel(string componentId)
    {
        if (Data == null || Data.wheels == null)
            return null;

        foreach (WheelData wheel in Data.wheels)
        {
            if (wheel.componentId == componentId)
                return wheel;
        }

        return null;
    }

    public VehicleSensorData GetSensor(string sensorId)
    {
        if (Data == null || Data.vehicleSensors == null)
            return null;

        foreach (VehicleSensorData sensor in Data.vehicleSensors)
        {
            if (sensor.sensorId == sensorId)
                return sensor;
        }

        return null;
    }
}
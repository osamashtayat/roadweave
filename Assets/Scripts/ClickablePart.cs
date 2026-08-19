using UnityEngine;

public class ClickablePart : MonoBehaviour
{
    [SerializeField] private string componentId;

    private void OnMouseDown()
    {
        if (componentId == "vehicle_body")
        {
            ShowVehicleBody();
        }
        else if (componentId.StartsWith("wheel_"))
        {
            ShowWheel();
        }
        else if (
            componentId == "cam_front" ||
            componentId == "radar_front" ||
            componentId == "lidar_top"
        )
        {
            ShowSensor();
        }
    }

    private void ShowVehicleBody()
    {
        VehicleBodyData body =
            DigitalTwinDataManager.Instance.Data.vehicle_body;

        string content =
            $"Speed: {body.vehicle_speed:F2} km/h\n" +
            $"Battery: {body.battery_level}%\n" +
            $"Remaining Distance: {body.available_distance} km\n" +
            $"Gear Position: {body.gear_position}\n" +
            $"Throttle: {body.throttle}\n" +
            $"Brake: {body.brake}\n" +
            $"Steering: {body.steering:F1}°\n" +
            $"Steering Speed: {body.steering_speed:F1}\n" +
            $"Yaw Rate: {body.yaw_rate:F1}\n" +
            $"Longitudinal Acceleration: {body.longitudinal_accel:F2}";

        UIManager.Instance.ShowInformation(
            "Vehicle Body",
            content
        );
    }

    private void ShowWheel()
    {
        WheelData wheel =
            DigitalTwinDataManager.Instance.GetWheel(componentId);

        if (wheel == null)
        {
            Debug.LogError("Wheel not found: " + componentId);
            return;
        }

        string status =
            wheel.wheelSpeed > 0 ? "Rotating" : "Stopped";

        string content =
            $"Wheel Speed: {wheel.wheelSpeed:F2} RPM\n" +
            $"Status: {status}";

        UIManager.Instance.ShowInformation(
            wheel.displayName,
            content
        );
    }

    private void ShowSensor()
    {
        VehicleSensorData sensor =
            DigitalTwinDataManager.Instance.GetSensor(componentId);

        if (sensor == null)
        {
            Debug.LogError("Sensor not found: " + componentId);
            return;
        }

        string content = "";

        if (componentId == "cam_front")
        {
            content =
                $"Status: {sensor.status}\n" +
                $"Timestamp: {sensor.timestamp}\n" +
                $"Image File: {sensor.imageFile}\n" +
                $"Data Source: {sensor.dataSource}";
        }
        else if (componentId == "radar_front")
        {
            content =
                $"Status: {sensor.status}\n" +
                $"Detections: {sensor.detections}\n" +
                $"Closest Distance: {sensor.closestDistanceMeters:F2} m\n" +
                $"Farthest Distance: {sensor.farthestDistanceMeters:F2} m\n" +
                $"Average Speed: {sensor.averageSpeedMetersPerSecond:F2} m/s\n" +
                $"Data Source: {sensor.dataSource}";
        }
        else if (componentId == "lidar_top")
        {
            content =
                $"Status: {sensor.status}\n" +
                $"Point Count: {sensor.pointCount}\n" +
                $"Timestamp: {sensor.timestamp}\n" +
                $"Data Source: {sensor.dataSource}";
        }

        UIManager.Instance.ShowInformation(
            sensor.sensorName,
            content
        );
    }
}
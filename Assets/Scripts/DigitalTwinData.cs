using System;

[Serializable]
public class DigitalTwinData
{
    public string scene;
    public long timestamp;

    public VehicleBodyData vehicle_body;
    public WheelData[] wheels;
    public VehicleSensorData[] vehicleSensors;
}

[Serializable]
public class VehicleBodyData
{
    public int available_distance;
    public int battery_level;
    public int brake;
    public int brake_switch;
    public int gear_position;
    public int left_signal;
    public int right_signal;

    public float rear_left_rpm;
    public float rear_right_rpm;
    public float steering;
    public float steering_speed;
    public int throttle;
    public long utime;
    public float vehicle_speed;
    public float yaw_rate;
    public float longitudinal_accel;
}

[Serializable]
public class WheelData
{
    public string componentId;
    public string displayName;
    public float wheelSpeed;
}

[Serializable]
public class VehicleSensorData
{
    public string sensorId;
    public string sensorName;
    public string status;
    public long timestamp;

    public string imageFile;

    public int detections;
    public float closestDistanceMeters;
    public float farthestDistanceMeters;
    public float averageSpeedMetersPerSecond;

    public int pointCount;
    public string dataSource;
}
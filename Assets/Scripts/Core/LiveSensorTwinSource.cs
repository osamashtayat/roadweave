/// <summary>
/// Entry point for a future real-vehicle gateway. Network and sensor-specific
/// code stays outside RoadWeave presentation code and publishes canonical snapshots here.
/// </summary>
public class LiveSensorTwinSource : TwinPushSourceBase
{
    protected override TwinSourceKind SourceKind => TwinSourceKind.LiveSensor;
}

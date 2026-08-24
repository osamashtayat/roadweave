using System;
using UnityEngine;

[Serializable]
public struct TwinGpsCoordinate
{
    public double latitudeDegrees;
    public double longitudeDegrees;
    public double altitudeMeters;

    public TwinGpsCoordinate(double latitudeDegrees, double longitudeDegrees, double altitudeMeters)
    {
        this.latitudeDegrees = latitudeDegrees;
        this.longitudeDegrees = longitudeDegrees;
        this.altitudeMeters = altitudeMeters;
    }
}

[Serializable]
public class TwinGeoReference
{
    [Tooltip("WGS84 latitude of the Unity-local origin in degrees.")]
    public double latitudeDegrees;
    [Tooltip("WGS84 longitude of the Unity-local origin in degrees.")]
    public double longitudeDegrees;
    [Tooltip("Ellipsoidal altitude of the Unity-local origin in meters.")]
    public double altitudeMeters;

    public TwinGeoReference() { }

    public TwinGeoReference(double latitudeDegrees, double longitudeDegrees, double altitudeMeters)
    {
        this.latitudeDegrees = latitudeDegrees;
        this.longitudeDegrees = longitudeDegrees;
        this.altitudeMeters = altitudeMeters;
    }
}

/// <summary>
/// Validates the boundary between an input source and RoadWeave's canonical
/// Unity-local frame. Sources must convert before publishing a snapshot.
/// </summary>
public static class TwinCoordinateFrameService
{
    public const string LegacyNuScenesUnityDescription =
        "Unity local: x=right, y=up, z=forward; origin=first CAN pose";

    // WGS84 ellipsoid constants. GPS is converted to Earth-Centered Earth-Fixed,
    // then rotated into local East-Up-North. Unity receives x=East, y=Up, z=North.
    private const double Wgs84SemiMajorAxisMeters = 6378137.0;
    private const double Wgs84EccentricitySquared = 6.69437999014e-3;

    public static bool TryCreateCanonicalFrame(
        string sourceCoordinateSystem,
        out TwinCoordinateFrame frame,
        out string error)
    {
        string description = (sourceCoordinateSystem ?? string.Empty).Trim();
        bool knownCanonicalId =
            string.Equals(description, TwinCoordinateFrame.CanonicalFrameId, StringComparison.OrdinalIgnoreCase);
        bool knownLegacyNuScenes =
            string.Equals(description, LegacyNuScenesUnityDescription, StringComparison.OrdinalIgnoreCase);
        bool hasCanonicalAxes =
            description.IndexOf("x=right", StringComparison.OrdinalIgnoreCase) >= 0 &&
            description.IndexOf("y=up", StringComparison.OrdinalIgnoreCase) >= 0 &&
            description.IndexOf("z=forward", StringComparison.OrdinalIgnoreCase) >= 0;
        bool explicitlyMeters = ContainsUnit(description, "meter") || ContainsUnit(description, "meters") ||
                                ContainsUnit(description, "metre") || ContainsUnit(description, "metres");
        bool explicitlyUnsupportedUnit =
            ContainsUnit(description, "foot") || ContainsUnit(description, "feet") || ContainsUnit(description, "ft") ||
            ContainsUnit(description, "centimeter") || ContainsUnit(description, "centimeters") ||
            ContainsUnit(description, "centimetre") || ContainsUnit(description, "centimetres") ||
            ContainsUnit(description, "cm") || ContainsUnit(description, "millimeter") ||
            ContainsUnit(description, "millimeters") || ContainsUnit(description, "mm");
        bool isCanonical = knownCanonicalId || knownLegacyNuScenes ||
                           (hasCanonicalAxes && explicitlyMeters && !explicitlyUnsupportedUnit);

        if (!isCanonical)
        {
            frame = null;
            error = $"Unsupported coordinate system '{sourceCoordinateSystem}'. " +
                    "RoadWeave accepts Unity-local x=right, y=up, z=forward only when the unit is explicitly meters.";
            return false;
        }

        frame = TwinCoordinateFrame.UnityLocal(description);
        error = null;
        return true;
    }

    public static bool IsCanonical(TwinCoordinateFrame frame)
    {
        return frame != null &&
               string.Equals(frame.frameId, TwinCoordinateFrame.CanonicalFrameId, StringComparison.Ordinal) &&
               string.Equals(frame.horizontalUnit, "meter", StringComparison.OrdinalIgnoreCase) &&
               string.Equals(frame.angleUnit, "degree", StringComparison.OrdinalIgnoreCase) &&
               string.Equals(frame.xAxis, "right", StringComparison.OrdinalIgnoreCase) &&
               string.Equals(frame.yAxis, "up", StringComparison.OrdinalIgnoreCase) &&
               string.Equals(frame.zAxis, "forward", StringComparison.OrdinalIgnoreCase);
    }

    private static bool ContainsUnit(string description, string unit)
    {
        if (string.IsNullOrWhiteSpace(description)) return false;
        string normalized = description.ToLowerInvariant()
            .Replace('=', ' ').Replace(':', ' ').Replace(';', ' ').Replace(',', ' ')
            .Replace('(', ' ').Replace(')', ' ').Replace('[', ' ').Replace(']', ' ');
        string[] tokens = normalized.Split(new[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
        foreach (string token in tokens)
            if (string.Equals(token, unit, StringComparison.Ordinal)) return true;
        return false;
    }

    public static Vector3 SourceToCanonicalPosition(Vector3 sourcePosition, TwinCoordinateFrame sourceFrame)
    {
        if (!IsCanonical(sourceFrame))
            throw new ArgumentException("The source frame must be converted explicitly before publishing.");
        return sourcePosition;
    }

    public static float SourceToCanonicalYaw(float sourceYawDegrees, TwinCoordinateFrame sourceFrame)
    {
        if (!IsCanonical(sourceFrame))
            throw new ArgumentException("The source frame must be converted explicitly before publishing.");
        return sourceYawDegrees;
    }

    /// <summary>
    /// Creates canonical Unity-local metadata for a GPS-backed local ENU origin.
    /// Published snapshots still contain meters in Unity coordinates; GPS values
    /// remain an adapter-boundary concern.
    /// </summary>
    public static TwinCoordinateFrame CreateLocalEnuFrame(TwinGeoReference origin)
    {
        string originText = IsValidGeoReference(origin)
            ? $"WGS84 ENU origin lat={origin.latitudeDegrees:F8}, lon={origin.longitudeDegrees:F8}, alt={origin.altitudeMeters:F3}m"
            : "invalid WGS84 ENU origin";
        TwinCoordinateFrame frame = TwinCoordinateFrame.UnityLocal(originText);
        frame.parentFrameId = "WGS84";
        return frame;
    }

    /// <summary>
    /// Converts WGS84 GPS to RoadWeave's canonical local position in meters:
    /// x=East, y=Up, z=North. The supplied georeference is the Unity origin.
    /// </summary>
    public static bool TryGpsToCanonicalPosition(
        TwinGpsCoordinate gps,
        TwinGeoReference origin,
        out Vector3 canonicalPositionMeters,
        out string error)
    {
        canonicalPositionMeters = Vector3.zero;
        if (!IsValidGeoReference(origin))
        {
            error = "GPS origin must contain finite WGS84 latitude, longitude, and altitude values.";
            return false;
        }
        if (!IsValidLatitude(gps.latitudeDegrees) || !IsValidLongitude(gps.longitudeDegrees) ||
            !IsFinite(gps.altitudeMeters))
        {
            error = "GPS observation contains an invalid latitude, longitude, or altitude.";
            return false;
        }

        DoubleVector3 originEcef = GeodeticToEcef(
            origin.latitudeDegrees,
            origin.longitudeDegrees,
            origin.altitudeMeters);
        DoubleVector3 pointEcef = GeodeticToEcef(
            gps.latitudeDegrees,
            gps.longitudeDegrees,
            gps.altitudeMeters);
        DoubleVector3 delta = pointEcef - originEcef;

        double latitude = DegreesToRadians(origin.latitudeDegrees);
        double longitude = DegreesToRadians(origin.longitudeDegrees);
        double sinLat = Math.Sin(latitude);
        double cosLat = Math.Cos(latitude);
        double sinLon = Math.Sin(longitude);
        double cosLon = Math.Cos(longitude);

        double east = -sinLon * delta.x + cosLon * delta.y;
        double north = -sinLat * cosLon * delta.x - sinLat * sinLon * delta.y + cosLat * delta.z;
        double up = cosLat * cosLon * delta.x + cosLat * sinLon * delta.y + sinLat * delta.z;
        if (!FitsUnityFloat(east) || !FitsUnityFloat(up) || !FitsUnityFloat(north))
        {
            error = "GPS position is too far from the configured local origin for Unity float precision.";
            return false;
        }

        canonicalPositionMeters = new Vector3((float)east, (float)up, (float)north);
        error = null;
        return true;
    }

    public static bool IsValidGeoReference(TwinGeoReference origin)
    {
        return origin != null &&
               IsValidLatitude(origin.latitudeDegrees) &&
               IsValidLongitude(origin.longitudeDegrees) &&
               IsFinite(origin.altitudeMeters);
    }

    private static DoubleVector3 GeodeticToEcef(double latitudeDegrees, double longitudeDegrees, double altitudeMeters)
    {
        double latitude = DegreesToRadians(latitudeDegrees);
        double longitude = DegreesToRadians(longitudeDegrees);
        double sinLat = Math.Sin(latitude);
        double cosLat = Math.Cos(latitude);
        double sinLon = Math.Sin(longitude);
        double cosLon = Math.Cos(longitude);
        double radius = Wgs84SemiMajorAxisMeters /
                        Math.Sqrt(1.0 - Wgs84EccentricitySquared * sinLat * sinLat);
        return new DoubleVector3(
            (radius + altitudeMeters) * cosLat * cosLon,
            (radius + altitudeMeters) * cosLat * sinLon,
            (radius * (1.0 - Wgs84EccentricitySquared) + altitudeMeters) * sinLat);
    }

    private static bool IsValidLatitude(double value) => IsFinite(value) && value >= -90d && value <= 90d;
    private static bool IsValidLongitude(double value) => IsFinite(value) && value >= -180d && value <= 180d;
    private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
    private static bool FitsUnityFloat(double value) => IsFinite(value) && Math.Abs(value) <= 10000000d;
    private static double DegreesToRadians(double degrees) => degrees * Math.PI / 180d;

    private struct DoubleVector3
    {
        public double x;
        public double y;
        public double z;

        public DoubleVector3(double x, double y, double z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public static DoubleVector3 operator -(DoubleVector3 left, DoubleVector3 right)
        {
            return new DoubleVector3(left.x - right.x, left.y - right.y, left.z - right.z);
        }
    }
}

using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Source-neutral contract used by Test Lab to request a world-space route.
/// A replay, live stream, map service, or procedural simulator can implement
/// this contract without changing the Test Lab coordinator.
/// </summary>
public interface ITestWorldRouteProvider
{
    string ProviderName { get; }
    bool IsRouteAvailable { get; }

    void EnterTestMode();
    void ExitTestMode();

    bool TryBuildTestRoute(
        Vector3 worldStartPosition,
        float sourceTimeSeconds,
        float minimumPointSpacing,
        List<Vector3> worldRoute);
}

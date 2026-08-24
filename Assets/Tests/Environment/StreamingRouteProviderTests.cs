using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using NUnit.Framework;
using UnityEngine;

public class StreamingRouteProviderTests
{
    [Test]
    public void BuildsForwardRouteAndBoundedContinuation()
    {
        GameObject root = new GameObject("StreamingRouteProviderTest");
        try
        {
            Type providerType = FindRuntimeType("StreamingRouteProvider");
            Component provider = root.AddComponent(providerType);
            List<Vector3> input = new List<Vector3>
            {
                Vector3.zero,
                new Vector3(0f, 0f, 10f),
                new Vector3(0f, 0f, 20f)
            };
            providerType.GetMethod("SetRoute").Invoke(provider, new object[] { input });

            List<Vector3> output = new List<Vector3>();
            bool built = (bool)providerType.GetMethod("TryBuildTestRoute").Invoke(
                provider,
                new object[] { Vector3.zero, 0f, 1f, output });

            Assert.That(built, Is.True);
            Assert.That(output.Count, Is.GreaterThan(3));
            Assert.That(output[0], Is.EqualTo(Vector3.zero));
            Assert.That(output[output.Count - 1].z, Is.EqualTo(270f).Within(0.01f));
            Assert.That(
                (int)providerType.GetProperty("RenderedRoadVertexCount").GetValue(provider),
                Is.GreaterThan(0));
            Assert.That(
                (int)providerType.GetProperty("RenderedFootpathVertexCount").GetValue(provider),
                Is.GreaterThan(0));
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(root);
        }
    }

    [Test]
    public void ClearMakesRouteUnavailable()
    {
        GameObject root = new GameObject("StreamingRouteProviderTest");
        try
        {
            Type providerType = FindRuntimeType("StreamingRouteProvider");
            Component provider = root.AddComponent(providerType);
            providerType.GetMethod("AppendPoint").Invoke(provider, new object[] { Vector3.zero });
            providerType.GetMethod("AppendPoint").Invoke(provider, new object[] { Vector3.forward * 5f });
            Assert.That((bool)providerType.GetProperty("IsRouteAvailable").GetValue(provider), Is.True);

            providerType.GetMethod("Clear").Invoke(provider, null);

            Assert.That((int)providerType.GetProperty("PointCount").GetValue(provider), Is.Zero);
            Assert.That((bool)providerType.GetProperty("IsRouteAvailable").GetValue(provider), Is.False);
            Assert.That(
                (int)providerType.GetProperty("RenderedRoadVertexCount").GetValue(provider),
                Is.Zero);
            Assert.That(
                (int)providerType.GetProperty("RenderedFootpathVertexCount").GetValue(provider),
                Is.Zero);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(root);
        }
    }

    [Test]
    public void BuildsContinuationWhenStartIsNewestBufferedPoint()
    {
        GameObject root = new GameObject("StreamingRouteProviderTest");
        try
        {
            Type providerType = FindRuntimeType("StreamingRouteProvider");
            Component provider = root.AddComponent(providerType);
            List<Vector3> input = new List<Vector3>
            {
                Vector3.zero,
                new Vector3(0f, 0f, 10f),
                new Vector3(0f, 0f, 20f)
            };
            providerType.GetMethod("SetRoute").Invoke(provider, new object[] { input });

            List<Vector3> output = new List<Vector3>();
            bool built = (bool)providerType.GetMethod("TryBuildTestRoute").Invoke(
                provider,
                new object[] { input[input.Count - 1], 0f, 1f, output });

            Assert.That(built, Is.True);
            Assert.That(output.Count, Is.GreaterThan(1));
            Assert.That(output[0], Is.EqualTo(input[input.Count - 1]));
            Assert.That(output[output.Count - 1].z, Is.EqualTo(270f).Within(0.01f));
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(root);
        }
    }

    [Test]
    public void AcceptedSimulatedSnapshotsAutomaticallyBuildOwnedRoute()
    {
        GameObject stateRoot = new GameObject("StreamingStateManagerTest");
        GameObject providerRoot = new GameObject("StreamingRouteProviderTest");
        try
        {
            Type stateManagerType = FindRuntimeType("DigitalTwinStateManager");
            Type snapshotType = FindRuntimeType("TwinSnapshot");
            Type providerType = FindRuntimeType("StreamingRouteProvider");
            Component stateManager = stateRoot.AddComponent(stateManagerType);
            Component provider = providerRoot.AddComponent(providerType);

            Type jsonUtilityType = typeof(GameObject).Assembly.GetType("UnityEngine.JsonUtility");
            System.Reflection.MethodInfo fromJson = jsonUtilityType.GetMethod(
                "FromJson",
                new[] { typeof(string), typeof(Type) });
            object first = fromJson.Invoke(null, new object[] { CreateSnapshotJson(0, 0f), snapshotType });
            object second = fromJson.Invoke(null, new object[] { CreateSnapshotJson(1, 10f), snapshotType });
            Assert.That((bool)stateManagerType.GetMethod("PublishSnapshot").Invoke(
                stateManager, new[] { first }), Is.True);
            Assert.That((bool)stateManagerType.GetMethod("PublishSnapshot").Invoke(
                stateManager, new[] { second }), Is.True);
            Assert.That((int)providerType.GetProperty("PointCount").GetValue(provider), Is.EqualTo(2));

            List<Vector3> route = new List<Vector3>();
            bool built = (bool)providerType.GetMethod("TryBuildTestRoute").Invoke(
                provider,
                new object[] { new Vector3(0f, 0f, 10f), 0f, 1f, route });
            Assert.That(built, Is.True);
            Assert.That(route.Count, Is.GreaterThan(1));
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(providerRoot);
            UnityEngine.Object.DestroyImmediate(stateRoot);
        }
    }

    private static string CreateSnapshotJson(long sequence, float z)
    {
        return "{\"metadata\":{\"sequenceNumber\":" + sequence +
               ",\"sourceTimestampSeconds\":" + sequence +
               ",\"validity\":1,\"freshness\":1," +
               "\"coordinateFrame\":{\"frameId\":\"roadweave.unity.local\",\"handedness\":\"left\"," +
               "\"horizontalUnit\":\"meter\",\"angleUnit\":\"degree\",\"xAxis\":\"right\"," +
               "\"yAxis\":\"up\",\"zAxis\":\"forward\"}," +
               "\"session\":{\"sourceId\":\"simulation\",\"sessionId\":\"simulation-1\"," +
               "\"sourceKind\":2,\"status\":2}}," +
               "\"ego\":{\"position\":{\"x\":0,\"y\":0,\"z\":" + z + "},\"validity\":1}," +
               "\"vehicle\":{\"validity\":1},\"wheels\":{\"validity\":1},\"actors\":[]}";
    }

    [Test]
    public void RoadWeaveSceneHasScenarioModelsAndDisabledLegacyManagers()
    {
        string projectRoot = Directory.GetParent(Application.dataPath).FullName;
        string scene = File.ReadAllText(Path.Combine(projectRoot, "Assets/Scenes/RoadWeave.unity"));

        StringAssert.Contains(
            "carPrefab: {fileID: 8771091787928289351, guid: b5e6d5ef0a7ab46c88e6bbcdc00b8f1d, type: 3}",
            scene);
        StringAssert.Contains(
            "truckPrefab: {fileID: -7013847372160066528, guid: ec7ab99961d50432188096bcb0c0a749, type: 3}",
            scene);
        StringAssert.Contains(
            "busPrefab: {fileID: 919132149155446097, guid: e1bcf5f5224fc4a098b82022cf83c17f, type: 3}",
            scene);
        Assert.That(Regex.IsMatch(
            scene,
            @"m_GameObject: \{fileID: 837622277\}\s+m_Enabled: 0[\s\S]*?Assembly-CSharp::DigitalTwinDataManager"),
            Is.True);
        Assert.That(Regex.IsMatch(
            scene,
            @"m_GameObject: \{fileID: 1688179358\}\s+m_Enabled: 0[\s\S]*?Assembly-CSharp::UIManager"),
            Is.True);

        Assert.That(
            File.ReadAllText(Path.Combine(projectRoot, "Assets/Prefabs/ReplayActors/Cars/bmw_m5_g90_2024.glb.meta")),
            Does.Contain("guid: b5e6d5ef0a7ab46c88e6bbcdc00b8f1d"));
        Assert.That(
            File.ReadAllText(Path.Combine(projectRoot, "Assets/Prefabs/ReplayActors/Trucks/truck/source/Truck.glb.meta")),
            Does.Contain("guid: ec7ab99961d50432188096bcb0c0a749"));
        Assert.That(
            File.ReadAllText(Path.Combine(projectRoot,
                "Assets/Prefabs/ReplayActors/Buses/etalon-a079-reworked/source/sourse/etalon_a079_2.0.fbx.meta")),
            Does.Contain("guid: e1bcf5f5224fc4a098b82022cf83c17f"));
    }

    private static Type FindRuntimeType(string name)
    {
        Type type = Type.GetType(name + ", Assembly-CSharp");
        Assert.That(type, Is.Not.Null, "Could not find runtime type " + name);
        return type;
    }
}

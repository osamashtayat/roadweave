using System;
using System.Collections.Generic;
using System.Reflection;
using System.Threading;
using NUnit.Framework;
using UnityEngine;

public class DigitalTwinCoreTests
{
    private readonly List<GameObject> roots = new List<GameObject>();
    private Type snapshotType;
    private Type metadataType;
    private Type sessionType;
    private Type actorStateType;

    [SetUp]
    public void SetUp()
    {
        snapshotType = RuntimeType("TwinSnapshot");
        metadataType = RuntimeType("TwinSnapshotMetadata");
        sessionType = RuntimeType("TwinSessionInfo");
        actorStateType = RuntimeType("TwinActorState");
    }

    [TearDown]
    public void TearDown()
    {
        for (int index = roots.Count - 1; index >= 0; index--)
            if (roots[index] != null) UnityEngine.Object.DestroyImmediate(roots[index]);
        roots.Clear();
    }

    [Test]
    public void StateManagerRejectsOutOfOrderSequence()
    {
        Component manager = Add("DigitalTwinStateManager", "State");
        Assert.That(Publish(manager, Snapshot(2)), Is.True);
        Assert.That(Publish(manager, Snapshot(1)), Is.False);
        Assert.That((float)manager.GetType().GetProperty("CurrentTime").GetValue(manager), Is.EqualTo(2f));
    }

    [Test]
    public void ConnectedSourceSwitchesCleanly()
    {
        Component manager = Add("DigitalTwinStateManager", "State");
        Component simulated = Add("SimulatedStreamTwinSource", "Simulated");
        Component live = Add("LiveSensorTwinSource", "Live");

        Assert.That(manager.GetType().GetProperty("ConnectedSource").GetValue(manager), Is.SameAs(live));
        Assert.That(manager.GetType().GetProperty("SessionControl").GetValue(manager), Is.SameAs(live));

        MethodInfo connect = manager.GetType().GetMethod("ConnectSource");
        connect.Invoke(manager, new object[] { simulated });
        Assert.That(manager.GetType().GetProperty("ConnectedSource").GetValue(manager), Is.SameAs(simulated));
        Assert.That(manager.GetType().GetProperty("CurrentSnapshot").GetValue(manager), Is.Null);
        simulated.GetType().GetMethod("AcceptSnapshot").Invoke(simulated, new[] { Snapshot(0) });
        Assert.That(manager.GetType().GetProperty("CurrentSnapshot").GetValue(manager), Is.Not.Null);
    }

    [Test]
    public void StructurallyEmptyLiveJsonIsRejected()
    {
        Component manager = Add("DigitalTwinStateManager", "State");
        Component live = Add("LiveSensorTwinSource", "Live");
        Assert.That((bool)live.GetType().GetMethod("AcceptJsonMessage")
            .Invoke(live, new object[] { "{}" }), Is.False);
        Assert.That(manager.GetType().GetProperty("CurrentSnapshot").GetValue(manager), Is.Null);
    }

    [Test]
    public void PausedPushSourceRejectsUpdatesAndItsLastStateAgesStale()
    {
        Component manager = Add("DigitalTwinStateManager", "State");
        Component simulated = Add("SimulatedStreamTwinSource", "Simulated");
        simulated.GetType().GetMethod("StartSession").Invoke(simulated, null);
        Assert.That((bool)simulated.GetType().GetMethod("AcceptSnapshot")
            .Invoke(simulated, new[] { Snapshot(0, 0.001f) }), Is.True);
        simulated.GetType().GetMethod("PauseSession").Invoke(simulated, null);
        Assert.That((bool)simulated.GetType().GetMethod("AcceptSnapshot")
            .Invoke(simulated, new[] { Snapshot(1, 0.001f) }), Is.False);
        Thread.Sleep(10);
        InvokePrivate(manager, "Update");
        Assert.That((bool)manager.GetType().GetProperty("IsFresh").GetValue(manager), Is.False);
    }

    [Test]
    public void SourceDeclaredStaleSnapshotRemainsStale()
    {
        Component manager = Add("DigitalTwinStateManager", "State");
        object snapshot = Snapshot(0);
        object metadata = snapshotType.GetField("metadata").GetValue(snapshot);
        SetEnum(metadata, metadataType, "freshness", "TwinDataFreshness", "Stale");
        Assert.That(Publish(manager, snapshot), Is.True);
        Assert.That((bool)manager.GetType().GetProperty("IsFresh").GetValue(manager), Is.False);
    }

    [Test]
    public void ReplacingSimulatedTemplatesKeepsSequenceMonotonic()
    {
        Component manager = Add("DigitalTwinStateManager", "State");
        Add("SimulatedStreamTwinSource", "Simulated");
        Component player = Add("SimulatedLiveTwinPlayer", "Player");
        Array templates = Array.CreateInstance(snapshotType, 1);
        templates.SetValue(Snapshot(0), 0);
        player.GetType().GetMethod("SetTemplates").Invoke(player, new object[] { templates });
        Assert.That((bool)player.GetType().GetMethod("StartPlayer").Invoke(player, null), Is.True);
        player.GetType().GetMethod("Tick").Invoke(player, new object[] { 0f });

        Array replacements = Array.CreateInstance(snapshotType, 1);
        replacements.SetValue(Snapshot(0), 0);
        player.GetType().GetMethod("SetTemplates").Invoke(player, new object[] { replacements });
        player.GetType().GetMethod("Tick").Invoke(player, new object[] { 0.1f });
        object current = manager.GetType().GetProperty("CurrentSnapshot").GetValue(manager);
        object currentMetadata = snapshotType.GetField("metadata").GetValue(current);
        Assert.That((long)metadataType.GetField("sequenceNumber").GetValue(currentMetadata), Is.EqualTo(1L));
    }

    [Test]
    public void ReplayValidationRejectsNullAndOutOfOrderEgoFrames()
    {
        Type egoType = RuntimeType("EgoReplayFrame");
        object package = MinimalReplayPackage();
        Array nullFrames = Array.CreateInstance(egoType, 1);
        nullFrames.SetValue(null, 0);
        Set(package, package.GetType(), "egoFrames", nullFrames);
        MethodInfo validate = RuntimeType("ReplaySnapshotSampler").GetMethod("ValidatePackage");
        object[] nullArguments = { package, null, null };
        Assert.That((bool)validate.Invoke(null, nullArguments), Is.False);

        object later = EgoFrame(egoType, 1f);
        object earlier = EgoFrame(egoType, 0f);
        Array reversed = Array.CreateInstance(egoType, 2);
        reversed.SetValue(later, 0);
        reversed.SetValue(earlier, 1);
        Set(package, package.GetType(), "egoFrames", reversed);
        object[] orderArguments = { package, null, null };
        Assert.That((bool)validate.Invoke(null, orderArguments), Is.False);
    }

    [Test]
    public void CoordinateContractRejectsNonMeterAndUnknownUnits()
    {
        MethodInfo convert = RuntimeType("TwinCoordinateFrameService").GetMethod("TryCreateCanonicalFrame");
        object[] feet = { "x=right, y=up, z=forward; units=feet", null, null };
        object[] centimeters = { "x=right, y=up, z=forward; units=centimeters", null, null };
        object[] unknown = { "x=right, y=up, z=forward", null, null };
        object[] meters = { "x=right, y=up, z=forward; units=meters", null, null };
        Assert.That((bool)convert.Invoke(null, feet), Is.False);
        Assert.That((bool)convert.Invoke(null, centimeters), Is.False);
        Assert.That((bool)convert.Invoke(null, unknown), Is.False);
        Assert.That((bool)convert.Invoke(null, meters), Is.True, meters[2] as string);
    }

    [Test]
    public void ReplayPackageCanOnlyBelongToActiveReplaySource()
    {
        Component manager = Add("DigitalTwinStateManager", "State");
        Component replay = Add("NuScenesReplayController", "Replay");
        object package = Activator.CreateInstance(RuntimeType("ReplayPackage"));
        MethodInfo attach = manager.GetType().GetMethod("AttachReplayPackage");
        Assert.That((bool)attach.Invoke(manager, new[] { package, replay }), Is.True);

        Add("LiveSensorTwinSource", "Live");
        Assert.That(manager.GetType().GetProperty("Package").GetValue(manager), Is.Null);
        Assert.That((bool)attach.Invoke(manager, new[] { package, replay }), Is.False);
        Assert.That(manager.GetType().GetProperty("Package").GetValue(manager), Is.Null);
    }

    [Test]
    public void ReplayStartAndRestartReconnectAfterLiveSourceSwitch()
    {
        Component manager = Add("DigitalTwinStateManager", "State");
        Component replay = Add("NuScenesReplayController", "Replay");
        object package = MinimalReplayPackage();
        replay.GetType().GetProperty("Package").SetValue(replay, package);
        replay.GetType().GetField("coordinateFrame", BindingFlags.Instance | BindingFlags.NonPublic)
            .SetValue(replay, RuntimeType("TwinCoordinateFrame").GetMethod("UnityLocal")
                .Invoke(null, new object[] { "test" }));

        Add("LiveSensorTwinSource", "LiveOne");
        replay.GetType().GetMethod("StartDrive").Invoke(replay, null);
        Assert.That(manager.GetType().GetProperty("ConnectedSource").GetValue(manager), Is.SameAs(replay));
        Assert.That(manager.GetType().GetProperty("Package").GetValue(manager), Is.SameAs(package));

        Add("LiveSensorTwinSource", "LiveTwo");
        replay.GetType().GetMethod("RestartReplay").Invoke(replay, new object[] { true });
        Assert.That(manager.GetType().GetProperty("ConnectedSource").GetValue(manager), Is.SameAs(replay));
        Assert.That(manager.GetType().GetProperty("CurrentSnapshot").GetValue(manager), Is.Not.Null);
    }

    [Test]
    public void EgoPresentationHidesOnUnavailableAndRecoversForNewSource()
    {
        Component manager = Add("DigitalTwinStateManager", "State");
        GameObject ego = GameObject.CreatePrimitive(PrimitiveType.Cube);
        roots.Add(ego);
        Component view = ego.AddComponent(RuntimeType("EgoVehicleReplayView"));
        Assert.That(Publish(manager, Snapshot(0)), Is.True);
        InvokePrivate(view, "LateUpdate");
        Assert.That(ego.GetComponent<Renderer>().enabled, Is.True);

        Component live = Add("LiveSensorTwinSource", "Live");
        InvokePrivate(view, "LateUpdate");
        Assert.That(ego.GetComponent<Renderer>().enabled, Is.False);
        live.GetType().GetMethod("AcceptSnapshot").Invoke(live, new[] { Snapshot(1) });
        InvokePrivate(view, "LateUpdate");
        Assert.That(ego.GetComponent<Renderer>().enabled, Is.True);
    }

    [Test]
    public void WheelIntegrationPreservesLargeDoubleTimestampDeltas()
    {
        Component manager = Add("DigitalTwinStateManager", "State");
        GameObject wheelObject = new GameObject("Wheel");
        roots.Add(wheelObject);
        GameObject viewObject = new GameObject("WheelView");
        roots.Add(viewObject);
        Component view = viewObject.AddComponent(RuntimeType("WheelReplayView"));
        view.GetType().GetField("frontLeftWheel", BindingFlags.Instance | BindingFlags.NonPublic)
            .SetValue(view, wheelObject.transform);
        InvokePrivate(view, "Awake");

        object first = Snapshot(0);
        Set(snapshotType.GetField("metadata").GetValue(first), metadataType,
            "sourceTimestampSeconds", 1700000000d);
        SetWheelRpm(first, 60f);
        Assert.That(Publish(manager, first), Is.True);
        InvokePrivate(view, "LateUpdate");

        object second = Snapshot(1);
        Set(snapshotType.GetField("metadata").GetValue(second), metadataType,
            "sourceTimestampSeconds", 1700000000.25d);
        SetWheelRpm(second, 60f);
        Assert.That(Publish(manager, second), Is.True);
        InvokePrivate(view, "LateUpdate");
        Assert.That(Mathf.DeltaAngle(90f, wheelObject.transform.localEulerAngles.x), Is.EqualTo(0f).Within(0.2f));
    }

    [Test]
    public void StreamingSnapshotAndActorsBecomeStaleTogether()
    {
        Component manager = Add("DigitalTwinStateManager", "State");
        object actor = Actor("actor-1");
        object snapshot = Snapshot(0, 0.001f, actor);
        Assert.That(Publish(manager, snapshot), Is.True);
        Thread.Sleep(10);
        manager.GetType().GetMethod("Update", BindingFlags.Instance | BindingFlags.NonPublic)
            .Invoke(manager, null);

        Assert.That((bool)manager.GetType().GetProperty("IsFresh").GetValue(manager), Is.False);
        object actors = manager.GetType().GetProperty("Actors").GetValue(manager);
        object firstActor = ((Array)actors).GetValue(0);
        Assert.That(actorStateType.GetField("freshness").GetValue(firstActor).ToString(), Is.EqualTo("Stale"));
    }

    [Test]
    public void ReplaySamplerPopulatesActorObservationClock()
    {
        Type packageType = RuntimeType("ReplayPackage");
        Type replayVectorType = RuntimeType("ReplayVector3");
        Type egoFrameType = RuntimeType("EgoReplayFrame");
        Type trackType = RuntimeType("ActorReplayTrack");
        Type actorFrameType = RuntimeType("ActorReplayFrame");
        object package = Activator.CreateInstance(packageType);
        Set(package, packageType, "schemaVersion", 1);
        Set(package, packageType, "durationSeconds", 1f);
        Set(package, packageType, "sampleIntervalSeconds", 0.5f);
        Set(package, packageType, "coordinateSystem", "Unity local: x=right, y=up, z=forward; origin=first CAN pose");

        object ego = Activator.CreateInstance(egoFrameType);
        Set(ego, egoFrameType, "time", 0f);
        Set(ego, egoFrameType, "position", ReplayVector(replayVectorType, 0f, 0f, 0f));
        Array egoFrames = Array.CreateInstance(egoFrameType, 1);
        egoFrames.SetValue(ego, 0);
        Set(package, packageType, "egoFrames", egoFrames);

        object from = Activator.CreateInstance(actorFrameType);
        Set(from, actorFrameType, "time", 0f);
        Set(from, actorFrameType, "position", ReplayVector(replayVectorType, 0f, 0f, 0f));
        object to = Activator.CreateInstance(actorFrameType);
        Set(to, actorFrameType, "time", 1f);
        Set(to, actorFrameType, "position", ReplayVector(replayVectorType, 0f, 0f, 1f));
        Array frames = Array.CreateInstance(actorFrameType, 2);
        frames.SetValue(from, 0);
        frames.SetValue(to, 1);
        object track = Activator.CreateInstance(trackType);
        Set(track, trackType, "id", "actor-1");
        Set(track, trackType, "category", "vehicle.car");
        Set(track, trackType, "prefabType", "car");
        Set(track, trackType, "size", ReplayVector(replayVectorType, 1.8f, 1.5f, 4.2f));
        Set(track, trackType, "frames", frames);
        Array tracks = Array.CreateInstance(trackType, 1);
        tracks.SetValue(track, 0);
        Set(package, packageType, "actors", tracks);

        object coordinate = RuntimeType("TwinCoordinateFrame").GetMethod("UnityLocal")
            .Invoke(null, new object[] { "test" });
        object session = Session("Replay", "Ready", "replay-session");
        object sampled = RuntimeType("ReplaySnapshotSampler").GetMethod("Sample")
            .Invoke(null, new[] { package, coordinate, (object)0.5f, (object)42L, session });
        object sampledActor = ((Array)snapshotType.GetField("actors").GetValue(sampled)).GetValue(0);

        Assert.That((double)actorStateType.GetField("observationTimestampSeconds").GetValue(sampledActor),
            Is.EqualTo(0.5d).Within(0.0001d));
        Assert.That((long)actorStateType.GetField("observationSequenceNumber").GetValue(sampledActor), Is.EqualTo(42L));
        Assert.That(actorStateType.GetField("freshness").GetValue(sampledActor).ToString(), Is.EqualTo("Fresh"));
    }

    [Test]
    public void GpsConvertsToCanonicalEastUpNorth()
    {
        Type gpsType = RuntimeType("TwinGpsCoordinate");
        Type originType = RuntimeType("TwinGeoReference");
        object origin = Activator.CreateInstance(originType, new object[] { 0d, 0d, 0d });
        object gps = Activator.CreateInstance(gpsType, new object[] { 0.0000090437d, 0.00000898315d, 1d });
        object[] arguments = { gps, origin, null, null };
        bool converted = (bool)RuntimeType("TwinCoordinateFrameService")
            .GetMethod("TryGpsToCanonicalPosition").Invoke(null, arguments);
        Vector3 position = (Vector3)arguments[2];

        Assert.That(converted, Is.True, arguments[3] as string);
        Assert.That(position.x, Is.EqualTo(1f).Within(0.03f));
        Assert.That(position.y, Is.EqualTo(1f).Within(0.03f));
        Assert.That(position.z, Is.EqualTo(1f).Within(0.03f));
    }

    [Test]
    public void UdpTransportHandsJsonToLiveSourceOnMainThread()
    {
        Component manager = Add("DigitalTwinStateManager", "State");
        Add("LiveSensorTwinSource", "Live");
        Component transport = Add("LiveJsonUdpTransport", "Transport");
        string json = JsonUtility.ToJson(Snapshot(0));

        Assert.That((bool)transport.GetType().GetMethod("SubmitReceivedMessage")
            .Invoke(transport, new object[] { json }), Is.True);
        Assert.That((int)transport.GetType().GetMethod("PublishPendingMessages")
            .Invoke(transport, null), Is.EqualTo(1));
        Assert.That(manager.GetType().GetProperty("CurrentSnapshot").GetValue(manager), Is.Not.Null);
    }

    private Component Add(string typeName, string objectName)
    {
        GameObject root = new GameObject(objectName);
        roots.Add(root);
        return root.AddComponent(RuntimeType(typeName));
    }

    private bool Publish(Component manager, object snapshot)
    {
        return (bool)manager.GetType().GetMethod("PublishSnapshot").Invoke(manager, new[] { snapshot });
    }

    private object Snapshot(long sequence, float staleAfter = 1f, params object[] actors)
    {
        object snapshot = Activator.CreateInstance(snapshotType);
        object metadata = Activator.CreateInstance(metadataType);
        Set(metadata, metadataType, "sequenceNumber", sequence);
        Set(metadata, metadataType, "sourceTimestampSeconds", (double)sequence);
        Set(metadata, metadataType, "staleAfterSeconds", staleAfter);
        SetEnum(metadata, metadataType, "validity", "TwinDataValidity", "Valid");
        SetEnum(metadata, metadataType, "freshness", "TwinDataFreshness", "Fresh");
        Set(metadata, metadataType, "coordinateFrame", RuntimeType("TwinCoordinateFrame")
            .GetMethod("UnityLocal").Invoke(null, new object[] { "test" }));
        Set(metadata, metadataType, "session", Session("SimulatedStream", "Running", "test-session"));
        Array actorArray = Array.CreateInstance(actorStateType, actors.Length);
        for (int index = 0; index < actors.Length; index++) actorArray.SetValue(actors[index], index);
        Set(snapshot, snapshotType, "metadata", metadata);
        Set(snapshot, snapshotType, "actors", actorArray);
        SetEnum(snapshotType.GetField("ego").GetValue(snapshot), RuntimeType("TwinEgoState"),
            "validity", "TwinDataValidity", "Valid");
        SetEnum(snapshotType.GetField("vehicle").GetValue(snapshot), RuntimeType("TwinVehicleState"),
            "validity", "TwinDataValidity", "Valid");
        SetEnum(snapshotType.GetField("wheels").GetValue(snapshot), RuntimeType("TwinWheelState"),
            "validity", "TwinDataValidity", "Valid");
        return snapshot;
    }

    private object Actor(string id)
    {
        object actor = Activator.CreateInstance(actorStateType);
        Set(actor, actorStateType, "id", id);
        SetEnum(actor, actorStateType, "semanticClass", "TwinActorClass", "Car");
        SetEnum(actor, actorStateType, "motionState", "TwinActorMotionState", "Moving");
        SetEnum(actor, actorStateType, "validity", "TwinDataValidity", "Valid");
        SetEnum(actor, actorStateType, "freshness", "TwinDataFreshness", "Fresh");
        return actor;
    }

    private object Session(string sourceKind, string status, string id)
    {
        object session = Activator.CreateInstance(sessionType);
        Set(session, sessionType, "sourceId", "test");
        Set(session, sessionType, "sessionId", id);
        SetEnum(session, sessionType, "sourceKind", "TwinSourceKind", sourceKind);
        SetEnum(session, sessionType, "status", "TwinSessionStatus", status);
        return session;
    }

    private static object ReplayVector(Type type, float x, float y, float z)
    {
        object value = Activator.CreateInstance(type);
        Set(value, type, "x", x);
        Set(value, type, "y", y);
        Set(value, type, "z", z);
        return value;
    }

    private object MinimalReplayPackage()
    {
        Type packageType = RuntimeType("ReplayPackage");
        Type egoType = RuntimeType("EgoReplayFrame");
        object package = Activator.CreateInstance(packageType);
        Set(package, packageType, "schemaVersion", 1);
        Set(package, packageType, "durationSeconds", 1f);
        Set(package, packageType, "sampleIntervalSeconds", 0.1f);
        Set(package, packageType, "coordinateSystem",
            "Unity local: x=right, y=up, z=forward; origin=first CAN pose");
        object ego = Activator.CreateInstance(egoType);
        Set(ego, egoType, "time", 0f);
        Set(ego, egoType, "position", ReplayVector(RuntimeType("ReplayVector3"), 0f, 0f, 0f));
        Array frames = Array.CreateInstance(egoType, 1);
        frames.SetValue(ego, 0);
        Set(package, packageType, "egoFrames", frames);
        return package;
    }

    private object EgoFrame(Type egoType, float time)
    {
        object ego = Activator.CreateInstance(egoType);
        Set(ego, egoType, "time", time);
        Set(ego, egoType, "position", ReplayVector(RuntimeType("ReplayVector3"), 0f, 0f, time));
        return ego;
    }

    private void SetWheelRpm(object snapshot, float rpm)
    {
        object wheels = snapshotType.GetField("wheels").GetValue(snapshot);
        Type wheelType = RuntimeType("TwinWheelState");
        Set(wheels, wheelType, "frontLeftRpm", rpm);
    }

    private static void InvokePrivate(Component component, string method) =>
        component.GetType().GetMethod(method, BindingFlags.Instance | BindingFlags.NonPublic)
            .Invoke(component, null);

    private static void Set(object target, Type type, string field, object value) =>
        type.GetField(field).SetValue(target, value);

    private static void SetEnum(object target, Type type, string field, string enumName, string value) =>
        Set(target, type, field, Enum.Parse(RuntimeType(enumName), value));

    private static Type RuntimeType(string name)
    {
        Type type = Type.GetType(name + ", Assembly-CSharp");
        Assert.That(type, Is.Not.Null, "Could not find runtime type " + name);
        return type;
    }
}

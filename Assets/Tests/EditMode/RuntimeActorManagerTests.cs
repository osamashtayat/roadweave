using System;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;

public class RuntimeActorManagerTests
{
    private GameObject stateObject;
    private GameObject managerObject;
    private Component stateManager;
    private Component actorManager;
    private Type stateManagerType;
    private Type actorManagerType;
    private Type snapshotType;
    private Type metadataType;
    private Type sessionType;
    private Type actorStateType;

    [SetUp]
    public void SetUp()
    {
        stateManagerType = FindRuntimeType("DigitalTwinStateManager");
        actorManagerType = FindRuntimeType("ReplayActorManager");
        snapshotType = FindRuntimeType("TwinSnapshot");
        metadataType = FindRuntimeType("TwinSnapshotMetadata");
        sessionType = FindRuntimeType("TwinSessionInfo");
        actorStateType = FindRuntimeType("TwinActorState");

        stateObject = new GameObject("TwinState-Test");
        stateManager = stateObject.AddComponent(stateManagerType);
        managerObject = new GameObject("ActorManager-Test");
        actorManager = managerObject.AddComponent(actorManagerType);
    }

    [TearDown]
    public void TearDown()
    {
        if (managerObject != null) UnityEngine.Object.DestroyImmediate(managerObject);
        if (stateObject != null) UnityEngine.Object.DestroyImmediate(stateObject);
    }

    [Test]
    public void FreshSnapshotsUpsertAndRemoveActorsByStableId()
    {
        Assert.That(Publish(Snapshot(0,
            Actor("car-1", new Vector3(1f, 0f, 5f), "Car"),
            Actor("truck-1", new Vector3(-2f, 0f, 8f), "Truck"))), Is.True);
        Assert.That(RuntimeActorCount(), Is.EqualTo(2));
        Assert.That(TryGetActor("car-1", out GameObject car), Is.True);
        Assert.That(car.transform.localPosition, Is.EqualTo(new Vector3(1f, 0f, 5f)));

        Assert.That(Publish(Snapshot(1,
            Actor("car-1", new Vector3(3f, 0f, 7f), "Car"))), Is.True);
        Assert.That(RuntimeActorCount(), Is.EqualTo(1));
        Assert.That(TryGetActor("truck-1", out _), Is.False);
        Assert.That(TryGetActor("car-1", out car), Is.True);
        Assert.That(car.transform.localPosition, Is.EqualTo(new Vector3(3f, 0f, 7f)));
    }

    [Test]
    public void StaleSnapshotClearsActors()
    {
        actorManagerType.GetMethod("SynchronizeSnapshot")
            .Invoke(actorManager, new[] { Snapshot(0, Actor("car-1", Vector3.forward, "Car")) });
        object staleEmpty = Snapshot(1);
        object metadata = snapshotType.GetField("metadata").GetValue(staleEmpty);
        SetEnum(metadata, metadataType, "freshness", "TwinDataFreshness", "Stale");
        actorManagerType.GetMethod("SynchronizeSnapshot").Invoke(actorManager, new[] { staleEmpty });

        Assert.That(RuntimeActorCount(), Is.EqualTo(0));
        Assert.That(TryGetActor("car-1", out _), Is.False);
    }

    [Test]
    public void SourceSwitchClearsOldRuntimeActors()
    {
        Assert.That(Publish(Snapshot(0, Actor("car-1", Vector3.forward, "Car"))), Is.True);
        Assert.That(RuntimeActorCount(), Is.EqualTo(1));
        GameObject sourceObject = new GameObject("LiveSource-Test");
        sourceObject.AddComponent(FindRuntimeType("LiveSensorTwinSource"));
        Assert.That(RuntimeActorCount(), Is.EqualTo(0));
        UnityEngine.Object.DestroyImmediate(sourceObject);
    }

    [Test]
    public void StaleActorObservationIsRemovedAndNeverExtrapolated()
    {
        object actor = Actor("car-1", Vector3.forward, "Car");
        actorManagerType.GetMethod("SynchronizeSnapshot").Invoke(actorManager, new[] { Snapshot(0, actor) });
        Assert.That(RuntimeActorCount(), Is.EqualTo(1));
        SetEnum(actor, actorStateType, "freshness", "TwinDataFreshness", "Stale");
        actorManagerType.GetMethod("SynchronizeSnapshot").Invoke(actorManager, new[] { Snapshot(1, actor) });
        Assert.That(RuntimeActorCount(), Is.EqualTo(0));
    }

    [Test]
    public void ActorsCreatedWhileHiddenReappearWithRuntimeRoot()
    {
        actorManagerType.GetMethod("SetRuntimeActorsVisible").Invoke(actorManager, new object[] { false });
        actorManagerType.GetMethod("SynchronizeSnapshot")
            .Invoke(actorManager, new[] { Snapshot(0, Actor("ped-1", Vector3.forward, "Pedestrian")) });
        Assert.That(TryGetActor("ped-1", out GameObject pedestrian), Is.True);
        Assert.That(pedestrian.activeSelf, Is.True);
        Assert.That(pedestrian.activeInHierarchy, Is.False);

        actorManagerType.GetMethod("SetRuntimeActorsVisible").Invoke(actorManager, new object[] { true });
        Assert.That(pedestrian.activeInHierarchy, Is.True);
    }

    [Test]
    public void GroundedPedestrianPlacesItsFeetOnTheEgoRoadPlane()
    {
        actorManagerType.GetField(
                "groundRoadActorsToEgoPlane",
                BindingFlags.Instance | BindingFlags.NonPublic)
            .SetValue(actorManager, true);

        Assert.That(Publish(Snapshot(
            0,
            Actor("ped-grounded", new Vector3(0f, 10f, 12f), "Pedestrian"))), Is.True);
        Assert.That(TryGetActor("ped-grounded", out GameObject pedestrian), Is.True);

        // The visual is centered on the actor root. Half the 1.8 m actor
        // height plus the manager's 0.015 m clearance places its feet at y=0.
        Assert.That(pedestrian.transform.localPosition.y, Is.EqualTo(0.915f).Within(0.001f));
    }

    private object Snapshot(long sequence, params object[] actors)
    {
        object snapshot = Activator.CreateInstance(snapshotType);
        object metadata = Activator.CreateInstance(metadataType);
        object session = Activator.CreateInstance(sessionType);
        metadataType.GetField("sequenceNumber").SetValue(metadata, sequence);
        metadataType.GetField("sourceTimestampSeconds").SetValue(metadata, (double)sequence);
        metadataType.GetField("receiptTimestampSeconds").SetValue(metadata, (double)Time.realtimeSinceStartup);
        SetEnum(metadata, metadataType, "validity", "TwinDataValidity", "Valid");
        SetEnum(metadata, metadataType, "freshness", "TwinDataFreshness", "Fresh");

        Type coordinateType = FindRuntimeType("TwinCoordinateFrame");
        object coordinate = coordinateType.GetMethod("UnityLocal")
            .Invoke(null, new object[] { "test" });
        metadataType.GetField("coordinateFrame").SetValue(metadata, coordinate);

        sessionType.GetField("sourceId").SetValue(session, "test-source");
        sessionType.GetField("sessionId").SetValue(session, "test-session");
        SetEnum(session, sessionType, "sourceKind", "TwinSourceKind", "SimulatedStream");
        SetEnum(session, sessionType, "status", "TwinSessionStatus", "Running");
        SetEnum(session, sessionType, "capabilities", "TwinSourceCapabilities", "SurroundingActors");
        metadataType.GetField("session").SetValue(metadata, session);

        Array actorArray = Array.CreateInstance(actorStateType, actors.Length);
        for (int index = 0; index < actors.Length; index++) actorArray.SetValue(actors[index], index);
        snapshotType.GetField("metadata").SetValue(snapshot, metadata);
        snapshotType.GetField("actors").SetValue(snapshot, actorArray);
        SetEnum(snapshotType.GetField("ego").GetValue(snapshot), FindRuntimeType("TwinEgoState"),
            "validity", "TwinDataValidity", "Valid");
        SetEnum(snapshotType.GetField("vehicle").GetValue(snapshot), FindRuntimeType("TwinVehicleState"),
            "validity", "TwinDataValidity", "Valid");
        SetEnum(snapshotType.GetField("wheels").GetValue(snapshot), FindRuntimeType("TwinWheelState"),
            "validity", "TwinDataValidity", "Valid");
        return snapshot;
    }

    private object Actor(string id, Vector3 position, string semanticClass)
    {
        object actor = Activator.CreateInstance(actorStateType);
        actorStateType.GetField("id").SetValue(actor, id);
        actorStateType.GetField("sourceClass").SetValue(actor, semanticClass);
        actorStateType.GetField("position").SetValue(actor, position);
        actorStateType.GetField("velocityMetersPerSecond").SetValue(actor, Vector3.forward);
        actorStateType.GetField("dimensionsMeters").SetValue(actor,
            semanticClass == "Pedestrian" ? new Vector3(0.7f, 1.8f, 0.7f) : new Vector3(1.8f, 1.5f, 4.2f));
        actorStateType.GetField("confidence").SetValue(actor, 1f);
        SetEnum(actor, actorStateType, "semanticClass", "TwinActorClass", semanticClass);
        SetEnum(actor, actorStateType, "motionState", "TwinActorMotionState", "Moving");
        SetEnum(actor, actorStateType, "validity", "TwinDataValidity", "Valid");
        SetEnum(actor, actorStateType, "freshness", "TwinDataFreshness", "Fresh");
        return actor;
    }

    private bool Publish(object snapshot)
    {
        return (bool)stateManagerType.GetMethod("PublishSnapshot").Invoke(stateManager, new[] { snapshot });
    }

    private int RuntimeActorCount()
    {
        return (int)actorManagerType.GetProperty("RuntimeActorCount").GetValue(actorManager);
    }

    private bool TryGetActor(string stableId, out GameObject actor)
    {
        object[] arguments = { stableId, null };
        bool found = (bool)actorManagerType.GetMethod("TryGetActor").Invoke(actorManager, arguments);
        actor = arguments[1] as GameObject;
        return found;
    }

    private static void SetEnum(object target, Type targetType, string fieldName, string enumTypeName, string value)
    {
        Type enumType = FindRuntimeType(enumTypeName);
        targetType.GetField(fieldName).SetValue(target, Enum.Parse(enumType, value));
    }

    private static Type FindRuntimeType(string name)
    {
        Type type = Type.GetType(name + ", Assembly-CSharp");
        Assert.That(type, Is.Not.Null, "Could not find runtime type " + name);
        return type;
    }
}

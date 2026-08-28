using System;
using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;

public class TestLabSafetyTests
{
    [Test]
    public void MlDecisionJsonCarriesWeatherModelOutput()
    {
        const string json =
            "{\"valid\":true,\"executedAction\":\"KEEP\"," +
            "\"weatherContext\":\"Fog\",\"weatherModelUsed\":true," +
            "\"weatherSpeedFactor\":0.41,\"weatherTargetSpeedMps\":5.7}";

        Type decisionType = FindRuntimeType("TestLabMlDecision");
        object decision = JsonUtility.FromJson(json, decisionType);

        Assert.That((bool)decisionType.GetField("weatherModelUsed").GetValue(decision), Is.True);
        Assert.That((string)decisionType.GetField("weatherContext").GetValue(decision), Is.EqualTo("Fog"));
        Assert.That(
            (float)decisionType.GetField("weatherSpeedFactor").GetValue(decision),
            Is.EqualTo(0.41f).Within(0.001f)
        );
        Assert.That(
            (float)decisionType.GetField("weatherTargetSpeedMps").GetValue(decision),
            Is.EqualTo(5.7f).Within(0.001f)
        );
    }

    [Test]
    public void MlDecisionJsonCarriesVirtualSensorReliability()
    {
        const string json =
            "{\"valid\":true,\"executedAction\":\"KEEP\"," +
            "\"overallSensorReliability\":0.73,\"sensorSafetyMode\":\"CAUTIOUS\"," +
            "\"sensorReliability\":[{\"sensorId\":\"front_camera\"," +
            "\"sensorType\":\"CAMERA\",\"reliability\":0.44,\"status\":\"DEGRADED\"}]}";

        Type decisionType = FindRuntimeType("TestLabMlDecision");
        object decision = JsonUtility.FromJson(json, decisionType);

        Assert.That(
            (float)decisionType.GetField("overallSensorReliability").GetValue(decision),
            Is.EqualTo(0.73f).Within(0.001f)
        );
        Assert.That(
            (string)decisionType.GetField("sensorSafetyMode").GetValue(decision),
            Is.EqualTo("CAUTIOUS")
        );
        Array sensors = (Array)decisionType.GetField("sensorReliability").GetValue(decision);
        Assert.That(sensors.Length, Is.EqualTo(1));
        object camera = sensors.GetValue(0);
        Assert.That(
            (string)camera.GetType().GetField("status").GetValue(camera),
            Is.EqualTo("DEGRADED")
        );
    }

    [Test]
    public void WeatherRangeFactorChangesActiveSensorRange()
    {
        GameObject root = new GameObject("SensorTest");
        try
        {
            Type sensorType = FindRuntimeType("SimulatedVehicleSensorSuite");
            Component sensor = root.AddComponent(sensorType);
            sensorType.GetMethod("SetRangeFactor").Invoke(sensor, new object[] { 0.5f });
            float range = (float)sensorType.GetProperty("EffectiveForwardRange").GetValue(sensor);
            Assert.That(range, Is.EqualTo(22.5f).Within(0.01f));
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(root);
        }
    }

    [Test]
    public void TestCoordinatorSelectsRouteProviderFromActiveSourceKind()
    {
        GameObject replayRoot = new GameObject("ReplayProviderTest");
        GameObject streamingRoot = new GameObject("StreamingProviderTest");
        GameObject coordinatorRoot = new GameObject("CoordinatorTest");
        try
        {
            Type replayProviderType = FindRuntimeType("ReplayRoadGenerator");
            Type streamingProviderType = FindRuntimeType("StreamingRouteProvider");
            Type coordinatorType = FindRuntimeType("LiveTestCoordinator");
            Type sourceKindType = FindRuntimeType("TwinSourceKind");
            Component replayProvider = replayRoot.AddComponent(replayProviderType);
            Component streamingProvider = streamingRoot.AddComponent(streamingProviderType);
            Component coordinator = coordinatorRoot.AddComponent(coordinatorType);
            SetField(coordinator, "routeProviderComponent", replayProvider);
            SetField(coordinator, "roadGenerator", replayProvider);
            SetField(coordinator, "streamingRouteProviderComponent", streamingProvider);

            MethodInfo resolve = coordinatorType.GetMethod(
                "ResolveRouteProviderForSource",
                PrivateInstance);
            Assert.That(resolve, Is.Not.Null);

            object[] replayArgs = { Enum.Parse(sourceKindType, "Replay"), null };
            Assert.That(resolve.Invoke(coordinator, replayArgs), Is.SameAs(replayProvider));
            Assert.That(replayArgs[1], Is.Null);

            object[] simulatedArgs = { Enum.Parse(sourceKindType, "SimulatedStream"), null };
            Assert.That(resolve.Invoke(coordinator, simulatedArgs), Is.SameAs(streamingProvider));
            Assert.That(simulatedArgs[1], Is.Null);

            object[] liveArgs = { Enum.Parse(sourceKindType, "LiveSensor"), null };
            Assert.That(resolve.Invoke(coordinator, liveArgs), Is.SameAs(streamingProvider));
            Assert.That(liveArgs[1], Is.Null);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(coordinatorRoot);
            UnityEngine.Object.DestroyImmediate(streamingRoot);
            UnityEngine.Object.DestroyImmediate(replayRoot);
        }
    }

    [Test]
    public void FastClosingRearActorMakesLaneUnsafe()
    {
        Type snapshotType = FindRuntimeType("SimulatedSensorSnapshot");
        Type observationType = FindRuntimeType("SimulatedActorObservation");
        object snapshot = Activator.CreateInstance(snapshotType);
        object rear = Activator.CreateInstance(observationType);
        observationType.GetField("gapDistance").SetValue(rear, 20f);
        observationType.GetField("timeToCollision").SetValue(rear, 2f);
        snapshotType.GetField("leftLaneRear").SetValue(snapshot, rear);

        bool clear = (bool)snapshotType.GetMethod("IsLeftLaneClear")
            .Invoke(snapshot, new object[] { 15f, 10f });

        Assert.That(clear, Is.False);
    }

    [Test]
    public void LegacyActorConfigurationSeparatesClassFromMotionAndAddsRootCollider()
    {
        GameObject actor = new GameObject("LegacySlowCar");
        GameObject visual = GameObject.CreatePrimitive(PrimitiveType.Cube);
        visual.transform.SetParent(actor.transform, false);
        try
        {
            Type markerType = FindRuntimeType("ScenarioActorMarker");
            Type actorType = FindRuntimeType("ScenarioActorType");
            Component marker = actor.AddComponent(markerType);
            object slowCar = Enum.Parse(actorType, "SlowCar");
            markerType.GetMethod("Configure", new[] { actorType, typeof(bool) })
                .Invoke(marker, new[] { slowCar, (object)true });

            Assert.That(markerType.GetProperty("SemanticClass").GetValue(marker).ToString(), Is.EqualTo("Car"));
            Assert.That(markerType.GetProperty("MotionState").GetValue(marker).ToString(), Is.EqualTo("Slow"));
            Assert.That(actor.GetComponent<Collider>(), Is.Not.Null);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(actor);
        }
    }

    [Test]
    public void PredictedCollisionStopsWithoutReportingConfirmedCollision()
    {
        GameObject ego = new GameObject("PredictiveEgo");
        GameObject actor = CreateActor("PredictedActor", "Car", "Stopped", true);
        try
        {
            Type controllerType = FindRuntimeType("AutonomousTestVehicleController");
            Component controller = CreateController(ego, controllerType);
            actor.transform.position = new Vector3(0f, 0f, 5f);
            Physics.SyncTransforms();

            bool collisionEvent = false;
            EventInfo collisionOccurred = controllerType.GetEvent("CollisionOccurred");
            collisionOccurred.AddEventHandler(controller, new Action(() => collisionEvent = true));

            MethodInfo predict = controllerType.GetMethod("PredictCollision", PrivateInstance);
            object[] predictArgs = { Quaternion.identity, Vector3.forward, 3f, default(RaycastHit) };
            bool predicted = (bool)predict.Invoke(controller, predictArgs);
            Assert.That(predicted, Is.True, "The full ego sweep should see the actor before movement.");

            RaycastHit hit = (RaycastHit)predictArgs[3];
            controllerType.GetMethod("ApplyPredictiveEmergencyStop", PrivateInstance)
                .Invoke(controller, new object[] { hit.distance });

            Assert.That(GetProperty<bool>(controller, "IsPredictiveEmergencyStopActive"), Is.True);
            Assert.That(GetProperty<bool>(controller, "HasConfirmedCollision"), Is.False);
            Assert.That(collisionEvent, Is.False);
            Assert.That(GetProperty<float>(controller, "CurrentSpeedKph"), Is.Zero.Within(0.001f));
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(actor);
            UnityEngine.Object.DestroyImmediate(ego);
        }
    }

    [Test]
    public void VerifiedOverlapReportsConfirmedCollision()
    {
        GameObject ego = new GameObject("CollisionEgo");
        GameObject actor = CreateActor("OverlappingActor", "Car", "Stopped", true);
        try
        {
            Type controllerType = FindRuntimeType("AutonomousTestVehicleController");
            Component controller = CreateController(ego, controllerType);
            actor.transform.position = new Vector3(0f, 0f, 1f);
            Physics.SyncTransforms();

            bool collisionEvent = false;
            controllerType.GetEvent("CollisionOccurred")
                .AddEventHandler(controller, new Action(() => collisionEvent = true));

            MethodInfo overlap = controllerType.GetMethod("TryGetConfirmedActorOverlap", PrivateInstance);
            object[] overlapArgs = { null };
            Assert.That((bool)overlap.Invoke(controller, overlapArgs), Is.True);
            controllerType.GetMethod("ReportCollision", PrivateInstance)
                .Invoke(controller, new[] { overlapArgs[0] });

            Assert.That(GetProperty<bool>(controller, "HasConfirmedCollision"), Is.True);
            Assert.That(collisionEvent, Is.True);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(actor);
            UnityEngine.Object.DestroyImmediate(ego);
        }
    }

    [Test]
    public void PassingPolicyKeepsSlowCarAndTruckLeftButReturnsAfterStoppedCar()
    {
        GameObject ego = new GameObject("PolicyEgo");
        GameObject actor = null;
        try
        {
            Type controllerType = FindRuntimeType("AutonomousTestVehicleController");
            Component controller = CreateController(ego, controllerType);
            MethodInfo policy = controllerType.GetMethod("ShouldRemainLeftAfterPassing", PrivateStatic);

            actor = CreateActor("SlowCar", "Car", "Slow", true);
            Assert.That((bool)policy.Invoke(null, new object[] { GetMarker(actor) }), Is.True);
            UnityEngine.Object.DestroyImmediate(actor);

            actor = CreateActor("Truck", "Truck", "Slow", true);
            Assert.That((bool)policy.Invoke(null, new object[] { GetMarker(actor) }), Is.True);
            UnityEngine.Object.DestroyImmediate(actor);

            actor = CreateActor("StoppedCar", "Car", "Stopped", true);
            Assert.That((bool)policy.Invoke(null, new object[] { GetMarker(actor) }), Is.False);
        }
        finally
        {
            if (actor != null) UnityEngine.Object.DestroyImmediate(actor);
            UnityEngine.Object.DestroyImmediate(ego);
        }
    }

    [Test]
    public void LeftCruiseReturnsRightOnlyForStoppedObstruction()
    {
        AssertLeftCruisePolicy("Stopped", "Returning");
        AssertLeftCruisePolicy("Slow", "CruisingLeft");
    }

    [Test]
    public void BlockedRecoveryRequiresStableClearGapBeforeRetry()
    {
        GameObject ego = new GameObject("BlockedRecoveryEgo");
        try
        {
            Type controllerType = FindRuntimeType("AutonomousTestVehicleController");
            Component controller = CreateController(ego, controllerType);
            SetOvertakePhase(controller, "Blocked");
            SetField(controller, "phaseElapsed", 2f);
            SetField(controller, "physicalLaneOffset", 0f);
            object clearSnapshot = Activator.CreateInstance(FindRuntimeType("SimulatedSensorSnapshot"));
            MethodInfo update = controllerType.GetMethod("UpdateOvertakeState", PrivateInstance);

            update.Invoke(controller, new[] { (object)Vector3.forward, clearSnapshot, 0.3f });
            Assert.That(GetOvertakePhase(controller), Is.EqualTo("Blocked"));
            update.Invoke(controller, new[] { (object)Vector3.forward, clearSnapshot, 0.4f });
            Assert.That(GetOvertakePhase(controller), Is.EqualTo("MovingOut"));
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(ego);
        }
    }

    [Test]
    public void PedestrianHazardRequestsFullObstacleBrake()
    {
        GameObject ego = new GameObject("PedestrianBrakeEgo");
        GameObject pedestrian = CreateActor("Pedestrian", "Pedestrian", "Moving", false);
        try
        {
            Type controllerType = FindRuntimeType("AutonomousTestVehicleController");
            Component controller = CreateController(ego, controllerType);
            SetField(controller, "currentSpeedMps", 8f);
            object observation = CreateObservation(GetMarker(pedestrian), 2f, 0.4f);
            object snapshot = Activator.CreateInstance(FindRuntimeType("SimulatedSensorSnapshot"));
            snapshot.GetType().GetField("pedestrianHazard").SetValue(snapshot, observation);
            object[] args = { snapshot, 8f };

            bool limited = (bool)controllerType.GetMethod("ApplySensorDecision", PrivateInstance)
                .Invoke(controller, args);
            Assert.That(limited, Is.True);
            Assert.That((float)args[1], Is.Zero.Within(0.001f));
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(pedestrian);
            UnityEngine.Object.DestroyImmediate(ego);
        }
    }

    [Test]
    public void WeatherModifierReducesRangeAndSelectsWeatherSlowdownReason()
    {
        GameObject ego = new GameObject("WeatherEgo");
        try
        {
            Type controllerType = FindRuntimeType("AutonomousTestVehicleController");
            Component controller = CreateController(ego, controllerType);
            controllerType.GetMethod("SetEnvironmentModifiers")
                .Invoke(controller, new object[] { 0.5f, 0.55f, 0.38f });
            Component sensor = ego.GetComponent(FindRuntimeType("SimulatedVehicleSensorSuite"));

            Assert.That(GetProperty<float>(sensor, "EffectiveForwardRange"), Is.EqualTo(17.1f).Within(0.01f));
            object reason = controllerType.GetMethod("GetEnvironmentDecelerationReason", PrivateInstance)
                .Invoke(controller, null);
            Assert.That(reason.ToString(), Is.EqualTo("Weather"));
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(ego);
        }
    }

    [Test]
    public void CurvedRouteProgressUsesCumulativeSegmentDistanceAndNeverMovesBackward()
    {
        GameObject ego = new GameObject("CurvedRouteEgo");
        try
        {
            Type controllerType = FindRuntimeType("AutonomousTestVehicleController");
            Component controller = CreateController(ego, controllerType);
            List<Vector3> route = new List<Vector3>
            {
                new Vector3(0f, 0f, 0f),
                new Vector3(0f, 0f, 10f),
                new Vector3(10f, 0f, 10f),
                new Vector3(10f, 0f, 20f)
            };
            controllerType.GetMethod("SetRoute").Invoke(controller, new object[] { route, 0f });
            Rigidbody body = ego.GetComponent<Rigidbody>();
            MethodInfo update = controllerType.GetMethod("UpdateRouteProgress", PrivateInstance);

            body.position = new Vector3(5f, 0f, 10f);
            update.Invoke(controller, null);
            float aroundCorner = GetField<float>(controller, "routeProgress");
            body.position = new Vector3(10f, 0f, 15f);
            update.Invoke(controller, null);
            float afterCorner = GetField<float>(controller, "routeProgress");

            Assert.That(aroundCorner, Is.EqualTo(15f).Within(0.2f));
            Assert.That(afterCorner, Is.EqualTo(25f).Within(0.2f));
            Assert.That(afterCorner, Is.GreaterThan(aroundCorner));
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(ego);
        }
    }

    [Test]
    public void MetricsAccumulateActualPathLengthInsteadOfStraightLineDisplacement()
    {
        GameObject root = new GameObject("MetricsPath");
        try
        {
            Type metricsType = FindRuntimeType("TestMetricsRecorder");
            Component recorder = root.AddComponent(metricsType);
            SetField(recorder, "lastSamplePosition", Vector3.zero);
            MethodInfo accumulate = metricsType.GetMethod("AccumulatePathDistance", PrivateInstance);
            accumulate.Invoke(recorder, new object[] { new Vector3(3f, 0f, 4f) });
            accumulate.Invoke(recorder, new object[] { new Vector3(6f, 0f, 0f) });
            Assert.That(GetField<float>(recorder, "travelledPathDistance"), Is.EqualTo(10f).Within(0.001f));
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(root);
        }
    }

    [Test]
    public void FallbackActorsHaveRecognizablePartsAndPracticalRootColliders()
    {
        Type managerType = FindRuntimeType("TestScenarioManager");
        Type actorType = FindRuntimeType("ScenarioActorType");
        GameObject pedestrian = null;
        GameObject car = null;
        try
        {
            pedestrian = (GameObject)managerType.GetMethod("CreateFallbackPedestrian", PrivateStatic)
                .Invoke(null, null);
            Assert.That(pedestrian.transform.Find("Head"), Is.Not.Null);
            Assert.That(pedestrian.transform.Find("Torso"), Is.Not.Null);
            ConfigureMarker(pedestrian, "Pedestrian", "Moving", false);
            Assert.That(pedestrian.GetComponent<Collider>(), Is.Not.Null);

            object stopped = Enum.Parse(actorType, "StoppedCar");
            car = (GameObject)managerType.GetMethod("CreateFallbackVehicle", PrivateStatic)
                .Invoke(null, new object[] { stopped, new Vector3(1.8f, 1.5f, 4.2f), Color.blue });
            Assert.That(car.transform.Find("Body"), Is.Not.Null);
            Assert.That(car.transform.Find("LeftWheel"), Is.Not.Null);
            ConfigureMarker(car, "Car", "Stopped", true);
            Assert.That(car.GetComponent<Collider>(), Is.Not.Null);
        }
        finally
        {
            if (pedestrian != null) UnityEngine.Object.DestroyImmediate(pedestrian);
            if (car != null) UnityEngine.Object.DestroyImmediate(car);
        }
    }

    [Test]
    public void MlProtocolIsVersionedAndMapsOnlySupportedActions()
    {
        Type bridgeType = FindRuntimeType("TestLabMlDecisionBridge");
        Type decisionType = FindRuntimeType("TestLabMlDecision");
        Assert.That(
            bridgeType.GetField("ProtocolVersion").GetRawConstantValue(),
            Is.EqualTo("roadweave.testlab-ml/1.0")
        );

        MethodInfo parse = decisionType.GetMethod("ParseAction", BindingFlags.Public | BindingFlags.Static);
        Assert.That(parse.Invoke(null, new object[] { "CHANGE_LEFT" }).ToString(), Is.EqualTo("ChangeLeft"));
        Assert.That(parse.Invoke(null, new object[] { "DROP_TABLE" }).ToString(), Is.EqualTo("None"));
    }

    [Test]
    public void ControllerAcceptsRuntimeMlBridgeWithoutSceneOrCanvasReference()
    {
        GameObject ego = new GameObject("MlBridgeEgo");
        try
        {
            Type controllerType = FindRuntimeType("AutonomousTestVehicleController");
            Type bridgeType = FindRuntimeType("TestLabMlDecisionBridge");
            Component controller = CreateController(ego, controllerType);
            Component bridge = ego.AddComponent(bridgeType);
            controllerType.GetMethod("SetMlDecisionBridge").Invoke(controller, new object[] { bridge });

            string summary = GetProperty<string>(controller, "DecisionSourceSummary");
            Assert.That(summary, Does.StartWith("ML:"));
            Assert.That(ego.GetComponent(bridgeType), Is.SameAs(bridge));
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(ego);
        }
    }

    [Test]
    public void RiskDrivingModeIsExplicitAndResetsItsInterventionCounters()
    {
        GameObject ego = new GameObject("RiskDrivingEgo");
        try
        {
            Type controllerType = FindRuntimeType("AutonomousTestVehicleController");
            Component controller = CreateController(ego, controllerType);
            controllerType.GetMethod("SetRiskDriving").Invoke(controller, new object[] { true });

            Assert.That(GetProperty<bool>(controller, "IsRiskDrivingActive"), Is.True);
            Assert.That(GetProperty<float>(controller, "CurrentDriverRequestedSpeedKph"), Is.EqualTo(80f));
            Assert.That(GetProperty<bool>(controller, "IsMlRiskTakeoverActive"), Is.False);
            Assert.That(GetProperty<string>(controller, "RiskDrivingSummary"), Does.Contain("Reckless driver requests"));

            controllerType.GetMethod("UpdateRiskDrivingPhase", PrivateInstance)
                .Invoke(controller, new object[] { 4f });
            Assert.That(GetProperty<bool>(controller, "IsMlRiskTakeoverActive"), Is.False);
            Assert.That(GetProperty<string>(controller, "RiskDrivingSummary"), Does.Contain("ML is observing"));

            controllerType.GetMethod("UpdateRiskDrivingPhase", PrivateInstance)
                .Invoke(controller, new object[] { 4f });
            Assert.That(GetProperty<bool>(controller, "IsMlRiskTakeoverActive"), Is.True);
            Assert.That(GetProperty<string>(controller, "RiskDrivingSummary"), Does.Contain("ML takeover"));

            controllerType.GetMethod("SetRiskDriving").Invoke(controller, new object[] { false });
            Assert.That(GetProperty<bool>(controller, "IsRiskDrivingActive"), Is.False);
            Assert.That(GetProperty<int>(controller, "MlRiskInterventionCount"), Is.Zero);
            Assert.That(GetProperty<int>(controller, "SafetyRiskInterventionCount"), Is.Zero);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(ego);
        }
    }

    [Test]
    public void RiskDrivingButtonIsInstalledAtRuntimeWithoutEditingTheCanvasAsset()
    {
        GameObject coordinatorRoot = new GameObject("RiskButtonCoordinator");
        GameObject panel = new GameObject("TestLabPanel", typeof(RectTransform));
        GameObject template = new GameObject("RunTestButton", typeof(RectTransform), typeof(UnityEngine.UI.Image), typeof(UnityEngine.UI.Button));
        template.transform.SetParent(panel.transform, false);
        try
        {
            Type coordinatorType = FindRuntimeType("LiveTestCoordinator");
            Component coordinator = coordinatorRoot.AddComponent(coordinatorType);
            SetField(coordinator, "testLabPanel", panel);
            coordinatorType.GetMethod("EnsureRuntimeRiskDrivingButton", PrivateInstance)
                .Invoke(coordinator, null);

            Transform installed = panel.transform.Find("RiskDrivingButton_Runtime");
            Assert.That(installed, Is.Not.Null);
            Assert.That(installed.GetComponent<UnityEngine.UI.Button>(), Is.Not.Null);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(template);
            UnityEngine.Object.DestroyImmediate(panel);
            UnityEngine.Object.DestroyImmediate(coordinatorRoot);
        }
    }

    private static void AssertLeftCruisePolicy(string motion, string expectedPhase)
    {
        GameObject ego = new GameObject("LeftCruiseEgo");
        GameObject actor = CreateActor("LeftObstacle", "Car", motion, true);
        try
        {
            Type controllerType = FindRuntimeType("AutonomousTestVehicleController");
            Component controller = CreateController(ego, controllerType);
            SetOvertakePhase(controller, "CruisingLeft");
            SetField(controller, "physicalLaneOffset", -5.5f);
            object snapshot = Activator.CreateInstance(FindRuntimeType("SimulatedSensorSnapshot"));
            snapshot.GetType().GetField("leftLaneFront")
                .SetValue(snapshot, CreateObservation(GetMarker(actor), 8f, float.PositiveInfinity));
            controller.GetType().GetMethod("UpdateOvertakeState", PrivateInstance)
                .Invoke(controller, new[] { (object)Vector3.forward, snapshot, 0.7f });
            Assert.That(GetOvertakePhase(controller), Is.EqualTo(expectedPhase));
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(actor);
            UnityEngine.Object.DestroyImmediate(ego);
        }
    }

    private static GameObject CreateActor(string name, string semanticClass, string motionState, bool canOvertake)
    {
        GameObject actor = new GameObject(name);
        BoxCollider collider = actor.AddComponent<BoxCollider>();
        collider.center = new Vector3(0f, 0.75f, 0f);
        collider.size = new Vector3(1.8f, 1.5f, 4.2f);
        ConfigureMarker(actor, semanticClass, motionState, canOvertake);
        return actor;
    }

    private static Component CreateController(GameObject ego, Type controllerType)
    {
        Component controller = ego.AddComponent(controllerType);
        if (controllerType.GetField("physicsBody", PrivateInstance).GetValue(controller) == null)
            controllerType.GetMethod("Awake", PrivateInstance).Invoke(controller, null);
        return controller;
    }

    private static void ConfigureMarker(GameObject actor, string semanticClass, string motionState, bool canOvertake)
    {
        Type markerType = FindRuntimeType("ScenarioActorMarker");
        Type classType = FindRuntimeType("ScenarioActorClass");
        Type motionType = FindRuntimeType("ScenarioActorMotionState");
        Component marker = actor.GetComponent(markerType) ?? actor.AddComponent(markerType);
        markerType.GetMethod("Configure", new[] { classType, motionType, typeof(bool) }).Invoke(marker,
            new[] { Enum.Parse(classType, semanticClass), Enum.Parse(motionType, motionState), (object)canOvertake });
    }

    private static Component GetMarker(GameObject actor)
    {
        return actor.GetComponent(FindRuntimeType("ScenarioActorMarker"));
    }

    private static object CreateObservation(Component actor, float gap, float ttc)
    {
        Type observationType = FindRuntimeType("SimulatedActorObservation");
        object observation = Activator.CreateInstance(observationType);
        observationType.GetField("actor").SetValue(observation, actor);
        observationType.GetField("gapDistance").SetValue(observation, gap);
        observationType.GetField("timeToCollision").SetValue(observation, ttc);
        return observation;
    }

    private static void SetOvertakePhase(Component controller, string phase)
    {
        Type phaseType = controller.GetType().GetNestedType("OvertakePhase", BindingFlags.NonPublic);
        SetField(controller, "overtakePhase", Enum.Parse(phaseType, phase));
    }

    private static string GetOvertakePhase(Component controller)
    {
        return GetField<object>(controller, "overtakePhase").ToString();
    }

    private static void SetField(Component target, string name, object value)
    {
        target.GetType().GetField(name, PrivateInstance).SetValue(target, value);
    }

    private static T GetField<T>(Component target, string name)
    {
        return (T)target.GetType().GetField(name, PrivateInstance).GetValue(target);
    }

    private static T GetProperty<T>(Component target, string name)
    {
        return (T)target.GetType().GetProperty(name).GetValue(target);
    }

    private const BindingFlags PrivateInstance = BindingFlags.Instance | BindingFlags.NonPublic;
    private const BindingFlags PrivateStatic = BindingFlags.Static | BindingFlags.NonPublic;

    private static Type FindRuntimeType(string name)
    {
        Type type = Type.GetType(name + ", Assembly-CSharp");
        Assert.That(type, Is.Not.Null, "Could not find runtime type " + name);
        return type;
    }
}

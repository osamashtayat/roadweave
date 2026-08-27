using System;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public enum TestLabMlAction
{
    None,
    Keep,
    Accelerate,
    Decelerate,
    ChangeLeft,
    ChangeRight,
    EmergencyStop
}

[Serializable]
public sealed class TestLabMlDecision
{
    public string schemaVersion;
    public string messageType;
    public string sessionId;
    public int sequenceNumber;
    public bool valid;
    public string modelVersion;
    public float processDurationMilliseconds;
    public float observationTimestampSeconds;
    public string requestedAction;
    public string executedAction;
    public float actionConfidence;
    public string riskLevel;
    public float riskConfidence;
    public float targetSpeedMps;
    public string overrideReason;
    public int historySamples;
    public string weatherContext;
    public bool weatherModelUsed;
    public float weatherSpeedFactor = 1f;
    public float weatherTargetSpeedMps;
    public string error;

    public TestLabMlAction Action => ParseAction(executedAction);

    public static TestLabMlAction ParseAction(string value)
    {
        switch (value)
        {
            case "KEEP": return TestLabMlAction.Keep;
            case "ACCELERATE": return TestLabMlAction.Accelerate;
            case "DECELERATE": return TestLabMlAction.Decelerate;
            case "CHANGE_LEFT": return TestLabMlAction.ChangeLeft;
            case "CHANGE_RIGHT": return TestLabMlAction.ChangeRight;
            case "EMERGENCY_STOP": return TestLabMlAction.EmergencyStop;
            default: return TestLabMlAction.None;
        }
    }
}

[DisallowMultipleComponent]
public sealed class TestLabMlDecisionBridge : MonoBehaviour
{
    public const string ProtocolVersion = "roadweave.testlab-ml/1.0";

    [Header("Local ML Decision Service")]
    [SerializeField] private bool enableMlDecisions = true;
    [SerializeField] private string serviceHost = "127.0.0.1";
    [SerializeField, Range(1, 65535)] private int servicePort = 5075;
    [SerializeField, Min(1f)] private float decisionRateHz = 5f;
    [SerializeField, Min(0.1f)] private float staleAfterSeconds = 0.8f;

    public bool MlDecisionsEnabled => enableMlDecisions;
    public bool HasFreshDecision
    {
        get
        {
            return latestDecision != null &&
                   latestDecision.valid &&
                   latestDecision.sessionId == sessionId &&
                   latestDecision.Action != TestLabMlAction.None &&
                   Time.realtimeSinceStartup - latestDecisionReceivedTime <= staleAfterSeconds;
        }
    }
    public string SessionId => sessionId;
    public int ReplyPort => replyPort;
    public string LastError { get; private set; } = "";
    public TestLabMlDecision LatestDecision => latestDecision;
    public string StatusSummary
    {
        get
        {
            if (!enableMlDecisions)
                return "ML: disabled (rule controller active)";
            if (!transportStarted)
                return string.IsNullOrEmpty(LastError)
                    ? "ML: transport not started (rule fallback active)"
                    : $"ML: {LastError} (rule fallback active)";
            if (latestDecision == null)
                return $"ML: waiting for {serviceHost}:{servicePort} (rule fallback active)";
            if (!HasFreshDecision)
                return "ML: response stale or invalid (rule fallback active)";

            string summary =
                $"ML: {latestDecision.executedAction} {latestDecision.actionConfidence:P0}, " +
                $"risk {latestDecision.riskLevel} {latestDecision.riskConfidence:P0}, " +
                $"target {latestDecision.targetSpeedMps * 3.6f:F1} km/h";
            if (latestDecision.weatherModelUsed)
            {
                summary +=
                    $", weather {latestDecision.weatherContext} " +
                    $"target {latestDecision.weatherTargetSpeedMps * 3.6f:F1} km/h";
            }
            if (!string.IsNullOrWhiteSpace(latestDecision.overrideReason))
                summary += $" ({latestDecision.overrideReason})";
            return summary;
        }
    }

    [Serializable]
    private sealed class EgoObservation
    {
        public float speedMps;
        public float accelerationMps2;
        public float yawRateDegreesPerSecond;
    }

    [Serializable]
    private sealed class ActorObservation
    {
        public bool present;
        public float gapMeters;
        public float closingSpeedMps;
        public float actorSpeedMps;
        public float timeToCollisionSeconds;
        public string semanticClass;
        public string motionState;
    }

    [Serializable]
    private sealed class ObservationRequest
    {
        public string schemaVersion = ProtocolVersion;
        public string messageType = "observation";
        public string sessionId;
        public int sequenceNumber;
        public float timestampSeconds;
        public int replyPort;
        public string currentLane;
        public float cruiseSpeedMps;
        public bool leftLaneClear;
        public bool rightLaneClear;
        public string weather;
        public EgoObservation ego;
        public ActorObservation rightFront;
        public ActorObservation rightRear;
        public ActorObservation leftFront;
        public ActorObservation leftRear;
        public ActorObservation pedestrian;
    }

    [Serializable]
    private sealed class ResetRequest
    {
        public string schemaVersion = ProtocolVersion;
        public string messageType = "reset";
        public string sessionId;
        public int sequenceNumber;
        public int replyPort;
    }

    private readonly ConcurrentQueue<string> receivedMessages = new ConcurrentQueue<string>();
    private readonly ConcurrentQueue<string> receiverErrors = new ConcurrentQueue<string>();
    private UdpClient receiver;
    private UdpClient sender;
    private Thread receiverThread;
    private IPEndPoint serviceEndPoint;
    private volatile bool stopReceiver;
    private bool transportStarted;
    private int replyPort;
    private string sessionId = "";
    private int nextSequence;
    private float nextRequestTime;
    private TestLabMlDecision latestDecision;
    private float latestDecisionReceivedTime;
    private bool loggedConnection;

    private void OnEnable()
    {
        if (Application.isPlaying && enableMlDecisions)
            StartTransport();
    }

    private void Update()
    {
        DrainReceiverErrors();
        while (receivedMessages.TryDequeue(out string json))
            ProcessResponse(json);
    }

    public void BeginSession()
    {
        sessionId = Guid.NewGuid().ToString("N");
        nextSequence = 0;
        nextRequestTime = 0f;
        latestDecision = null;
        latestDecisionReceivedTime = 0f;
        loggedConnection = false;
        LastError = "";

        if (!enableMlDecisions)
            return;
        if (!transportStarted)
            StartTransport();
        if (!transportStarted)
            return;

        ResetRequest request = new ResetRequest
        {
            sessionId = sessionId,
            sequenceNumber = nextSequence++,
            replyPort = replyPort
        };
        SendJson(JsonUtility.ToJson(request));
    }

    public void RequestDecision(
        SimulatedSensorSnapshot sensors,
        float speedMps,
        float accelerationMps2,
        float yawRateDegreesPerSecond,
        float physicalLaneOffset,
        float leftLaneOffset,
        float cruiseSpeedMps,
        string weather)
    {
        if (!enableMlDecisions || sensors == null)
            return;
        if (string.IsNullOrEmpty(sessionId))
            BeginSession();
        if (!transportStarted || Time.realtimeSinceStartup < nextRequestTime)
            return;

        float interval = 1f / Mathf.Max(1f, decisionRateHz);
        nextRequestTime = Time.realtimeSinceStartup + interval;
        bool onLeft = physicalLaneOffset <= leftLaneOffset * 0.5f;
        ObservationRequest request = new ObservationRequest
        {
            sessionId = sessionId,
            sequenceNumber = nextSequence++,
            timestampSeconds = Time.realtimeSinceStartup,
            replyPort = replyPort,
            currentLane = onLeft ? "LEFT" : "RIGHT",
            cruiseSpeedMps = Mathf.Max(0f, cruiseSpeedMps),
            leftLaneClear = !onLeft && sensors.IsLeftLaneClear(25f, 12f),
            rightLaneClear = onLeft && sensors.IsRightLaneClear(18f, 10f),
            weather = string.IsNullOrWhiteSpace(weather) ? "Dry" : weather,
            ego = new EgoObservation
            {
                speedMps = Sanitize(speedMps),
                accelerationMps2 = Sanitize(accelerationMps2),
                yawRateDegreesPerSecond = Sanitize(yawRateDegreesPerSecond)
            },
            rightFront = BuildObservation(sensors.rightLaneFront),
            rightRear = BuildObservation(sensors.rightLaneRear),
            leftFront = BuildObservation(sensors.leftLaneFront),
            leftRear = BuildObservation(sensors.leftLaneRear),
            pedestrian = BuildObservation(sensors.pedestrianHazard)
        };
        SendJson(JsonUtility.ToJson(request));
    }

    public bool TryGetFreshDecision(out TestLabMlDecision decision)
    {
        decision = HasFreshDecision ? latestDecision : null;
        return decision != null;
    }

    private static ActorObservation BuildObservation(SimulatedActorObservation source)
    {
        if (source == null)
            return new ActorObservation { present = false };

        return new ActorObservation
        {
            present = true,
            gapMeters = Mathf.Max(0f, Sanitize(source.gapDistance)),
            closingSpeedMps = Sanitize(source.relativeClosingSpeed),
            actorSpeedMps = Sanitize(source.actorLongitudinalSpeed),
            timeToCollisionSeconds = Mathf.Clamp(
                float.IsInfinity(source.timeToCollision) ? 20f : Sanitize(source.timeToCollision),
                0f,
                20f
            ),
            semanticClass = source.actor != null ? source.actor.SemanticClass.ToString() : "Unknown",
            motionState = source.actor != null ? source.actor.MotionState.ToString() : "Unknown"
        };
    }

    private static float Sanitize(float value)
    {
        return float.IsNaN(value) || float.IsInfinity(value) ? 0f : value;
    }

    private void StartTransport()
    {
        if (transportStarted || !enableMlDecisions)
            return;

        try
        {
            IPAddress address;
            if (!IPAddress.TryParse(serviceHost, out address))
            {
                IPAddress[] addresses = Dns.GetHostAddresses(serviceHost);
                address = Array.Find(addresses, candidate => candidate.AddressFamily == AddressFamily.InterNetwork);
            }
            if (address == null)
                throw new InvalidOperationException($"No IPv4 address was found for {serviceHost}.");

            serviceEndPoint = new IPEndPoint(address, servicePort);
            receiver = new UdpClient(new IPEndPoint(IPAddress.Loopback, 0));
            replyPort = ((IPEndPoint)receiver.Client.LocalEndPoint).Port;
            sender = new UdpClient(AddressFamily.InterNetwork);
            stopReceiver = false;
            receiverThread = new Thread(ReceiveLoop)
            {
                IsBackground = true,
                Name = "RoadWeave Test Lab ML Receiver"
            };
            receiverThread.Start();
            transportStarted = true;
            LastError = "";
        }
        catch (Exception error)
        {
            LastError = $"transport could not start: {error.Message}";
            StopTransport();
            Debug.LogError($"RoadWeave Test Lab ML {LastError}");
        }
    }

    private void ReceiveLoop()
    {
        IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);
        while (!stopReceiver)
        {
            try
            {
                byte[] bytes = receiver.Receive(ref remote);
                receivedMessages.Enqueue(Encoding.UTF8.GetString(bytes));
            }
            catch (ObjectDisposedException)
            {
                if (!stopReceiver)
                    receiverErrors.Enqueue("response socket closed unexpectedly");
                return;
            }
            catch (SocketException error)
            {
                if (!stopReceiver)
                    receiverErrors.Enqueue($"response socket error: {error.Message}");
                return;
            }
            catch (Exception error)
            {
                if (!stopReceiver)
                    receiverErrors.Enqueue($"response receive error: {error.Message}");
                return;
            }
        }
    }

    private void SendJson(string json)
    {
        if (!transportStarted || sender == null || serviceEndPoint == null)
            return;
        try
        {
            byte[] bytes = Encoding.UTF8.GetBytes(json);
            sender.Send(bytes, bytes.Length, serviceEndPoint);
        }
        catch (Exception error)
        {
            LastError = $"request send failed: {error.Message}";
        }
    }

    private void ProcessResponse(string json)
    {
        try
        {
            TestLabMlDecision response = JsonUtility.FromJson<TestLabMlDecision>(json);
            if (response == null || response.schemaVersion != ProtocolVersion)
            {
                LastError = "received an unsupported ML response";
                return;
            }
            if (response.sessionId != sessionId)
                return;
            if (!response.valid)
            {
                LastError = string.IsNullOrEmpty(response.error)
                    ? "ML service rejected a request"
                    : response.error;
                return;
            }
            if (response.messageType == "reset_ack")
                return;
            if (response.messageType != "decision" || response.Action == TestLabMlAction.None)
                return;
            if (latestDecision != null && response.sequenceNumber <= latestDecision.sequenceNumber)
                return;

            latestDecision = response;
            latestDecisionReceivedTime = Time.realtimeSinceStartup;
            LastError = "";
            if (!loggedConnection)
            {
                loggedConnection = true;
                Debug.Log(
                    $"RoadWeave Test Lab connected to the ML decision service on " +
                    $"{serviceHost}:{servicePort}. Model {response.modelVersion}."
                );
            }
        }
        catch (Exception error)
        {
            LastError = $"invalid ML response: {error.Message}";
        }
    }

    private void DrainReceiverErrors()
    {
        while (receiverErrors.TryDequeue(out string error))
            LastError = error;
    }

    private void StopTransport()
    {
        stopReceiver = true;
        transportStarted = false;
        try { receiver?.Close(); }
        catch { }
        try { sender?.Close(); }
        catch { }
        if (receiverThread != null && receiverThread.IsAlive)
            receiverThread.Join(250);
        receiverThread = null;
        receiver = null;
        sender = null;
        serviceEndPoint = null;
        replyPort = 0;
    }

    private void OnDisable()
    {
        StopTransport();
    }

    private void OnApplicationQuit()
    {
        StopTransport();
    }
}

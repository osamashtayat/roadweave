using System;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// Receives complete canonical TwinSnapshot JSON messages from an external
/// simulator and forwards them through SimulatedStreamTwinSource.
///
/// Data direction:
/// Python simulator -> UDP 5055 -> this component -> SimulatedStreamTwinSource.
///
/// Control direction:
/// Drive/Pause -> SimulatedStreamTwinSource session -> UDP 5056 -> Python.
/// </summary>
[DisallowMultipleComponent]
public sealed class SimulatedJsonUdpTransport : MonoBehaviour
{
    [Header("RoadWeave Source")]
    [SerializeField] private SimulatedStreamTwinSource simulatedSource;

    [Header("Incoming Snapshot Stream")]
    [Tooltip("Use 127.0.0.1 when Python and Unity run on the same computer.")]
    [SerializeField] private string bindAddress = "127.0.0.1";

    [SerializeField, Range(1, 65535)]
    private int dataPort = 5055;

    [SerializeField, Min(1024)]
    private int receiveBufferBytes = 262144;

    [SerializeField, Min(1)]
    private int maximumPendingMessages = 128;

    [SerializeField, Min(1)]
    private int maximumMessagesPerFrame = 16;

    [Header("Outgoing Simulator Controls")]
    [SerializeField] private string simulatorAddress = "127.0.0.1";

    [SerializeField, Range(1, 65535)]
    private int simulatorControlPort = 5056;

    [Header("Startup")]
    [SerializeField] private bool startAutomatically = true;

    [Tooltip("Keep the Python world paused until the Unity Drive button is pressed.")]
    [SerializeField] private bool beginPaused = true;

    private readonly ConcurrentQueue<string> pendingMessages =
        new ConcurrentQueue<string>();

    private readonly ConcurrentQueue<string> pendingErrors =
        new ConcurrentQueue<string>();

    private UdpClient receiveClient;
    private UdpClient controlClient;
    private IPEndPoint controlEndPoint;
    private Thread receiveThread;
    private volatile bool receiveLoopRunning;
    private long packetsReceived;
    private long packetsPublished;
    private string observedSessionId;

    public bool IsRunning => receiveLoopRunning;
    public int DataPort => dataPort;
    public long PacketsReceived => Interlocked.Read(ref packetsReceived);
    public long PacketsPublished => Interlocked.Read(ref packetsPublished);
    public string LastError { get; private set; }

    private void Awake()
    {
        ResolveSource();
    }

    private void OnEnable()
    {
        ResolveSource();
        SubscribeToSource();

        if (startAutomatically)
            StartTransport();
    }

    private void Update()
    {
        while (pendingErrors.TryDequeue(out string error))
        {
            LastError = error;
            Debug.LogError(error);

            if (simulatedSource != null)
                simulatedSource.ReportError(error);
        }

        PublishPendingMessages();
    }

    private void OnDisable()
    {
        StopTransport();
        UnsubscribeFromSource();
    }

    private void OnDestroy()
    {
        StopTransport();
        UnsubscribeFromSource();
    }

    public bool StartTransport()
    {
        if (receiveLoopRunning)
            return true;

        ResolveSource();
        SubscribeToSource();

        if (simulatedSource == null)
        {
            ReportStartError(
                "SimulatedJsonUdpTransport requires a SimulatedStreamTwinSource.");
            return false;
        }

        if (!IPAddress.TryParse(bindAddress, out IPAddress localAddress))
        {
            ReportStartError(
                $"The UDP bind address '{bindAddress}' is not valid.");
            return false;
        }

        if (!IPAddress.TryParse(simulatorAddress, out IPAddress simulatorIp))
        {
            ReportStartError(
                $"The simulator address '{simulatorAddress}' is not valid.");
            return false;
        }

        try
        {
            receiveClient = new UdpClient(
                new IPEndPoint(localAddress, dataPort));

            receiveClient.Client.ReceiveBufferSize =
                Mathf.Max(1024, receiveBufferBytes);

            receiveClient.Client.ReceiveTimeout = 250;

            controlClient = new UdpClient();

            controlEndPoint = new IPEndPoint(
                simulatorIp,
                simulatorControlPort);

            receiveLoopRunning = true;

            receiveThread = new Thread(ReceiveLoop)
            {
                IsBackground = true,
                Name = $"RoadWeave-Simulated-UDP-{dataPort}"
            };

            receiveThread.Start();

            LastError = null;

            // Start a fresh source session whenever the transport binds. In the
            // initial Ready state, stationary heartbeat snapshots are accepted,
            // so the Live Twin can appear before Drive is pressed. Paused is
            // reserved for an explicit user pause after driving has started.
            simulatedSource.RestartSession(!beginPaused);

            Debug.Log(
                $"RoadWeave simulated transport is listening on " +
                $"{bindAddress}:{dataPort}. Simulator controls are sent to " +
                $"{simulatorAddress}:{simulatorControlPort}.");

            return true;
        }
        catch (Exception exception)
        {
            receiveLoopRunning = false;

            try
            {
                receiveClient?.Close();
            }
            catch (Exception)
            {
                // Ignore cleanup errors.
            }

            try
            {
                controlClient?.Close();
            }
            catch (Exception)
            {
                // Ignore cleanup errors.
            }

            receiveClient = null;
            controlClient = null;
            controlEndPoint = null;

            ReportStartError(
                $"Could not start the simulated UDP transport. " +
                $"{exception.Message}");

            return false;
        }
    }

    public void StopTransport()
    {
        bool wasRunning = receiveLoopRunning;

        if (simulatedSource != null &&
            simulatedSource.Session != null &&
            simulatedSource.Session.status == TwinSessionStatus.Running)
        {
            simulatedSource.PauseSession();
        }
        else
        {
            SendControlCommand("PAUSE");
        }

        receiveLoopRunning = false;

        UdpClient receiver = receiveClient;
        receiveClient = null;

        try
        {
            receiver?.Close();
        }
        catch (Exception)
        {
            // Ignore cleanup errors.
        }

        Thread thread = receiveThread;
        receiveThread = null;

        if (thread != null &&
            thread.IsAlive &&
            thread != Thread.CurrentThread)
        {
            thread.Join(300);
        }

        try
        {
            controlClient?.Close();
        }
        catch (Exception)
        {
            // Ignore cleanup errors.
        }

        controlClient = null;
        controlEndPoint = null;

        if (wasRunning)
            Debug.Log("RoadWeave simulated UDP transport stopped.");
    }

    /// <summary>
    /// Converts queued JSON messages to TwinSnapshot objects on Unity's main
    /// thread and forwards them through the simulated source.
    /// </summary>
    public int PublishPendingMessages()
    {
        ResolveSource();

        if (simulatedSource == null)
            return 0;

        int processed = 0;
        int published = 0;
        int maximum = Mathf.Max(1, maximumMessagesPerFrame);

        while (processed < maximum &&
               pendingMessages.TryDequeue(out string json))
        {
            processed++;

            if (simulatedSource.AcceptJsonMessage(json))
            {
                published++;
                Interlocked.Increment(ref packetsPublished);
            }
        }

        return published;
    }

    /// <summary>
    /// Thread-safe handoff used by the network thread.
    /// </summary>
    public bool SubmitReceivedMessage(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
            return false;

        int maximum = Math.Max(1, maximumPendingMessages);

        // If the producer is faster than Unity, discard the oldest pending
        // messages. For a digital twin, the newest current state is preferable
        // to an ever-growing backlog of old states.
        while (pendingMessages.Count >= maximum)
            pendingMessages.TryDequeue(out _);

        pendingMessages.Enqueue(json);
        Interlocked.Increment(ref packetsReceived);

        return true;
    }

    private void ReceiveLoop()
    {
        IPEndPoint remoteEndPoint =
            new IPEndPoint(IPAddress.Any, 0);

        while (receiveLoopRunning)
        {
            try
            {
                UdpClient receiver = receiveClient;

                if (receiver == null)
                    break;

                byte[] bytes = receiver.Receive(ref remoteEndPoint);

                if (bytes != null && bytes.Length > 0)
                {
                    string json = Encoding.UTF8.GetString(bytes);
                    SubmitReceivedMessage(json);
                }
            }
            catch (SocketException exception)
            {
                if (exception.SocketErrorCode == SocketError.TimedOut)
                    continue;

                if (receiveLoopRunning)
                {
                    pendingErrors.Enqueue(
                        $"Simulated UDP receive failed: {exception.Message}");
                }

                break;
            }
            catch (ObjectDisposedException)
            {
                break;
            }
            catch (Exception exception)
            {
                if (receiveLoopRunning)
                {
                    pendingErrors.Enqueue(
                        $"Simulated UDP receive stopped: {exception.Message}");
                }

                break;
            }
        }

        receiveLoopRunning = false;

        try
        {
            receiveClient?.Close();
        }
        catch (Exception)
        {
            // Ignore cleanup errors.
        }

        receiveClient = null;
    }

    private void ResolveSource()
    {
        if (simulatedSource == null)
            simulatedSource =
                FindAnyObjectByType<SimulatedStreamTwinSource>();
    }

    private void SubscribeToSource()
    {
        if (simulatedSource == null)
            return;

        simulatedSource.SessionChanged -= HandleSessionChanged;
        simulatedSource.SessionChanged += HandleSessionChanged;
    }

    private void UnsubscribeFromSource()
    {
        if (simulatedSource != null)
            simulatedSource.SessionChanged -= HandleSessionChanged;
    }

    private void HandleSessionChanged(TwinSessionInfo session)
    {
        if (session == null)
            return;

        bool newSession =
            !string.Equals(
                observedSessionId,
                session.sessionId,
                StringComparison.Ordinal);

        observedSessionId = session.sessionId;

        if (newSession)
        {
            if (session.status == TwinSessionStatus.Running)
                SendControlCommand("RESET_START");
            else
                SendControlCommand("RESET_PAUSE");

            return;
        }

        switch (session.status)
        {
            case TwinSessionStatus.Running:
                SendControlCommand("START");
                break;

            case TwinSessionStatus.Ready:
            case TwinSessionStatus.Paused:
            case TwinSessionStatus.Finished:
            case TwinSessionStatus.Error:
            case TwinSessionStatus.Disconnected:
                SendControlCommand("PAUSE");
                break;
        }
    }

    private void SendControlCommand(string command)
    {
        if (controlClient == null ||
            controlEndPoint == null ||
            string.IsNullOrWhiteSpace(command))
        {
            return;
        }

        try
        {
            byte[] bytes = Encoding.UTF8.GetBytes(command);
            controlClient.Send(bytes, bytes.Length, controlEndPoint);
        }
        catch (Exception exception)
        {
            // A control-send failure should not crash Unity. The stream will
            // become stale if the simulator is no longer reachable.
            LastError =
                $"Could not send simulator command '{command}'. " +
                exception.Message;

            Debug.LogWarning(LastError);
        }
    }

    private void ReportStartError(string message)
    {
        LastError = message;
        Debug.LogError(message);

        if (simulatedSource != null)
            simulatedSource.ReportError(message);
    }
}

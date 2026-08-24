using System;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// Configurable UDP ingress for canonical TwinSnapshot JSON messages. Socket IO
/// runs on a background thread; parsing and source publication always happen in
/// Update on Unity's main thread.
/// </summary>
public sealed class LiveJsonUdpTransport : MonoBehaviour
{
    [SerializeField] private LiveSensorTwinSource liveSource;
    [SerializeField] private string bindAddress = "0.0.0.0";
    [SerializeField, Range(1, 65535)] private int port = 5055;
    [SerializeField, Min(1024)] private int receiveBufferBytes = 262144;
    [SerializeField, Min(1)] private int maximumPendingMessages = 128;
    [SerializeField, Min(1)] private int maximumMessagesPerFrame = 16;
    [SerializeField] private bool startAutomatically;
    [SerializeField] private bool pauseSourceWhenStopped = true;

    private readonly ConcurrentQueue<string> pendingMessages = new ConcurrentQueue<string>();
    private readonly ConcurrentQueue<string> pendingErrors = new ConcurrentQueue<string>();
    private UdpClient udpClient;
    private Thread receiveThread;
    private volatile bool receiveLoopRunning;
    private long packetsReceived;
    private long packetsPublished;

    public bool IsRunning => receiveLoopRunning;
    public int Port => port;
    public long PacketsReceived => Interlocked.Read(ref packetsReceived);
    public long PacketsPublished => Interlocked.Read(ref packetsPublished);
    public string LastError { get; private set; }

    private void Awake()
    {
        ResolveSource();
    }

    private void OnEnable()
    {
        if (startAutomatically)
            StartTransport();
    }

    private void Update()
    {
        while (pendingErrors.TryDequeue(out string error))
        {
            LastError = error;
            Debug.LogError(error);
            liveSource?.ReportError(error);
        }
        PublishPendingMessages();
    }

    private void OnDisable() => StopTransport();
    private void OnDestroy() => StopTransport();

    public bool StartTransport()
    {
        if (receiveLoopRunning)
            return true;
        ResolveSource();
        if (liveSource == null)
        {
            ReportStartError("LiveJsonUdpTransport requires a LiveSensorTwinSource.");
            return false;
        }
        if (!IPAddress.TryParse(bindAddress, out IPAddress address))
        {
            ReportStartError($"UDP bind address '{bindAddress}' is not a valid numeric IP address.");
            return false;
        }

        try
        {
            udpClient = new UdpClient(new IPEndPoint(address, port));
            udpClient.Client.ReceiveBufferSize = Mathf.Max(1024, receiveBufferBytes);
            udpClient.Client.ReceiveTimeout = 250;
            receiveLoopRunning = true;
            receiveThread = new Thread(ReceiveLoop)
            {
                IsBackground = true,
                Name = $"RoadWeave-UDP-{port}"
            };
            receiveThread.Start();
            LastError = null;
            liveSource.StartSession();
            return true;
        }
        catch (Exception exception)
        {
            receiveLoopRunning = false;
            udpClient?.Close();
            udpClient = null;
            ReportStartError($"Could not start RoadWeave UDP transport on {bindAddress}:{port}. {exception.Message}");
            return false;
        }
    }

    public void StopTransport()
    {
        bool wasRunning = receiveLoopRunning;
        receiveLoopRunning = false;
        UdpClient client = udpClient;
        udpClient = null;
        try { client?.Close(); }
        catch (Exception) { }

        Thread thread = receiveThread;
        receiveThread = null;
        if (thread != null && thread.IsAlive && thread != Thread.CurrentThread)
            thread.Join(300);

        if (wasRunning && pauseSourceWhenStopped)
            liveSource?.PauseSession();
    }

    /// <summary>
    /// Thread-safe handoff used by the UDP loop and reusable by another transport.
    /// The message is not parsed until PublishPendingMessages runs on the main thread.
    /// </summary>
    public bool SubmitReceivedMessage(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
            return false;
        int maximum = Math.Max(1, maximumPendingMessages);
        while (pendingMessages.Count >= maximum)
            pendingMessages.TryDequeue(out _);
        pendingMessages.Enqueue(json);
        Interlocked.Increment(ref packetsReceived);
        return true;
    }

    /// <summary>Publishes queued messages on the caller's thread; normally Unity Update.</summary>
    public int PublishPendingMessages()
    {
        ResolveSource();
        if (liveSource == null)
            return 0;
        int published = 0;
        int processed = 0;
        int maximum = Mathf.Max(1, maximumMessagesPerFrame);
        while (processed < maximum && pendingMessages.TryDequeue(out string json))
        {
            processed++;
            if (liveSource.AcceptJsonMessage(json))
            {
                published++;
                Interlocked.Increment(ref packetsPublished);
            }
        }
        return published;
    }

    private void ReceiveLoop()
    {
        IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);
        while (receiveLoopRunning)
        {
            try
            {
                UdpClient client = udpClient;
                if (client == null)
                    break;
                byte[] bytes = client.Receive(ref remote);
                if (bytes != null && bytes.Length > 0)
                    SubmitReceivedMessage(Encoding.UTF8.GetString(bytes));
            }
            catch (SocketException exception)
            {
                if (exception.SocketErrorCode == SocketError.TimedOut)
                    continue;
                if (receiveLoopRunning)
                    QueueBackgroundError($"RoadWeave UDP receive failed: {exception.Message}");
                break;
            }
            catch (ObjectDisposedException)
            {
                break;
            }
            catch (Exception exception)
            {
                if (receiveLoopRunning)
                    QueueBackgroundError($"RoadWeave UDP receive stopped: {exception.Message}");
                break;
            }
        }
        receiveLoopRunning = false;
        try { udpClient?.Close(); }
        catch (Exception) { }
        udpClient = null;
    }

    private void QueueBackgroundError(string message)
    {
        pendingErrors.Enqueue(message);
    }

    private void ReportStartError(string message)
    {
        LastError = message;
        Debug.LogError(message);
        liveSource?.ReportError(message);
    }

    private void ResolveSource()
    {
        if (liveSource == null)
            liveSource = FindAnyObjectByType<LiveSensorTwinSource>();
    }
}

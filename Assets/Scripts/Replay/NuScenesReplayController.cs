using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

public class NuScenesReplayController : MonoBehaviour
{
    [Header("Replay File")]
    [SerializeField] private string replayFileName = "Replays/scene-0001-replay.json";

    [Header("Playback")]
    [SerializeField] private DigitalTwinStateManager stateManager;
    [SerializeField, Min(0.01f)] private float playbackSpeed = 1f;
    [SerializeField] private bool playAutomatically;
    [SerializeField] private bool loop;

    public ReplayPackage Package { get; private set; }
    public bool IsLoaded => Package != null;

    private float replayTime;

    private void Awake()
    {
        if (stateManager == null)
            stateManager = GetComponent<DigitalTwinStateManager>();

        if (stateManager == null)
            stateManager = FindFirstObjectByType<DigitalTwinStateManager>();
    }

    private IEnumerator Start()
    {
        yield return LoadReplay();

        if (playAutomatically && IsLoaded)
            StartDrive();
    }

    private void Update()
    {
        if (!IsLoaded || stateManager == null || !stateManager.IsPlaying)
            return;

        replayTime += Time.deltaTime * playbackSpeed;

        if (replayTime >= Package.durationSeconds)
        {
            if (loop)
            {
                replayTime = 0f;
                stateManager.UpdateState(replayTime);
            }
            else
            {
                replayTime = Package.durationSeconds;
                stateManager.UpdateState(replayTime);
                stateManager.SetStatus(ReplayStatus.Finished);
            }
            return;
        }

        stateManager.UpdateState(replayTime);
    }

    private IEnumerator LoadReplay()
    {
        if (stateManager == null)
        {
            Debug.LogError("NuScenesReplayController needs a DigitalTwinStateManager.");
            yield break;
        }

        stateManager.SetStatus(ReplayStatus.Loading);
        string path = Path.Combine(Application.streamingAssetsPath, replayFileName);
        string json = null;

        if (path.Contains("://"))
        {
            using (UnityWebRequest request = UnityWebRequest.Get(path))
            {
                yield return request.SendWebRequest();
                if (request.result != UnityWebRequest.Result.Success)
                {
                    FailLoading(path, request.error);
                    yield break;
                }
                json = request.downloadHandler.text;
            }
        }
        else
        {
            if (!File.Exists(path))
            {
                FailLoading(path, "File does not exist");
                yield break;
            }
            json = File.ReadAllText(path);
        }

        Package = JsonUtility.FromJson<ReplayPackage>(json);
        if (Package == null || Package.egoFrames == null || Package.egoFrames.Length == 0)
        {
            FailLoading(path, "The JSON is empty or has an unsupported structure");
            yield break;
        }

        replayTime = 0f;
        stateManager.Initialize(Package);
        Debug.Log($"Loaded {Package.sceneId}: {Package.durationSeconds:F1} seconds, {Package.actors?.Length ?? 0} actors.");
    }

    private void FailLoading(string path, string reason)
    {
        Debug.LogError($"Could not load replay at {path}. {reason}");
        if (stateManager != null)
            stateManager.SetError();
    }

    public void HandleDriveButton()
    {
        if (!IsLoaded || stateManager == null)
            return;

        if (stateManager.Status == ReplayStatus.Playing)
            PauseReplay();
        else if (stateManager.Status == ReplayStatus.Finished)
            RestartReplay(true);
        else
            StartDrive();
    }

    public void StartDrive()
    {
        if (!IsLoaded || stateManager == null)
        {
            Debug.LogWarning("The Drive button was pressed before the replay finished loading.");
            return;
        }

        if (stateManager.Status == ReplayStatus.Finished)
            replayTime = 0f;

        stateManager.UpdateState(replayTime);
        stateManager.SetStatus(ReplayStatus.Playing);
    }

    public void PauseReplay()
    {
        if (stateManager != null && stateManager.IsPlaying)
            stateManager.SetStatus(ReplayStatus.Paused);
    }

    public void RestartReplay(bool startPlaying = false)
    {
        if (!IsLoaded || stateManager == null)
            return;

        replayTime = 0f;
        stateManager.UpdateState(replayTime);
        stateManager.SetStatus(startPlaying ? ReplayStatus.Playing : ReplayStatus.Ready);
    }
}

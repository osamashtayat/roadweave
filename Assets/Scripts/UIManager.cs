using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class UIManager : MonoBehaviour
{
    public static UIManager Instance;

    [Header("UI References")]
    public GameObject infoPanel;
    public TMP_Text titleText;
    public TMP_Text contentText;
    [SerializeField] private Button closeButton;
    [SerializeField] private CanvasScaler canvasScaler;
    [SerializeField] private Vector2 referenceResolution = new Vector2(1920f, 1080f);

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Debug.LogWarning("More than one UIManager is active. Keeping the first instance.", this);
            return;
        }

        Instance = this;
        ConfigureResponsiveCanvas();
        if (closeButton != null)
        {
            closeButton.onClick.RemoveListener(HidePanel);
            closeButton.onClick.AddListener(HidePanel);
        }
    }

    private void Start()
    {
        if (infoPanel != null)
            infoPanel.SetActive(false);
    }

    public void ShowInformation(string title, string content)
    {
        if (titleText != null) titleText.text = title;
        if (contentText != null) contentText.text = content;

        if (infoPanel != null) infoPanel.SetActive(true);
    }

    public void HidePanel()
    {
        if (infoPanel != null) infoPanel.SetActive(false);
    }

    private void ConfigureResponsiveCanvas()
    {
        if (canvasScaler == null)
            canvasScaler = GetComponentInParent<CanvasScaler>();
        if (canvasScaler == null)
            canvasScaler = FindFirstObjectByType<CanvasScaler>();
        if (canvasScaler == null)
            return;

        canvasScaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        canvasScaler.referenceResolution = referenceResolution;
        canvasScaler.screenMatchMode = CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
        canvasScaler.matchWidthOrHeight = 0.5f;
    }

    private void OnDestroy()
    {
        if (closeButton != null)
            closeButton.onClick.RemoveListener(HidePanel);
        if (Instance == this)
            Instance = null;
    }
}

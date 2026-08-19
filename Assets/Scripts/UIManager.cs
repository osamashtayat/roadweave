using TMPro;
using UnityEngine;

public class UIManager : MonoBehaviour
{
    public static UIManager Instance;

    [Header("UI References")]
    public GameObject infoPanel;
    public TMP_Text titleText;
    public TMP_Text contentText;

    private void Awake()
    {
        Instance = this;
    }

    private void Start()
    {
        infoPanel.SetActive(false);
    }

    public void ShowInformation(string title, string content)
    {
        titleText.text = title;
        contentText.text = content;

        infoPanel.SetActive(true);
    }

    public void HidePanel()
    {
        infoPanel.SetActive(false);
    }
}
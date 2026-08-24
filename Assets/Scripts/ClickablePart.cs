using UnityEngine;

public class ClickablePart : MonoBehaviour
{
    [SerializeField] private string componentId;

    private void OnMouseDown()
    {
        if (TwinDashboard.Instance == null)
        {
            Debug.LogWarning("TwinDashboard is not available, so the selected part cannot display data.");
            return;
        }

        if (!TwinDashboard.Instance.SelectComponent(componentId))
        {
            Debug.LogWarning(
                $"Component '{componentId}' is not part of the current source-neutral dashboard contract.");
        }
    }
}

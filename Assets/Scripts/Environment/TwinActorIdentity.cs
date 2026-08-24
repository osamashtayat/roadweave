using UnityEngine;

/// <summary>
/// Stable source identity attached to a runtime actor root. Visual models can
/// be replaced without changing the identity used by sensors and adapters.
/// </summary>
[DisallowMultipleComponent]
public sealed class TwinActorIdentity : MonoBehaviour
{
    [SerializeField] private string stableId;
    [SerializeField] private string sourceKind;

    public string StableId => stableId;
    public string SourceKind => sourceKind;

    public void Configure(string newStableId, string newSourceKind)
    {
        stableId = string.IsNullOrWhiteSpace(newStableId)
            ? gameObject.name
            : newStableId;
        sourceKind = string.IsNullOrWhiteSpace(newSourceKind)
            ? "unknown"
            : newSourceKind;
    }
}

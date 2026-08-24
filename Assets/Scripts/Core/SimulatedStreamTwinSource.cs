/// <summary>
/// Entry point for a future simulated real-time pipeline. A simulator adapter
/// converts its output to TwinSnapshot and calls AcceptSnapshot or AcceptJsonMessage.
/// </summary>
public class SimulatedStreamTwinSource : TwinPushSourceBase
{
    protected override TwinSourceKind SourceKind => TwinSourceKind.SimulatedStream;
}

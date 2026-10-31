namespace RecursiveAI.OneModel.Format;

/// <summary>What kind of model file a file looks like. See <see cref="ModelFormat.Detect(string)"/>.</summary>
public enum ModelFileKind
{
    /// <summary>Neither: empty, too short, or some other format.</summary>
    Unknown = 0,

    /// <summary>An ONNX model (a protobuf <c>ModelProto</c>).</summary>
    Onnx = 1,

    /// <summary>A model in the OneModel format (starts with the magic <c>ONEMODEL</c>).</summary>
    OneModel = 2,
}

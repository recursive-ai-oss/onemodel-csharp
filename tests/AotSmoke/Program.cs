using System;
using System.Globalization;
using RecursiveAI.OneModel;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Onnx;

// Calls the library from a NativeAOT binary and prints the result; CI fails on any trim or AOT warning when publishing.
ulong hash = OneModelInfo.SelfCheck();
Console.WriteLine(hash.ToString("x16", CultureInfo.InvariantCulture));

// Imports a tiny ONNX model (ir_version 8, opset 13, one Identity node from x to y) written out byte by byte.
byte[] model =
[
    0x08, 0x08,
    0x3A, 0x1C,
        0x0A, 0x10, 0x0A, 0x01, 0x78, 0x12, 0x01, 0x79, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6E, 0x74, 0x69, 0x74, 0x79,
        0x5A, 0x03, 0x0A, 0x01, 0x78,
        0x62, 0x03, 0x0A, 0x01, 0x79,
    0x42, 0x02, 0x10, 0x0D,
];
using IrModel imported = OnnxImporter.Load(model);
bool importOk = imported.Graph.Nodes.Length == 1 && imported.Graph.Nodes[0].OpType == "Identity" && imported.Graph.Outputs[0].Name == "y"
    && imported.DefaultOpset == 13;
Console.WriteLine("import " + (importOk ? "ok" : "FAILED"));

return hash == 0x4242dc5249c33625UL && importOk ? 0 : 1;

using System;
using System.Globalization;
using OneModel.GoldenModels;
using RecursiveAI.OneModel;

// With arguments: the golden-hash runner (tools/GoldenModels), on the browser-wasm runtime.
if (args.Length > 0)
{
    return Golden.Run(args, Console.Out, Console.Error);
}

// The same call as the AOT smoke app, on the browser-wasm runtime.
ulong hash = OneModelInfo.SelfCheck();
Console.WriteLine(hash.ToString("x16", CultureInfo.InvariantCulture));
return hash == 0x4242dc5249c33625UL ? 0 : 1;

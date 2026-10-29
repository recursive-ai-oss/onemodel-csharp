using System;
using System.Globalization;
using RecursiveAI.OneModel;

// The same call as the AOT smoke app, on the browser-wasm runtime.
ulong hash = OneModelInfo.SelfCheck();
Console.WriteLine(hash.ToString("x16", CultureInfo.InvariantCulture));
return hash == 0x4242dc5249c33625UL ? 0 : 1;

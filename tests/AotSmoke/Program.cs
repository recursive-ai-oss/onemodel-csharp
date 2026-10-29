using System;
using System.Globalization;
using RecursiveAI.OneModel;

// Calls the library from a NativeAOT binary and prints the result; CI fails on any trim or AOT warning when publishing.
ulong hash = OneModelInfo.SelfCheck();
Console.WriteLine(hash.ToString("x16", CultureInfo.InvariantCulture));
return hash == 0x4242dc5249c33625UL ? 0 : 1;

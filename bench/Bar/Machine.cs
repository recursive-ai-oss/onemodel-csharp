using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace OneModel.Bar;

/// <summary>What the numbers were measured on. Every row carries it.</summary>
internal static class Machine
{
    public static string Name => Environment.MachineName;

    public static string Os => RuntimeInformation.OSDescription.Replace(',', ';');

    public static string Arch => RuntimeInformation.OSArchitecture.ToString();

    public static string Runtime => RuntimeInformation.FrameworkDescription;

    public static int Cores => Environment.ProcessorCount;

    public static string Cpu
    {
        get
        {
            try
            {
                if (File.Exists("/proc/cpuinfo"))
                {
                    foreach (string l in File.ReadLines("/proc/cpuinfo"))
                    {
                        if (l.StartsWith("model name", StringComparison.Ordinal))
                        {
                            return l[(l.IndexOf(':', StringComparison.Ordinal) + 1)..].Trim();
                        }
                    }
                }
            }
            catch (IOException)
            {
            }

            string? mac = Run("sysctl", "-n machdep.cpu.brand_string");
            return mac ?? "unknown";
        }
    }

    /// <summary>Instruction-set paths the .NET runtime reports as usable on this machine.</summary>
    public static string IsaFlags
    {
        get
        {
            var f = new List<string>();
            if (RuntimeInformation.ProcessArchitecture is Architecture.X64 or Architecture.X86)
            {
                if (Sse42.IsSupported) { f.Add("sse4.2"); }
                if (Avx2.IsSupported) { f.Add("avx2"); }
                if (Fma.IsSupported) { f.Add("fma"); }
                if (Avx512F.IsSupported) { f.Add("avx512f"); }
                if (Avx512BW.IsSupported) { f.Add("avx512bw"); }
                if (Avx512DQ.IsSupported) { f.Add("avx512dq"); }
                if (Avx512Vbmi.IsSupported) { f.Add("avx512vbmi"); }
                if (AvxVnni.IsSupported) { f.Add("avxvnni"); }
                if (Avx512F.IsSupported && IsVnni512()) { f.Add("avx512vnni"); }
            }
            else if (RuntimeInformation.ProcessArchitecture == Architecture.Arm64)
            {
                if (AdvSimd.IsSupported) { f.Add("neon"); }
                if (Dp.IsSupported) { f.Add("dotprod"); }
            }

            return string.Join(' ', f);
        }
    }

    private static bool IsVnni512()
    {
        // avx512vnni has no dedicated .NET flag: read it from the CPU feature list where available.
        try
        {
            if (File.Exists("/proc/cpuinfo"))
            {
                foreach (string l in File.ReadLines("/proc/cpuinfo"))
                {
                    if (l.StartsWith("flags", StringComparison.Ordinal))
                    {
                        return l.Contains("avx512_vnni", StringComparison.Ordinal);
                    }
                }
            }
        }
        catch (IOException)
        {
        }

        return false;
    }

    /// <summary>1-minute load average, or NaN where the host does not expose it.</summary>
    public static double Load1()
    {
        try
        {
            if (File.Exists("/proc/loadavg"))
            {
                return double.Parse(File.ReadAllText("/proc/loadavg").Split(' ')[0], CultureInfo.InvariantCulture);
            }
        }
        catch (Exception e) when (e is IOException or FormatException)
        {
        }

        string? s = Run("sysctl", "-n vm.loadavg"); // "{ 1.23 1.50 1.60 }"
        if (s is not null)
        {
            string[] p = s.Trim('{', '}', ' ', '\n').Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (p.Length > 0 && double.TryParse(p[0], NumberStyles.Float, CultureInfo.InvariantCulture, out double v))
            {
                return v;
            }
        }

        return double.NaN;
    }

    private static string? Run(string exe, string args)
    {
        try
        {
            using var p = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo(exe, args)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            });
            if (p is null)
            {
                return null;
            }

            string o = p.StandardOutput.ReadToEnd().Trim();
            p.WaitForExit();
            return p.ExitCode == 0 ? o : null;
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return null;
        }
    }
}

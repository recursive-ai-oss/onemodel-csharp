using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;

namespace OneModel.Bar;

/// <summary>A dense array read from a .npy file (format 1.x/2.x, little-endian, C order).</summary>
internal sealed class NpyArray
{
    public required string Dtype { get; init; }
    public required long[] Shape { get; init; }
    public float[]? F32 { get; init; }
    public long[]? I64 { get; init; }

    public static NpyArray Load(string path)
    {
        byte[] b = File.ReadAllBytes(path);
        if (b.Length < 10 || b[0] != 0x93 || Encoding.ASCII.GetString(b, 1, 5) != "NUMPY")
        {
            throw new InvalidDataException($"{path}: not a .npy file");
        }

        int major = b[6];
        int headerLen, headerStart;
        if (major == 1)
        {
            headerLen = BinaryPrimitives.ReadUInt16LittleEndian(b.AsSpan(8, 2));
            headerStart = 10;
        }
        else
        {
            headerLen = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(b.AsSpan(8, 4)));
            headerStart = 12;
        }

        if (headerStart + headerLen > b.Length)
        {
            throw new InvalidDataException($"{path}: header runs past the end of the file");
        }

        string header = Encoding.ASCII.GetString(b, headerStart, headerLen);
        Match d = Regex.Match(header, @"'descr':\s*'([^']+)'");
        Match f = Regex.Match(header, @"'fortran_order':\s*(True|False)");
        Match s = Regex.Match(header, @"'shape':\s*\(([^)]*)\)");
        if (!d.Success || !f.Success || !s.Success || f.Groups[1].Value == "True")
        {
            throw new InvalidDataException($"{path}: unsupported .npy header {header}");
        }

        var shape = new List<long>();
        foreach (string part in s.Groups[1].Value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            shape.Add(long.Parse(part, CultureInfo.InvariantCulture));
        }

        long count = 1;
        foreach (long x in shape)
        {
            count = checked(count * x);
        }

        string descr = d.Groups[1].Value;
        int data = headerStart + headerLen;
        switch (descr)
        {
            case "<f4":
                {
                    var a = new float[count];
                    Check(path, b, data, count * 4);
                    Buffer.BlockCopy(b, data, a, 0, a.Length * 4);
                    return new NpyArray { Dtype = descr, Shape = [.. shape], F32 = a };
                }
            case "<i8":
                {
                    var a = new long[count];
                    Check(path, b, data, count * 8);
                    Buffer.BlockCopy(b, data, a, 0, a.Length * 8);
                    return new NpyArray { Dtype = descr, Shape = [.. shape], I64 = a };
                }
            default:
                throw new InvalidDataException($"{path}: unsupported dtype {descr}");
        }
    }

    private static void Check(string path, byte[] b, int offset, long bytes)
    {
        if (!BitConverter.IsLittleEndian)
        {
            throw new PlatformNotSupportedException("big-endian hosts are not supported by the bar runner");
        }

        if (offset + bytes > b.Length)
        {
            throw new InvalidDataException($"{path}: data runs past the end of the file");
        }
    }
}

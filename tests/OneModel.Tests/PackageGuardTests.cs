using System.Text.RegularExpressions;

namespace RecursiveAI.OneModel.Tests;

/// <summary>
/// Enforces that the shipped library depends on the shared framework only. A PackageReference is also rejected at build time
/// by Directory.Build.targets; this adds the checks a build target cannot make cheaply.
/// </summary>
public partial class PackageGuardTests
{
    private static string RepoRoot([System.Runtime.CompilerServices.CallerFilePath] string path = "")
    {
        string? dir = Path.GetDirectoryName(path);
        while (dir is not null && !File.Exists(Path.Combine(dir, "global.json")))
        {
            dir = Path.GetDirectoryName(dir);
        }

        return dir ?? throw new InvalidOperationException("global.json not found above " + path);
    }

    private static IEnumerable<string> SrcFiles(string pattern) =>
        Directory.EnumerateFiles(Path.Combine(RepoRoot(), "src"), pattern, SearchOption.AllDirectories)
            .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}")
                     && !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}"));

    [Fact]
    public void SourceProjectsHaveNoPackageReference()
    {
        var projects = SrcFiles("*.csproj").ToList();
        Assert.NotEmpty(projects);
        foreach (string p in projects)
        {
            string text = File.ReadAllText(p);
            Assert.DoesNotContain("<PackageReference", text, StringComparison.Ordinal);
            Assert.DoesNotContain("<PackageVersion", text, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void SourceCodeHasNoPInvoke()
    {
        var files = SrcFiles("*.cs").ToList();
        Assert.NotEmpty(files);
        foreach (string f in files)
        {
            Assert.False(PInvoke().IsMatch(File.ReadAllText(f)), $"{f} declares a native import");
        }
    }

    [GeneratedRegex(@"\b(DllImport|LibraryImport|DefaultDllImportSearchPaths|NativeLibrary)\b")]
    private static partial Regex PInvoke();
}

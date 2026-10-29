using RecursiveAI.OneModel;

namespace RecursiveAI.OneModel.Tests;

public class SelfCheckTests
{
    [Fact]
    public void Fnv1aOfEmptyInputIsTheOffsetBasis()
    {
        Assert.Equal(14695981039346656037UL, OneModelInfo.Fnv1a([]));
    }

    [Fact]
    public void Fnv1aMatchesTheReferenceValueForA()
    {
        // Published FNV-1a 64-bit test vector: "a" hashes to af63dc4c8601ec8c.
        Assert.Equal(0xaf63dc4c8601ec8cUL, OneModelInfo.Fnv1a("a"u8));
    }

    [Fact]
    public void SelfCheckHasTheDocumentedValue()
    {
        Assert.Equal(0x4242dc5249c33625UL, OneModelInfo.SelfCheck());
    }
}

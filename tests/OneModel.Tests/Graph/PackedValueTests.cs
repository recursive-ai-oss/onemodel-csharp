using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace RecursiveAI.OneModel.Tests.Graph;

public class PackedValueTests
{
    [Fact]
    public void SequenceRoundTripsMixedShapesAndTypes()
    {
        using Tensor a = Tensor.FromSpan<float>([1, 2, 3, 4, 5, 6], Shape.Create(2, 3));
        using Tensor b = Tensor.FromSpan<float>([], Shape.Create(0, 4));
        using Tensor c = Tensor.FromSpan<float>([7], Shape.Create());
        using Tensor packed = PackedValue.Sequence([a, b, c]);
        Assert.True(PackedValue.IsPacked(packed));
        Assert.Equal(PackedKind.Sequence, PackedValue.Kind(packed));
        Tensor[] items = PackedValue.Items(packed);
        try
        {
            Assert.Equal(3, items.Length);
            Assert.Equal(Shape.Create(2, 3), items[0].Shape);
            Assert.Equal([1f, 2, 3, 4, 5, 6], items[0].Data<float>().ToArray());
            Assert.Equal(Shape.Create(0, 4), items[1].Shape);
            Assert.Equal(7f, items[2].Data<float>()[0]);
        }
        finally
        {
            foreach (Tensor t in items)
            {
                t.Dispose();
            }
        }
    }

    [Fact]
    public void OptionalsKeepTheirKind()
    {
        using Tensor a = Tensor.FromSpan<long>([9, 8], Shape.Create(2));
        using Tensor empty = PackedValue.EmptyOptional();
        using Tensor one = PackedValue.Optional(a);
        using Tensor seq = PackedValue.OptionalSequence([a, a]);
        Assert.Equal(PackedKind.OptionalEmpty, PackedValue.Kind(empty));
        Assert.Empty(PackedValue.Items(empty));
        Assert.Equal(PackedKind.OptionalTensor, PackedValue.Kind(one));
        Tensor[] got = PackedValue.Items(one);
        Assert.Equal([9L, 8], got[0].Data<long>().ToArray());
        got[0].Dispose();
        Assert.Equal(PackedKind.OptionalSequence, PackedValue.Kind(seq));
        Tensor[] two = PackedValue.Items(seq);
        Assert.Equal(2, two.Length);
        foreach (Tensor t in two)
        {
            t.Dispose();
        }
    }

    [Fact]
    public void PlainTensorsAndDamagedBytesAreRejected()
    {
        using Tensor plain = Tensor.FromSpan<byte>([1, 2, 3, 4, 5, 6], Shape.Create(6));
        Assert.False(PackedValue.IsPacked(plain));
        Assert.Throws<ArgumentException>(() => PackedValue.Items(plain));
        using Tensor a = Tensor.FromSpan<float>([1, 2, 3], Shape.Create(3));
        using Tensor packed = PackedValue.Sequence([a]);
        using Tensor cut = Tensor.FromSpan<byte>(packed.Data<byte>()[..^3], Shape.Create(packed.Length - 3));
        Assert.Throws<FormatException>(() => PackedValue.Items(cut));
    }
}

namespace RecursiveAI.OneModel.Kernels;

/// <summary>
/// Work applied to each finished mc x nc tile of C inside <see cref="Gemm"/>, while the tile is still in cache (a bias, an
/// activation). Called once per tile, possibly from several threads at once for different tiles; an implementation must not write
/// anything outside its tile and must not depend on which thread or in what order tiles arrive, so results stay independent of the
/// thread count.
/// </summary>
internal unsafe interface IGemmEpilogue
{
    /// <summary>Transforms the tile in place.</summary>
    /// <param name="tile">The tile's top-left element in C (row-major, row stride <paramref name="ldc"/>).</param>
    /// <param name="ldc">Row stride of C in elements.</param>
    /// <param name="row0">Row of C where the tile starts.</param>
    /// <param name="col0">Column of C where the tile starts.</param>
    /// <param name="rows">Rows in the tile, at least 1.</param>
    /// <param name="cols">Columns in the tile, at least 1.</param>
    /// <param name="mode">The mode the product ran in.</param>
    void Apply(float* tile, nint ldc, int row0, int col0, int rows, int cols, KernelMode mode);
}

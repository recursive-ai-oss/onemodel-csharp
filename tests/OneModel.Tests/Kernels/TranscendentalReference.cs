using System;

namespace RecursiveAI.OneModel.Tests.Kernels;

/// <summary>Double-precision references for the fp32 transcendental functions, accurate to about 1e-13 relative, and the ulp error measure.</summary>
internal static class TranscendentalReference
{
    private static readonly double SqrtPi = Math.Sqrt(Math.PI);

    public static double Exp(double x) => Math.Exp(x);

    public static double Log(double x) => Math.Log(x);

    public static double Tanh(double x) => Math.Tanh(x);

    public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    /// <summary>erf by its Maclaurin series for |x| &lt; 2.5 and 1 - erfc beyond.</summary>
    public static double Erf(double x)
    {
        double ax = Math.Abs(x);
        double r;
        if (ax < 2.5)
        {
            r = Series(ax);
        }
        else if (ax >= 6.5)
        {
            r = 1.0;
        }
        else
        {
            r = 1.0 - ErfcLarge(ax);
        }

        return x < 0 ? -r : r;
    }

    /// <summary>erfc for any finite x, with relative accuracy for large positive x (continued fraction).</summary>
    public static double Erfc(double x)
    {
        if (x < 0)
        {
            return 2.0 - Erfc(-x);
        }

        return x < 2.5 ? 1.0 - Series(x) : ErfcLarge(x);
    }

    public static double GeluErf(double x) => 0.5 * x * Erfc(-x / Math.Sqrt(2.0));

    public static double GeluTanh(double x)
    {
        double u = Math.Sqrt(2.0 / Math.PI) * (x + (0.044715 * x * x * x));
        return x * Sigmoid(2.0 * u);
    }

    private static double Series(double x)
    {
        // erf(x) = 2/sqrt(pi) * sum_n (-1)^n x^(2n+1) / (n! (2n+1))
        double term = x;
        double sum = x;
        double x2 = x * x;
        for (int n = 1; n < 80; n++)
        {
            term *= -x2 / n;
            double add = term / ((2 * n) + 1);
            sum += add;
            if (Math.Abs(add) < 1e-18 * Math.Abs(sum))
            {
                break;
            }
        }

        return 2.0 / SqrtPi * sum;
    }

    private static double ErfcLarge(double x)
    {
        // erfc(x) = exp(-x^2)/sqrt(pi) / (x + (1/2)/(x + 1/(x + (3/2)/(x + ...)))), evaluated from the tail.
        double t = x;
        for (int k = 300; k >= 1; k--)
        {
            t = x + (k / 2.0 / t);
        }

        return Math.Exp(-x * x) / SqrtPi / t;
    }

    /// <summary>Error of <paramref name="got"/> in units of the fp32 spacing at the correctly rounded reference value.</summary>
    public static double UlpError(float got, double reference)
    {
        if (double.IsNaN(reference))
        {
            return float.IsNaN(got) ? 0 : double.PositiveInfinity;
        }

        float rf = (float)reference;
        if (float.IsInfinity(rf) || float.IsInfinity(got))
        {
            return BitConverter.SingleToInt32Bits(rf) == BitConverter.SingleToInt32Bits(got) ? 0 : double.PositiveInfinity;
        }

        if (float.IsNaN(got))
        {
            return double.PositiveInfinity;
        }

        float a = Math.Abs(rf);
        double spacing = a == 0 ? Math.Pow(2, -149) : (double)MathF.BitIncrement(a) - a;
        return Math.Abs(got - reference) / spacing;
    }
}

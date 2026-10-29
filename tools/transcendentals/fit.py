#!/usr/bin/env python3
"""Fits the polynomial coefficients used by the exact and fast fp32 transcendental functions.

Run it from a venv with numpy and mpmath (see requirements.txt). It prints the coefficients as C# float literals together with the
maximum relative error of the fitted polynomial, first in exact arithmetic and then after rounding the coefficients to fp32 and
evaluating Horner's rule with one fp32 rounding per operation (no fused multiply-add), which is what the exact kernels do.

The fit is a minimax fit in relative error found by Lawson's iteratively reweighted least squares in a Chebyshev basis (Lawson 1961;
see Trefethen, "Approximation Theory and Approximation Practice", ch. 10 for the equioscillation property it converges to).
The output is deterministic: there is no random input.
"""
import sys
import numpy as np
import mpmath as mp

mp.mp.dps = 50
F32 = np.float32


def lawson_fit(target, weight, a, b, degree, powers_from=0, nodes=4000, iters=400):
    """Polynomial p(t) = sum_{k=powers_from}^{degree} c_k t^k minimising max |(p - target) * weight| on [a, b]."""
    ts = [(a + b) / 2 + (b - a) / 2 * mp.cos(mp.pi * (i + 0.5) / nodes) for i in range(nodes)]
    ts = sorted(ts)
    tv = np.array([float(t) for t in ts])
    y = np.array([float(target(t)) for t in ts])
    w = np.array([float(weight(t)) for t in ts])
    cols = list(range(powers_from, degree + 1))
    # Scale t to [-1, 1] and use a Chebyshev basis for conditioning, then convert to monomials of t.
    a_f, b_f = float(a), float(b)
    u = (2 * tv - (a_f + b_f)) / (b_f - a_f)
    # basis in t: t^k, evaluated through the scaled variable to keep the system well conditioned
    A = np.stack([np.polynomial.chebyshev.chebvander(u, degree)[:, k] for k in range(degree + 1)], axis=1)
    # Express the monomial columns t^k as Chebyshev series in u so the unknowns stay the monomial coefficients c_k.
    M = np.zeros((degree + 1, degree + 1))
    for k in range(degree + 1):
        # t = (b - a)/2 * u + (a + b)/2
        poly_u = np.polynomial.polynomial.polypow([(a_f + b_f) / 2, (b_f - a_f) / 2], k) if k > 0 else np.array([1.0])
        cheb = np.polynomial.chebyshev.poly2cheb(poly_u)
        M[: len(cheb), k] = cheb
    B = (A @ M)[:, cols]
    lam = np.full(len(tv), 1.0 / len(tv))
    best = None
    for _ in range(iters):
        sw = np.sqrt(lam) * w
        c, *_ = np.linalg.lstsq(B * sw[:, None], y * sw, rcond=None)
        err = np.abs((B @ c - y) * w)
        m = err.max()
        if best is None or m < best[0]:
            best = (m, c.copy())
        lam = lam * err
        lam /= lam.sum()
    return dict(zip(cols, best[1])), best[0]


def f32_coeffs(c):
    return {k: F32(v) for k, v in c.items()}


def horner32(cf, t, lo):
    """sum_{k>=lo} c_k t^k as c_lo + t*(c_{lo+1} + t*(...)), one fp32 rounding per operation."""
    ks = sorted(cf)
    acc = F32(cf[ks[-1]]) * np.ones_like(t)
    for k in reversed(ks[:-1]):
        acc = (acc * t).astype(F32) + F32(cf[k])
        acc = acc.astype(F32)
    return acc


def report(name, cf, mx):
    print(f"// {name}: fitted relative error {mx:.3e}")
    for k in sorted(cf):
        f = F32(cf[k])
        print(f"//   c{k} = {float(f)!r}f  (0x{np.float32(f).view(np.uint32):08X})  exact {float(cf[k])!r}")


def exp_fit(degree):
    # exp(r) = 1 + r + r^2 * Q(r), Q = sum_{k=0}^{degree-2} q_k r^k on [-ln2/2, ln2/2]
    h = mp.log(2) / 2
    tgt = lambda r: (mp.e ** r - 1 - r) / (r * r) if r != 0 else mp.mpf(1) / 2
    wgt = lambda r: r * r / mp.e ** r
    cf, mx = lawson_fit(tgt, wgt, -h, h, degree - 2)
    report(f"exp, degree {degree}: exp(r) = 1 + r + r*r*Q(r), |r| <= ln2/2", cf, mx)
    return cf


def tanh_fit(degree, zmax):
    # tanh(x) = x + x*z*P(z), z = x^2 in [0, zmax]
    tgt = lambda z: (mp.tanh(mp.sqrt(z)) / mp.sqrt(z) - 1) / z if z > 0 else mp.mpf(-1) / 3
    wgt = lambda z: z / (mp.tanh(mp.sqrt(z)) / mp.sqrt(z)) if z > 0 else mp.mpf(0)
    cf, mx = lawson_fit(tgt, wgt, mp.mpf(0), mp.mpf(zmax), degree)
    report(f"tanh, degree {degree} in z: tanh(x) = x + x*z*P(z), |x| <= {mp.sqrt(zmax)}", cf, mx)
    return cf


def erf_small_fit(degree, zmax):
    # erf(x) = x * P(z), z = x^2 in [0, zmax]
    tgt = lambda z: mp.erf(mp.sqrt(z)) / mp.sqrt(z) if z > 0 else 2 / mp.sqrt(mp.pi)
    wgt = lambda z: 1 / tgt(z)
    cf, mx = lawson_fit(tgt, wgt, mp.mpf(0), mp.mpf(zmax), degree)
    report(f"erf small, degree {degree} in z: erf(x) = x*P(z), |x| <= {mp.sqrt(zmax)}", cf, mx)
    return cf


def erfc_fit(degree, lo, hi):
    # erfc(x) = exp(-x^2) * G(t), G fitted as a polynomial in t = 1/x on [1/hi, 1/lo] of G(x) = erfc(x) exp(x^2) / ... (relative)
    tgt = lambda t: mp.erfc(1 / t) * mp.e ** (1 / (t * t))
    wgt = lambda t: 1 / tgt(t)
    cf, mx = lawson_fit(tgt, wgt, mp.mpf(1) / hi, mp.mpf(1) / lo, degree)
    report(f"erfc scaled, degree {degree} in t=1/x: erfc(x) = exp(-x*x) * G(t), x in [{lo}, {hi}]", cf, mx)
    return cf


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("exp", "all"):
        for d in (5, 6, 7):
            exp_fit(d)
    if what in ("tanh", "all"):
        for d in (5, 6, 7, 8):
            tanh_fit(d, 0.36)
    if what in ("erf", "all"):
        for d in (7, 8, 9):
            erf_small_fit(d, 1.0)
        for d in (8, 10, 12, 14):
            erfc_fit(d, 1.0, 3.92)


def erfcx_fit(degree, tmin=0):
    # erfc(x) = exp(-x^2) * t * H(t), t = 1/x in (0, 1]; H is smooth and tends to 1/sqrt(pi) as t -> 0.
    tgt = lambda t: mp.erfc(1 / t) * mp.e ** (1 / (t * t)) / t if t > 0 else 1 / mp.sqrt(mp.pi)
    wgt = lambda t: 1 / tgt(t)
    cf, mx = lawson_fit(tgt, wgt, mp.mpf(tmin), mp.mpf(1), degree)
    report(f"erfcx, degree {degree} in t=1/x: erfc(x) = exp(-x*x) * t * H(t), x >= 1", cf, mx)
    return cf


def erfcx_fit_centered(degree, tmax=1.0):
    # erfc(x) = exp(-x^2) * t * H(s) with t = 1/x in (0, tmax], fitted in the centred variable s = 2t/tmax - 1 in [-1, 1];
    # the coefficients stay small, so Horner's rule loses little. H tends to 1/sqrt(pi) as t -> 0.
    def tgt(s):
        t = (s + 1) / 2 * tmax
        return mp.erfc(1 / t) * mp.e ** (1 / (t * t)) / t if t > 0 else 1 / mp.sqrt(mp.pi)
    wgt = lambda s: 1 / tgt(s)
    cf, mx = lawson_fit(tgt, wgt, mp.mpf(-1), mp.mpf(1), degree)
    report(f"erfcx centred, degree {degree}, x >= {1 / tmax}: erfc(x) = exp(-x*x) * t * H(s), s = 2t/{tmax} - 1", cf, mx)
    return cf

#!/usr/bin/env python3
"""Fits the rational function behind the fast-mode erf Gelu (Transcendental.GeluErfFast).

    erf(x / sqrt 2) ~= x * P(x^2) / Q(x^2)      on |x| <= LIMIT, P and Q of degree 5, Q(0) = 1

The fit is a minimax fit in absolute error: a linearised least squares in a Chebyshev basis, re-weighted until the error levels
out (Lawson 1961). It prints the coefficients as C# float literals, then the error of the fit in exact arithmetic and of the
rounded coefficients evaluated in fp32 with fused multiply-adds, which is what the fast kernel does. Needs numpy only; erf comes
from the standard library. The output is deterministic.

    python3 gelu_rational.py
"""
import math

import numpy as np
from numpy.polynomial import chebyshev as C
from numpy.polynomial import polynomial as P

LIMIT = 5.6
M = N = 5
F32 = np.float32
erf = np.vectorize(math.erf)


def fit():
    count = 6000
    xs = LIMIT * 0.5 * (1 - np.cos(np.pi * (np.arange(count) + 0.5) / count))
    xs = np.concatenate([[1e-9], xs])
    y = erf(xs / math.sqrt(2))
    v = 2 * (xs / LIMIT) ** 2 - 1
    bp = C.chebvander(v, M)
    bq = C.chebvander(v, N)[:, 1:]
    w = np.ones_like(xs)
    best = None
    for _ in range(300):
        a = np.concatenate([xs[:, None] * bp, -(y[:, None] * bq)], axis=1)
        sol, *_ = np.linalg.lstsq(a * w[:, None], y * w, rcond=None)
        p, q = sol[: M + 1], sol[M + 1 :]
        f = xs * (bp @ p) / (1 + bq @ q)
        err = np.abs(f - y)
        if best is None or err.max() < best[0]:
            best = (err.max(), p.copy(), q.copy())
        w = w * np.sqrt(err / err.max() + 1e-3)
        w /= w.max()
    return best


def monomial(c):
    """Chebyshev series in v = 2 z / LIMIT^2 - 1 to monomial coefficients in z."""
    pc = C.cheb2poly(c)
    out = np.zeros(1)
    for k, a in enumerate(pc):
        out = P.polyadd(out, a * P.polypow([-1, 2], k) if k > 0 else [a])
    return np.array([out[k] / LIMIT ** (2 * k) for k in range(len(out))])


def fma32(a, b, c):
    return F32(np.float64(a) * np.float64(b) + np.float64(c))


def horner32(cf, z):
    r = np.full_like(z, F32(cf[-1]))
    for a in cf[-2::-1]:
        r = fma32(r, z, F32(a))
    return r


def main():
    err, p, q = fit()
    pz = monomial(p)
    qz = monomial(np.concatenate([[1.0], q]))
    scale = qz[0]
    pz, qz = pz / scale, qz / scale
    print(f"fit error in exact arithmetic: {err:.3e}")
    for name, c in (("P", pz), ("Q", qz)):
        print(name, ", ".join(f"{F32(v)!r}f".replace("np.float32(", "").replace(")", "") for v in c))
    x = np.linspace(0, LIMIT, 2000001).astype(F32)
    z = (x * x).astype(F32)
    e = x * (horner32(pz, z) / horner32(qz, z))
    truth = erf(x.astype(np.float64) / math.sqrt(2))
    print(f"fp32 evaluation, max abs error of erf: {np.abs(e.astype(np.float64) - truth).max():.3e}")


if __name__ == "__main__":
    main()

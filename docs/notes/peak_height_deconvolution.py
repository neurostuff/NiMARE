"""Peak-height deconvolution: the images-free correction that turned out not to work.

Kept because the machinery is validated and correct, and because a collection of
well-powered studies would give it something to work on. See section 12 of
``effect_size_cbma.md``: on the data available here the reported peak heights are
indistinguishable from peaks of pure noise, so there is no signal to deconvolve.

Peak heights as a deconvolution: peak height = noncentrality + peak overshoot.

For a smooth field with locally constant mean lambda, Y = lambda + X with X zero-mean, and
the local maxima of Y are those of X shifted by lambda. So a signal peak's height is
lambda + (height of a zero-mean peak) -- NOT lambda + N(0,1). The zero-mean peak height
already carries the "maximum over a neighbourhood" inflation, which is the whole problem.
"""

import numpy as np
from scipy.optimize import minimize

SQRT3 = np.sqrt(3.0)
_PEAK0_NORM = 2.0 * np.exp(-1.5)  # integral of x(x^2-3)exp(-x^2/2) over x >= sqrt(3)


def peak0_pdf(x):
    """Height density of a local maximum of a smooth zero-mean unit-variance 3D field."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    ok = x > SQRT3
    out[ok] = x[ok] * (x[ok] ** 2 - 3.0) * np.exp(-0.5 * x[ok] ** 2) / _PEAK0_NORM
    return out


def _lambda_grid(lo=-3.0, hi=10.0, n=260):
    return np.linspace(lo, hi, n)


def _alt_density(z, u, mu, tau, grid=None):
    """P(peak height = z, z > u) with lambda ~ N(mu, tau^2), by quadrature over lambda."""
    grid = _lambda_grid() if grid is None else grid
    prior = np.exp(-0.5 * ((grid - mu) / tau) ** 2) / (tau * np.sqrt(2 * np.pi))
    dens = peak0_pdf(z[:, None] - grid[None, :]) * prior[None, :]
    numer = np.trapezoid(dens, grid, axis=1)
    # Normalise by the mass above each peak's own threshold.
    tail = np.trapezoid(
        (1.0 - _peak0_cdf(u[:, None] - grid[None, :])) * prior[None, :], grid, axis=1
    )
    return numer, np.clip(tail, 1e-12, None)


def _peak0_cdf(x):
    """CDF of the zero-mean peak height: 1 - (x^2-1)exp(-x^2/2)/norm for x >= sqrt(3)."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    ok = x > SQRT3
    out[ok] = 1.0 - (x[ok] ** 2 - 1.0) * np.exp(-0.5 * x[ok] ** 2) / _PEAK0_NORM
    return np.clip(out, 0.0, 1.0)


def _nll(params, z, u):
    mu, log_tau = params
    tau = np.exp(log_tau)
    numer, tail = _alt_density(z, u, mu, tau)
    return -np.sum(np.log(np.clip(numer / tail, 1e-300, None)))


def fit(z, u):
    """Estimate the (mu, tau) of the noncentrality distribution behind the reported peaks."""
    z = np.abs(np.asarray(z, float))
    u = np.abs(np.asarray(u, float))
    best = None
    for mu0 in (0.0, 0.5, 1.5, 3.0):
        out = minimize(
            _nll,
            [mu0, np.log(1.0)],
            args=(z, u),
            method="Nelder-Mead",
            options=dict(maxiter=2000, xatol=1e-5, fatol=1e-5),
        )
        if best is None or out.fun < best.fun:
            best = out
    return float(best.x[0]), float(np.exp(best.x[1]))


def shrink(z, u, mu, tau):
    """Posterior mean noncentrality for each reported peak."""
    sign = np.sign(z)
    az = np.abs(np.asarray(z, float))
    u = np.abs(np.asarray(u, float))
    grid = _lambda_grid()
    prior = np.exp(-0.5 * ((grid - mu) / tau) ** 2) / (tau * np.sqrt(2 * np.pi))
    dens = peak0_pdf(az[:, None] - grid[None, :]) * prior[None, :]
    mass = np.trapezoid(dens, grid, axis=1)
    mean = np.trapezoid(dens * grid[None, :], grid, axis=1)
    return sign * np.where(mass > 1e-300, mean / np.clip(mass, 1e-300, None), 0.0)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "docs/notes")
    from validate_cbes import true_field, study_image, report_peaks

    U = 3.2905
    for peak_g, smooth in ((0.6, 2.0), (1.0, 2.0), (0.6, 3.0)):
        rng = np.random.default_rng(0)
        truth = true_field(peak_g=peak_g)
        z, ns, tg = [], [], []
        for _ in range(40):
            n = int(rng.integers(15, 40))
            img = study_image(truth, n, smooth, rng)
            hits, vals = report_peaks(img, n, U)
            if not len(hits):
                continue
            z.extend(np.asarray(vals) * np.sqrt(n))
            ns.extend([n] * len(hits))
            tg.extend(truth[tuple(hits.T)])
        z, ns, tg = np.array(z), np.array(ns), np.array(tg)
        mu, tau = fit(z, np.full(len(z), U))
        lam = shrink(z, np.full(len(z), U), mu, tau)
        print(
            f"peak_g={peak_g} smooth={smooth}: {len(z):4d} peaks | mu={mu:5.2f} tau={tau:4.2f} | "
            f"true g {tg.mean():.3f} | raw {(z/np.sqrt(ns)).mean():.3f} | "
            f"corrected {(lam/np.sqrt(ns)).mean():.3f}"
        )

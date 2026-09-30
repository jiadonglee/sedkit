"""Two solutions of a Gaia photocentre orbit: a faint companion or a luminous near-twin.

A photocentre orbit fixes one combination of the mass ratio q and the G-band
flux ratio beta = F2/F1 (Shahaf et al. 2019),

    A = a0 / parallax * (M1/Msun)**(-1/3) * (P/yr)**(-2/3)
      = (q - beta) / ((1 + q)**(2/3) * (1 + beta)).

For a coeval main-sequence companion the stellar model gives beta(q), and
A(q, beta(q)) rises and then falls towards q = 1, so one orbit usually has
two solutions: a faint companion (low q, near the dark-companion value) and a
luminous one (near equal mass). The luminous solution predicts how much
brighter than a single star the system is; an SED fit at each solution
decides between them.

    from sedkit.orbit import solve_orbit, rank_roots
    roots = solve_orbit(a0_mas=0.6978, parallax_mas=13.913, period_day=339.57, m1=0.686)
    ranked = rank_roots(sed, roots)          # needs an observed SED

Command line: python -m sedkit.orbit --a0 0.6978 --parallax 13.913 --period 339.57 --m1 0.686
"""

from dataclasses import replace

import numpy as np
from scipy.optimize import brentq

from .model import StellarModel

Q_GRID = np.round(np.arange(0.10, 1.0001, 0.01), 2)
KS = 63  # channel index of Ks in the 168-channel model output


def amrf(q, beta):
    """Astrometric mass-ratio function A(q, beta)."""
    q, beta = np.asarray(q, float), np.asarray(beta, float)
    return (q - beta) / ((1 + q)**(2 / 3) * (1 + beta))


def amrf_observed(a0_mas, parallax_mas, period_day, m1):
    """A of an orbit from a0, the two-body parallax, the period and the primary mass."""
    return a0_mas / parallax_mas * m1**(-1 / 3) * (period_day / 365.25)**(-2 / 3)


def dark_mass_ratio(a_obs):
    """q of a dark companion (beta = 0); NaN when no q <= 1 reaches A."""
    if not a_obs < amrf(1.0, 0.0):
        return np.nan
    return brentq(lambda q: amrf(q, 0.0) - a_obs, 1e-6, 1.0)


def locus(m1, age_gyr=5.0, feh=0.0, model=None):
    """beta_G(q) and beta_Ks(q) of the coeval companion on Q_GRID; NaN where unsupported."""
    model = StellarModel() if model is None else model
    beta_g, beta_k = np.full(len(Q_GRID), np.nan), np.full(len(Q_GRID), np.nan)
    for i, q in enumerate(Q_GRID):
        pair = model.evaluate(m1, float(q), age_gyr, feh)
        if pair is not None:
            beta_g[i] = pair["beta_g"]
            beta_k[i] = pair["components"][1, KS] / pair["components"][0, KS]
    return beta_g, beta_k


def _brightening(beta):
    return float(2.5 * np.log10(1 + beta))


def solve_amrf(a_obs, m1, age_gyr=5.0, feh=0.0, model=None):
    """Solutions of A(q, beta(q)) = a_obs, faintest first.

    Each is a dict with kind ('dark': the beta = 0 root below the model's
    lowest supported q; 'faint': below the maximum of A(q, beta(q));
    'luminous': above it), q, m2 [Msun], beta_G, and delta_G and delta_Ks,
    the brightening of the pair over the primary alone in magnitudes.
    """
    beta_g, beta_k = locus(m1, age_gyr, feh, model)
    curve = amrf(Q_GRID, beta_g)
    ok = np.isfinite(curve)
    out = []
    first = np.flatnonzero(ok)[0] if ok.any() else None
    if first is not None and curve[first] > a_obs:
        q = dark_mass_ratio(a_obs)
        if np.isfinite(q) and q < Q_GRID[first]:
            out.append(dict(kind="dark", q=float(q), m2=float(q * m1), beta_G=0.0,
                            delta_G=0.0, delta_Ks=0.0))
    if first is None:
        return out
    peak = Q_GRID[np.nanargmax(curve)]
    d = curve - a_obs
    for j in np.flatnonzero(ok[:-1] & ok[1:] & (np.sign(d[:-1]) != np.sign(d[1:]))):
        w = -d[j] / (d[j + 1] - d[j])
        q = Q_GRID[j] + w * (Q_GRID[j + 1] - Q_GRID[j])
        bg = beta_g[j] + w * (beta_g[j + 1] - beta_g[j])
        bk = beta_k[j] + w * (beta_k[j + 1] - beta_k[j])
        out.append(dict(kind="luminous" if q > peak else "faint", q=float(q), m2=float(q * m1),
                        beta_G=float(bg), delta_G=_brightening(bg), delta_Ks=_brightening(bk)))
    return out


def solve_orbit(a0_mas, parallax_mas, period_day, m1, age_gyr=5.0, feh=0.0, model=None):
    """Solutions for a photocentre orbit; use the parallax of the two-body solution."""
    return solve_amrf(amrf_observed(a0_mas, parallax_mas, period_day, m1), m1, age_gyr, feh, model)


def rank_roots(sed, roots, *, parallax_mas=None, parallax_error_mas=None, model=None, **fit_kwargs):
    """Fit the SED at each solution and rank them by the fit objective.

    A luminous or faint root is fitted as a binary with q fixed at the root
    and the primary refitted; a dark root, or a root whose companion lies
    outside the model, as a single star. Pass the parallax of the two-body
    orbit, which can differ from the single-star catalogue value.

    Returns copies of the roots with the fit and its objective ('fit',
    'objective', 'chi2', 'n_fit') and 'delta', the objective above the best
    root, sorted best first. A delta of order 25 or more between the two
    branches is a clear choice; a small delta leaves both open.
    """
    from .fit import fit

    model = StellarModel() if model is None else model
    if parallax_mas is not None:
        sed = replace(sed, parallax_mas=parallax_mas,
                      parallax_error_mas=sed.parallax_error_mas if parallax_error_mas is None
                      else parallax_error_mas)
    ranked = []
    for root in roots:
        binary = root["kind"] != "dark" and root["q"] >= 0.1
        result = fit(sed, "binary" if binary else "single", model=model,
                     q=root["q"] if binary else None, **fit_kwargs)
        ranked.append(dict(root, fitted_as="binary" if binary else "single", fit=result,
                           objective=float(result["objective"]), chi2=float(result["chi2"]),
                           n_fit=int(result["n_fit"])))
    best = min((r["objective"] for r in ranked), default=np.nan)
    for r in ranked:
        r["delta"] = r["objective"] - best
    return sorted(ranked, key=lambda r: r["delta"])


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--a0", type=float, help="photocentre semi-major axis [mas]")
    p.add_argument("--parallax", type=float, help="parallax of the two-body solution [mas]")
    p.add_argument("--period", type=float, help="period [day]")
    p.add_argument("--amrf", type=float, help="A instead of a0, parallax and period")
    p.add_argument("--m1", type=float, required=True, help="primary mass [Msun]")
    p.add_argument("--age", type=float, default=5.0, help="age [Gyr]")
    p.add_argument("--feh", type=float, default=0.0, help="[M/H]")
    a = p.parse_args(argv)
    if a.amrf is None and None in (a.a0, a.parallax, a.period):
        p.error("give --amrf, or --a0, --parallax and --period")
    a_obs = a.amrf if a.amrf is not None else amrf_observed(a.a0, a.parallax, a.period, a.m1)
    print(f"A = {a_obs:.4f}, dark-companion q = {dark_mass_ratio(a_obs):.3f}, M1 = {a.m1:g} Msun")
    print(f'{"kind":9s} {"q":>6s} {"M2[Msun]":>9s} {"beta_G":>7s} {"dG[mag]":>8s} {"dKs[mag]":>9s}')
    for s in solve_amrf(a_obs, a.m1, a.age, a.feh):
        print(f"{s['kind']:9s} {s['q']:6.3f} {s['m2']:9.3f} {s['beta_G']:7.3f} "
              f"{s['delta_G']:8.3f} {s['delta_Ks']:9.3f}")


if __name__ == "__main__":
    main()

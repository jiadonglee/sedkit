"""Two solutions of a Gaia photocentre orbit: a faint companion or a luminous near-twin.

A photocentre orbit fixes one combination of the mass ratio q and the G-band
flux ratio beta = F2/F1 (Shahaf et al. 2019),

    A = a0 / parallax * (M1/Msun)**(-1/3) * (P/yr)**(-2/3)
      = (q - beta) / ((1 + q)**(2/3) * (1 + beta)).

For a coeval main-sequence companion the stellar model gives beta(q), and
A(q, beta(q)) rises and then falls towards q = 1, so one orbit usually has
two solutions: a faint companion (low q, near the dark-companion value) and a
luminous one (near equal mass). The luminous solution predicts how much
brighter than a single star the system is; an SED fit along each solution
branch, with q re-solved from the orbit at every trial primary, decides
between them.

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


def _beta_g(model, masses, age_gyr, feh):
    """PARSEC beta_G of each companion mass to masses[0]; NaN where a star lies outside the model."""
    labels, physical = model._track(masses, age_gyr, feh)
    ok = np.isfinite(labels).all(1) & np.isfinite(physical).all(1)
    weight = model.hot_weight(labels[:, 0])
    cool, hot = ok & (weight < 1), ok & (weight > 0)
    ok[cool] &= model.in_domain(labels[cool])
    if hot.any():
        ok[hot] &= model.in_hot_domain(labels[hot, 0], physical[hot, 0], feh)
    mg = labels[:, 1] + labels[:, 2]
    beta = 10**(-0.4 * (mg[1:] - mg[0]))
    return np.where(ok[0] & ok[1:], beta, np.nan)


def branch_mass_ratio(kind, a_obs, m1, age_gyr=5.0, feh=0.0, model=None, near=None):
    """q of the given kind ('dark', 'faint' or 'luminous') at A = a_obs; NaN if the branch has none.

    A faint or luminous q is bracketed on Q_GRID and refined on the PARSEC beta_G; with several
    crossings of one kind, the one closest to near is returned.
    """
    if kind == "dark":
        return dark_mass_ratio(a_obs)
    model = StellarModel() if model is None else model
    curve = amrf(Q_GRID, _beta_g(model, m1 * np.r_[1.0, Q_GRID], age_gyr, feh))
    ok = np.isfinite(curve)
    if not ok.any():
        return np.nan
    peak = Q_GRID[np.nanargmax(curve)]
    d = curve - a_obs
    roots = []
    for j in np.flatnonzero(ok[:-1] & ok[1:] & (np.sign(d[:-1]) != np.sign(d[1:]))):
        if (Q_GRID[j] >= peak) == (kind == "luminous"):
            f = lambda q: amrf(q, _beta_g(model, np.array([m1, m1 * q]), age_gyr, feh)[0]) - a_obs
            roots.append(brentq(f, Q_GRID[j], Q_GRID[j + 1], xtol=1e-6))
    if not roots:
        return np.nan
    return min(roots, key=lambda q: abs(q - (roots[0] if near is None else near)))


def solve_amrf(a_obs, m1, age_gyr=5.0, feh=0.0, model=None):
    """Solutions of A(q, beta(q)) = a_obs, faintest first.

    Each is a dict with kind ('dark': the beta = 0 root below the model's
    lowest supported q; 'faint': below the maximum of A(q, beta(q));
    'luminous': above it), q, m1 and m2 [Msun], beta_G, delta_G and delta_Ks,
    the brightening of the pair over the primary alone in magnitudes, and amrf.
    """
    beta_g, beta_k = locus(m1, age_gyr, feh, model)
    curve = amrf(Q_GRID, beta_g)
    ok = np.isfinite(curve)
    out = []
    first = np.flatnonzero(ok)[0] if ok.any() else None
    if first is not None and curve[first] > a_obs:
        q = dark_mass_ratio(a_obs)
        if np.isfinite(q) and q < Q_GRID[first]:
            out.append(dict(kind="dark", q=float(q), m1=float(m1), m2=float(q * m1), beta_G=0.0,
                            delta_G=0.0, delta_Ks=0.0, amrf=float(a_obs)))
    if first is None:
        return out
    peak = Q_GRID[np.nanargmax(curve)]
    d = curve - a_obs
    for j in np.flatnonzero(ok[:-1] & ok[1:] & (np.sign(d[:-1]) != np.sign(d[1:]))):
        w = -d[j] / (d[j + 1] - d[j])
        q = Q_GRID[j] + w * (Q_GRID[j + 1] - Q_GRID[j])
        bg = beta_g[j] + w * (beta_g[j + 1] - beta_g[j])
        bk = beta_k[j] + w * (beta_k[j + 1] - beta_k[j])
        out.append(dict(kind="luminous" if q > peak else "faint", q=float(q), m1=float(m1),
                        m2=float(q * m1), beta_G=float(bg), delta_G=_brightening(bg),
                        delta_Ks=_brightening(bk), amrf=float(a_obs)))
    return out


def solve_orbit(a0_mas, parallax_mas, period_day, m1, age_gyr=5.0, feh=0.0, model=None):
    """Solutions for a photocentre orbit; use the parallax of the two-body solution."""
    roots = solve_amrf(amrf_observed(a0_mas, parallax_mas, period_day, m1), m1, age_gyr, feh, model)
    return [dict(r, parallax_mas=float(parallax_mas)) for r in roots]


def rank_roots(sed, roots, *, parallax_mas=None, parallax_error_mas=None, model=None, **fit_kwargs):
    """Fit the SED along each solution branch and rank the branches by the fit objective.

    The orbit fixes A * M1**(1/3) * parallax. A luminous or faint root is
    fitted as a binary whose q is re-solved on its branch at every trial
    primary mass, age, metallicity and parallax, so the fitted pair reproduces
    the orbit; a dark root, or a root whose companion lies below the model,
    is fitted as a single star. Pass the parallax of the two-body orbit,
    which can differ from the single-star catalogue value.

    Returns copies of the roots with q, m1, m2, beta_G, delta_G, delta_Ks and
    amrf at the fitted parameters, the fit and its objective ('fit',
    'objective', 'chi2', 'n_fit') and 'delta', the objective above the best
    root, sorted best first. A delta of order 25 or more between the two branches is a clear choice;
    a small delta leaves both open.
    """
    from .fit import fit

    model = StellarModel() if model is None else model
    if parallax_mas is not None:
        sed = replace(sed, parallax_mas=parallax_mas,
                      parallax_error_mas=sed.parallax_error_mas if parallax_error_mas is None
                      else parallax_error_mas)
    ranked = []
    for root in roots:
        # A * M1**(1/3) * parallax is fixed by a0 and the period
        invariant = root["amrf"] * root["m1"]**(1 / 3) * root.get("parallax_mas", sed.parallax_mas)

        def a_obs(m1, parallax):
            return invariant / (m1**(1 / 3) * parallax)

        def q_branch(m1, age_gyr, feh, parallax, kind=root["kind"], near=root["q"]):
            return branch_mass_ratio(kind, a_obs(m1, parallax), m1, age_gyr, feh, model, near)

        binary = root["kind"] != "dark" and root["q"] >= 0.1
        result = fit(sed, "binary" if binary else "single", model=model,
                     q=q_branch if binary else None, **fit_kwargs)
        m1, a_fit = result["m1"], a_obs(result["m1"], result["parallax_mas"])
        if binary:
            parts = result["components"]
            q, beta = result["q"], result["beta_g"]
            delta_ks = _brightening(parts[1, KS] / parts[0, KS])
        else:
            q, beta, delta_ks = dark_mass_ratio(a_fit), 0.0, 0.0
        ranked.append(dict(root, q=float(q), m1=float(m1), m2=float(q * m1), beta_G=float(beta),
                           delta_G=_brightening(beta), delta_Ks=delta_ks, amrf=float(a_fit),
                           fitted_as="binary" if binary else "single", fit=result,
                           objective=float(result["objective"]), chi2=float(result["chi2"]),
                           n_fit=int(result["n_fit"])))
    best = min((r["objective"] for r in ranked), default=np.nan)
    for r in ranked:
        r["delta"] = r["objective"] - best
    return sorted(ranked, key=lambda r: r["delta"])



# ---------------------------------------------------------------- two luminous stars of different kinds

def relative_semimajor_mas(m_tot, period_day, parallax_mas):
    """Angular relative semi-major axis a = parallax (M_tot P**2)**(1/3), with P in years."""
    return parallax_mas * (m_tot * (period_day / 365.25)**2)**(1 / 3)


def photocentre_a0(m1, m2, beta, period_day, parallax_mas):
    """Photocentre semi-major axis a |B - beta| of star 1 (mass m1, G-band light fraction beta) and star 2.

    B = m1 / (m1 + m2). With r the vector from star 1 to star 2, the photocentre lies at (B - beta) r from
    the barycentre: for B > beta it moves with star 2, for B < beta with star 1.
    """
    b = m1 / (m1 + m2)
    return relative_semimajor_mas(m1 + m2, period_day, parallax_mas) * abs(b - beta)


def thiele_innes(a0, inclination_deg, omega_deg, node_deg):
    """Thiele-Innes A, B, F, G of an orbit (same unit as a0); (omega + 180, node + 180) gives the same."""
    w, n, i = np.radians([omega_deg, node_deg, inclination_deg])
    return (a0 * (np.cos(w) * np.cos(n) - np.sin(w) * np.sin(n) * np.cos(i)),
            a0 * (np.cos(w) * np.sin(n) + np.sin(w) * np.cos(n) * np.cos(i)),
            a0 * (-np.sin(w) * np.cos(n) - np.cos(w) * np.sin(n) * np.cos(i)),
            a0 * (-np.sin(w) * np.sin(n) + np.cos(w) * np.cos(n) * np.cos(i)))


def campbell(a, b, f, g):
    """a0, inclination, omega and node (degrees) from Thiele-Innes elements.

    The photocentre orbit fixes (omega, node) only up to (omega + 180, node + 180): this returns the
    solution with 0 <= node < 180, the Gaia DR3 convention. Astrometry therefore cannot tell whether the
    photocentre moves with one star or the other from the phase of either star's radial velocities.
    """
    u = (a * a + b * b + f * f + g * g) / 2
    v = a * g - b * f
    a0 = np.sqrt(u + np.sqrt((u + v) * (u - v)))
    inclination = np.degrees(np.arccos(np.clip(v / a0**2, -1, 1)))
    plus, minus = np.arctan2(b - f, a + g), np.arctan2(-b - f, a - g)
    omega, node = np.degrees((plus + minus) / 2), np.degrees((plus - minus) / 2)
    if node < 0:
        omega, node = omega + 180, node + 180
    if node >= 180:
        omega, node = omega - 180, node - 180
    return dict(a0=float(a0), inclination=float(inclination), omega=float(omega % 360), node=float(node))


def _branch_mass(sign, a_obs, m2, beta):
    """m1 with M_tot**(1/3) * sign * (B - beta) = a_obs; NaN if the branch has no root.

    sign = +1 (B > beta) rises monotonically in m1 from below zero; sign = -1 (B < beta) falls from
    beta m2**(1/3), so each branch has at most one root.
    """
    f = lambda m1: (m1 + m2)**(1 / 3) * sign * (m1 / (m1 + m2) - beta) - a_obs
    low, high = 1e-6, 1e3
    if sign > 0:
        return brentq(f, low, high, xtol=1e-8) if f(low) < 0 < f(high) else np.nan
    m_cross = beta * m2 / (1 - beta) if beta < 1 else high   # B = beta
    if not (f(low) > 0 > f(min(m_cross, high))):
        return np.nan
    return brentq(f, low, min(m_cross, high), xtol=1e-8)


def solve_luminous_pair(a0_mas, parallax_mas, period_day, m2, beta, *, errors=None):
    """Mass of luminous star 1 (e.g. a hot subdwarf) from a photocentre orbit, star 2's mass and the G-band
    light fraction beta of star 1, on both branches.

    a0 = a |B - beta| with B = m1 / (m1 + m2) and a = parallax (M_tot P**2)**(1/3); the AMRF of the orbit
    is A = a0 / (parallax P**(2/3)) = M_tot**(1/3) |B - beta|. Returns one dict per branch, 'B>beta' (the
    photocentre moves with star 2) and 'B<beta' (with star 1): m1, B, m_tot and a_mas, NaN where the branch
    has no root. errors={'a0': s, 'parallax': s, 'm2': s, 'beta': s} adds 'sigma_m1', the quadrature sum of
    the m1 changes for one-sigma steps in each, and 'dm1_dbeta'.
    """
    a_obs = a0_mas / (parallax_mas * (period_day / 365.25)**(2 / 3))
    out = []
    for name, sign in (("B>beta", 1), ("B<beta", -1)):
        m1 = _branch_mass(sign, a_obs, m2, beta)
        row = dict(branch=name, m1=float(m1), amrf=float(a_obs))
        if np.isfinite(m1):
            row.update(B=float(m1 / (m1 + m2)), m_tot=float(m1 + m2),
                       a_mas=float(relative_semimajor_mas(m1 + m2, period_day, parallax_mas)))
        if errors and np.isfinite(m1):
            terms = {}
            base = dict(a0=a0_mas, parallax=parallax_mas, m2=m2, beta=beta)
            for key, sigma in errors.items():
                shifted = dict(base, **{key: base[key] + sigma})
                a_s = shifted["a0"] / (shifted["parallax"] * (period_day / 365.25)**(2 / 3))
                terms[key] = _branch_mass(sign, a_s, shifted["m2"], shifted["beta"]) - m1
            row["sigma_m1"] = float(np.sqrt(np.nansum([t * t for t in terms.values()])))
            row["sigma_terms"] = {k: float(v) for k, v in terms.items()}
            h = 1e-4
            row["dm1_dbeta"] = float((_branch_mass(sign, a_obs, m2, beta + h) - m1) / h)
        out.append(row)
    return out


def mass_from_rv(k2_kms, period_day, eccentricity, inclination_deg, m2):
    """Mass of star 1 from the RV semi-amplitude of star 2, the inclination and m2 (independent of beta).

    Star 2's mass function f = P K2**3 (1 - e**2)**(3/2) / (2 pi G) = (m1 sin i)**3 / (m1 + m2)**2.
    """
    g, msun, day = 6.674e-11, 1.989e30, 86400.0
    f = period_day * day * (k2_kms * 1e3)**3 * (1 - eccentricity**2)**1.5 / (2 * np.pi * g) / msun
    s3 = np.sin(np.radians(inclination_deg))**3
    return float(brentq(lambda m1: m1**3 * s3 / (m1 + m2)**2 - f, 1e-6, 1e3, xtol=1e-9))


def branch_from_rv(a0_mas, parallax_mas, period_day, m2, k2_kms, eccentricity, inclination_deg,
                   beta_sed=None, beta_sed_error=None):
    """Choose the branch of a luminous pair with the RV orbit of star 2.

    K2, the inclination and m2 give m1 and so B without beta; a0 / a then gives beta = B - a0/a on the
    B > beta branch and B + a0/a on the B < beta branch. A branch is allowed when its beta lies in [0, 1];
    with beta_sed (the SED light fraction of star 1) the allowed branch closest to it in units of
    beta_sed_error is chosen. The phase of the RV curve does not decide: see campbell().
    """
    m1 = mass_from_rv(k2_kms, period_day, eccentricity, inclination_deg, m2)
    b = m1 / (m1 + m2)
    a = relative_semimajor_mas(m1 + m2, period_day, parallax_mas)
    rows = []
    for name, beta in (("B>beta", b - a0_mas / a), ("B<beta", b + a0_mas / a)):
        row = dict(branch=name, m1=m1, B=b, a_mas=a, beta=float(beta), allowed=bool(0 <= beta <= 1))
        if beta_sed is not None:
            row["beta_minus_sed"] = float(beta - beta_sed)
            if beta_sed_error:
                row["z"] = float((beta - beta_sed) / beta_sed_error)
        rows.append(row)
    allowed = [r for r in rows if r["allowed"]]
    if beta_sed is not None and allowed:
        choice = min(allowed, key=lambda r: abs(r["beta_minus_sed"]))["branch"]
    else:
        choice = allowed[0]["branch"] if len(allowed) == 1 else None
    return dict(m1=m1, B=b, a_mas=a, branches=rows, chosen=choice)


def solve_dark_companion(a0_mas, parallax_mas, period_day, m1, beta=0.0):
    """Companion mass of a luminous star 1 (e.g. a hot subdwarf) whose companion is faint in G.

    beta = F2 / F1 in G, as in amrf(); A(q, beta) rises monotonically for q > beta, so the root is unique
    and q may exceed 1. Returns q and m2, NaN if the orbit is too small for any q.
    """
    a_obs = amrf_observed(a0_mas, parallax_mas, period_day, m1)
    f = lambda q: float(amrf(q, beta)) - a_obs
    if not f(beta + 1e-9) < 0 < f(100.0):
        return dict(q=np.nan, m2=np.nan, amrf=float(a_obs))
    q = brentq(f, beta + 1e-9, 100.0, xtol=1e-9)
    return dict(q=float(q), m2=float(q * m1), amrf=float(a_obs))


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

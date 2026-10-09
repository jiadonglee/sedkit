# Photocentre orbits: faint companion or hidden twin?

A Gaia astrometric orbit follows the photocentre of the pair. It fixes the
astrometric mass-ratio function (Shahaf et al. 2019)

    A = a0 / parallax * (M1/Msun)^(-1/3) * (P/yr)^(-2/3)
      = (q - beta) / ((1 + q)^(2/3) * (1 + beta)),

one combination of the mass ratio `q` and the G-band flux ratio
`beta = F2/F1`, not `q` itself. For a coeval main-sequence companion
`beta` rises with `q`, so `A(q, beta(q))` rises and then falls towards
`q = 1`. One orbit therefore usually has two solutions: a faint companion
near the dark-companion value, and a luminous companion of nearly equal
mass whose light almost cancels the photocentre motion. Astrometric
brown-dwarf and planet candidates with a small `A` are exactly where the
second solution hides.

## Three checks

```python
from sedkit import download
from sedkit.orbit import solve_orbit, rank_roots

# LP 769-9: Gaia DR3 Orbital solution, two-body parallax 13.913 mas
roots = solve_orbit(a0_mas=0.6978, parallax_mas=13.913, period_day=339.57, m1=0.686)
for r in roots:
    print(r["kind"], r["q"], r["m2"], r["beta_G"], r["delta_G"], r["delta_Ks"])

sed = download("5148853253106611200")
for r in rank_roots(sed, roots, parallax_mas=13.913, fit_parallax=True):
    print(r["kind"], r["q"], r["delta"])
```

1. `solve_orbit` returns every solution: `kind` (`dark`, `faint` or
   `luminous`), `q`, `m2`, `beta_G`, and `delta_G`, `delta_Ks`, the
   predicted brightening of the pair over a single primary in magnitudes.
   `solve_amrf(A, m1)` takes `A` directly. No spectrum is needed.
2. Compare `delta_G` or `delta_Ks` with the height of the star above the
   single-star sequence of a colour--magnitude diagram. A metal-poor primary
   lies below the solar-metallicity sequence and can hide the excess.
3. `rank_roots` fits the observed SED along each solution branch: as a
   binary at a luminous or faint root, with `q` re-solved from the orbit on
   that branch at every trial primary mass, age, metallicity and parallax,
   and as a single star at a dark root. The returned `q`, `m1`, `m2`,
   `beta_G` and brightenings are those of the fitted system, which
   reproduces the orbit. It returns the solutions best first with `delta`,
   the fit objective above the best one. Use the two-body parallax.

Command line, for the first check:

```bash
python -m sedkit.orbit --a0 0.6978 --parallax 13.913 --period 339.57 --m1 0.686
python -m sedkit.orbit --amrf 0.14 --m1 0.50
```

## Two observed candidates

The [notebook](../examples/02_gaia_orbit_twin.ipynb) applies the three checks to two
Gaia DR3 astrometric substellar candidates with radial-velocity follow-up,
from downloaded XP and 2MASS photometry.

| Source | Follow-up | Luminous root | Predicted ΔKs | Preferred root | Δ objective |
| --- | --- | ---: | ---: | --- | ---: |
| Gaia DR3 1916454200349735680 | near-equal-mass binary | q = 0.99 | 0.72 mag | luminous | 659 |
| LP 769-9 | substellar companion | q = 0.97 | 0.70 mag | dark | 929 |

The binary lies 0.68 mag above the single-star sequence in Ks, as the
luminous root predicts; LP 769-9 lies 0.15 mag above it. The SED makes the
same choice from the spectrum alone.

## Luminous companions of a different kind

A hot subdwarf with an F, G or K companion is a pair of two luminous
stars of different ages; neither is on the other's isochrone. With B the
subdwarf's mass fraction and beta_G its share of the G-band light, the
photocentre sits at (B - beta_G) r from the barycentre, r pointing from the
subdwarf to the companion, so

    a0 = a |B - beta_G|,    a = parallax (M_tot P^2)^(1/3),
    A  = a0 / (parallax P^(2/3)) = M_tot^(1/3) |B - beta_G|.

For B > beta_G the photocentre moves with the companion, for B < beta_G
with the subdwarf.

```python
from sedkit.orbit import solve_luminous_pair, branch_from_rv, campbell
for row in solve_luminous_pair(a0_mas=0.30, parallax_mas=2.0, period_day=800,
                               m2=1.1, beta=0.45, errors=dict(beta=0.03, a0=0.02)):
    print(row["branch"], row["m1"], row["B"], row["sigma_m1"], row["dm1_dbeta"])
```

- `solve_luminous_pair(a0_mas, parallax_mas, period_day, m2, beta, errors=)`
  returns the subdwarf mass m1 on both branches, `"B>beta"` and
  `"B<beta"`. On the first, A rises monotonically with m1 and has one root;
  on the second, A falls from beta m2^(1/3) and has a root only for smaller
  A. Here m2 is the companion's mass from the SED fit and beta its
  `fractions["beta_G"]`. `errors` gives one-sigma steps in `a0`,
  `parallax`, `m2` and `beta`; `sigma_m1` is their quadrature sum and
  `dm1_dbeta` the sensitivity to beta, 2.3--2.6 for typical sdB + F/G
  systems.
- `mass_from_rv(k2_kms, period_day, eccentricity, inclination_deg, m2)`
  gives m1 from the companion's RV orbit and the astrometric inclination,
  independent of beta: f = P K2^3 (1 - e^2)^(3/2) / (2 pi G)
  = (m1 sin i)^3 / M_tot^2.
- `branch_from_rv(...)` then fixes B and a, and gives beta on each branch,
  B - a0/a and B + a0/a. A branch is allowed when beta lies in [0, 1]; with
  the SED beta_G the allowed branch closest to it is chosen, and the
  difference is the consistency test of the light scale.
- `solve_dark_companion(a0_mas, parallax_mas, period_day, m1, beta=0)`
  covers a subdwarf with a companion faint in G (K/M dwarf, white dwarf):
  the subdwarf is the luminous star, `amrf` with beta = F2/F1 from the SED
  gives q and the companion mass, and q may exceed 1.
- `thiele_innes` and `campbell` convert between Thiele--Innes and Campbell
  elements in the Gaia DR3 convention (0 <= node < 180 degrees).

The phase of the companion's RV curve against the photocentre orbit does
not choose the branch. A photocentre orbit fixes (omega, node) only up to
(omega + 180, node + 180): the Thiele--Innes elements are unchanged by that
flip. "The photocentre follows the companion" and "the photocentre follows
the subdwarf, with the other node ascending" fit the same astrometry and
the same single-star RV curve. The branch follows from the masses: the RV
amplitude and inclination give B without beta, and a0 then gives beta,
which the SED decides between.

## Scope

- The companion is a coeval main-sequence star of the bundled model; white
  dwarfs and unresolved triples are not modelled. A spectrum that rejects
  both main-sequence solutions points to a dark or compact companion.
- Primary mass, age and metallicity are inputs (defaults 5 Gyr, solar);
  `rank_roots` accepts the [extinction fit options](extinction.md).
  The primary mass enters `A` as `M1^(-1/3)`.
- `beta_G` comes from the PARSEC G magnitudes of the two components, on
  the Gaia DR2 G passband of the bundled tracks and without extinction.
  A reddened pair of unequal temperature needs the attenuated, passband-
  integrated G flux ratio instead.
  Solutions are interpolated on `q = 0.10--1.00` in steps of 0.01; below the
  lowest supported companion mass the faint solution is the dark root.
- `delta` is a model-preference diagnostic, not a probability. It inherits
  the model errors described in [Model and limitations](model.md).

This is the check of Sect. 6 of Li et al. (XPS paper, in preparation), where
the spectra also include SPHEREx.

[API](api.md) · [README](../README.md)

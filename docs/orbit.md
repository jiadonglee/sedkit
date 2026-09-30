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
for r in rank_roots(sed, roots, parallax_mas=13.913):
    print(r["kind"], r["q"], r["delta"])
```

1. `solve_orbit` returns every solution: `kind` (`dark`, `faint` or
   `luminous`), `q`, `m2`, `beta_G`, and `delta_G`, `delta_Ks`, the
   predicted brightening of the pair over a single primary in magnitudes.
   `solve_amrf(A, m1)` takes `A` directly. No spectrum is needed.
2. Compare `delta_G` or `delta_Ks` with the height of the star above the
   single-star sequence of a colour--magnitude diagram. A metal-poor primary
   lies below the solar-metallicity sequence and can hide the excess.
3. `rank_roots` fits the observed SED at each solution, as a binary with
   `q` fixed at a luminous or faint root and as a single star at a dark root,
   with the primary refitted. It returns the solutions best first with
   `delta`, the fit objective above the best one. Use the two-body parallax.

Command line, for the first check:

```bash
python -m sedkit.orbit --a0 0.6978 --parallax 13.913 --period 339.57 --m1 0.686
python -m sedkit.orbit --amrf 0.14 --m1 0.50
```

## Two observed candidates

[run.py](../examples/orbit_20260930/run.py) applies the three checks to two
Gaia DR3 astrometric substellar candidates with radial-velocity follow-up,
from included XP and 2MASS snapshots.

| Source | Follow-up | Luminous root | Predicted ΔKs | Preferred root | Δ objective |
| --- | --- | ---: | ---: | --- | ---: |
| Gaia DR3 1916454200349735680 | near-equal-mass binary | q = 0.99 | 0.72 mag | luminous | 637 |
| LP 769-9 | substellar companion | q = 0.97 | 0.70 mag | dark | 1118 |

The binary lies 0.68 mag above the single-star sequence in Ks, as the
luminous root predicts; LP 769-9 lies 0.15 mag above it. The SED makes the
same choice from the spectrum alone.

![Two solutions of two Gaia orbits](../examples/orbit_20260930/orbit_roots.png)

Left: `A(q)` for a dark (dotted) and a main-sequence companion (solid), the
observed `A` (dashed) and its two solutions. Right: residuals of the SED fit
at each solution.

## Scope

- The companion is a coeval main-sequence star of the bundled model; white
  dwarfs and unresolved triples are not modelled. A spectrum that rejects
  both main-sequence solutions points to a dark or compact companion.
- Primary mass, age and metallicity are inputs (defaults 5 Gyr, solar);
  extinction is not modelled. The primary mass enters `A` as `M1^(-1/3)`.
- `beta_G` comes from the PARSEC G magnitudes of the two components.
  Solutions are interpolated on `q = 0.10--1.00` in steps of 0.01; below the
  lowest supported companion mass the faint solution is the dark root.
- `delta` is a model-preference diagnostic, not a probability. It inherits
  the model errors described in [Model and limitations](model.md).

This is the check of Sect. 6 of Li et al. (XPS paper, in preparation), where
the spectra also include SPHEREx.

[API](api.md) · [README](../README.md)

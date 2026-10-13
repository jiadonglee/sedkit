# Orbits and joint likelihoods

[中文](../zh/orbits.md) · [GitHub](https://github.com/jiadonglee/spbin)


### Coeval main-sequence branches

```python
from spbin.orbit import solve_orbit, rank_roots

roots = solve_orbit(
    a0_mas=0.6978, parallax_mas=13.913,
    period_day=339.57, m1=0.686,
)
ranked = rank_roots(sed, roots, parallax_mas=13.913, fit_parallax=True)
```

`solve_orbit` returns dark, faint or luminous near-equal-mass branches. `rank_roots` refits the SED along the orbit branches and sorts them. `solve_amrf` accepts the dimensionless astrometric mass-ratio function directly. Ranked `delta` is the branch objective minus the best objective.

### Luminous components of different kinds

`solve_luminous_pair` uses a total-light fraction to solve branches `B>beta` and `B<beta`. `mass_from_rv` derives star-1 mass from companion RV amplitude and inclination. `branch_from_rv` connects RV and photocentre scales. `solve_dark_companion` finds companion mass for a specified companion/primary flux ratio. Periods are days, parallax and angular semimajor axes are mas, RVs are km/s and angles are degrees.

| Interface or field | Light convention |
| --- | --- |
| `StellarModel.evaluate()["beta_g"]`, `orbit.amrf`, `solve_dark_companion(beta=...)` | F2/F1 |
| WD/sdB `fractions["beta_G"]` | WD or sdB F1/(F1+F2) |
| `solve_luminous_pair(beta=...)` | Star-1 F1/(F1+F2) |
| `whitedwarf_light_limit()["beta_G_upper"]` | WD/(primary+WD) |
| `whitedwarf_light_limit()["flux_ratio_G_upper"]` | WD/primary |

For two luminous components, `a0 = a × |B-beta|`, with B=M1/(M1+M2). A WD or sdB light fraction passed to `solve_luminous_pair` identifies that component as star 1.

### External samplers

```python
from spbin import loglike_sed

ll = loglike_sed(
    sed, m1=0.75, q=0.8, age_gyr=5.0, feh=0.0,
    parallax_mas=20.0, extinction=0.0, model=model,
)
```

`loglike_sed` and `loglike_whitedwarf_sed` evaluate shared parameter proposals with model uncertainty and its determinant, without parameter priors or a catalogue-parallax penalty. Unsupported proposals return −∞. Add the appropriate priors and observation likelihoods once in the external joint model. `loglike_sed` omits the fixed −N/2 log(2π) constant, allowing comparisons on the same data mask.

In `loglike_whitedwarf_sed`, `primary_mass` denotes the main-sequence mass and `mass` the WD mass. Use that same WD mass in its cooling radius and the external orbit. See [orblet integration](../orblet.md) for parameter mapping.


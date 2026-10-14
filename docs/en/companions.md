# Dedicated stellar and companion routes

[中文](../zh/companions.md) · [GitHub](https://github.com/jiadonglee/spbin)


### Red giants

```python
from spbin import fit_giant_companion

result = fit_giant_companion(
    sed, teff=(4895, 150), logg=(2.54, 0.3), feh=(-1.22, 0.2),
    threshold=10,
)
print(result["best_m2"], result["detection"], result["m2_excluded"])
```

`teff`, `logg` and `feh` are Gaussian label priors on the APOGEE scale. The default empirical giant template is combined with solar-metallicity, 0.01-Gyr main-sequence companions on a 1.5–15-solar-mass grid. `use_wise=True` is the default here.

`GiantTemplate().evaluate(teff, logg, mh)` returns template flux and error information; `in_support` checks label support. Giant brightness is freely scaled by default. `luminosity="parsec"` adds an evolutionary luminosity prior and mass bounds; `"massfree"` retains mass bounds. `parallax=(mean, sigma)` replaces the catalogue value. `dust_prior` constrains extinction at the trial distance.

`rows` holds each mass fit, with M2=0 denoting the giant alone. `detection` is the giant-only objective minus the profile minimum. A row's `delta` is its objective minus the giant-only value. `m2_excluded` is the first grid mass above the best mass exceeding the requested objective threshold. See [giant validation](../validation-giant.md).

### Hot subdwarfs

```python
from spbin import fit_subdwarf_companion

result = fit_subdwarf_companion(
    sed, companions=("dwarf", "subgiant"),
    extinction=0.0, subdwarf_prior={"logg": (5.6, 0.3)},
)
best = result["hypotheses"][result["preferred"]]
print(best["subdwarf"], best["fractions"]["beta_G"])
```

Default hypotheses are `fgk`, `sdb`, `sdb+dwarf` and `sdb+subgiant`. `companions=()` disables composites while retaining the single-star comparison. Helium tiers are `H`, `mid` and `He`. The subdwarf has a free radius; companions use a dwarf evolutionary model or an empirical subgiant template.

`subdwarf_prior` and `companion_prior` accept dictionaries of `(mean, sigma)` label constraints. `companion_mass`, `companion_age_gyr` and `companion_feh` can fix dwarf parameters. The default `extinction=None` requires an extinction prior; alternatively fix E explicitly. WISE, SPHEREx, blue XP and GALEX are disabled by default.

Results contain `hypotheses`, `preferred`, `delta` above the best objective, `mask`, `n_fit` and fitted data. Component results include `subdwarf`, `companion`, `fractions`, `profile` and `ranges`. `fractions["beta_G"]` is the subdwarf's fraction of total G light.

`SubdwarfModel().predict(teff, logg, radius, tier)` returns 10-pc channel fluxes, blue-channel fluxes and a coarse spectrum. `spbin.subdwarf.blue_xp` and `galex` supply optional diagnostic data. See [subdwarf validation](../validation-subdwarf.md).

### DA white dwarfs

```python
from spbin import WhiteDwarfModel, fit_whitedwarf_companion

wd = WhiteDwarfModel(hydrogen_layer="thick")
prediction = wd.predict(teff=12000, mass=0.6)
physical = wd.physical(teff=12000, mass=0.6)
result = fit_whitedwarf_companion(sed, model=wd, extinction=0.0)
best = result["hypotheses"][result["preferred"]]
```

The default model uses Koester spectra calibrated on real single DAs and thick-H C/O cooling relations. `hydrogen_layer="thin"` supports a thin-H comparison; `calibration=None` uses the raw spectral model.

`predict` returns a dictionary with 10-pc `flux`, blue/coarse spectra and physical parameters. `physical` returns `teff`, `mass`, `radius`, `logg`, `cooling_age_gyr` and `luminosity`. `teff_at_age(mass, age_gyr)` inverts supported cooling ages. Free-radius prediction requires both `logg` and `radius`.

Default fits compare `dwarf`, `wd` and `wd+dwarf`. `hypotheses` selects these and optional `ms+ms`. `companions=False` compares the two single-star hypotheses. `free_radius=True` fits WD temperature, gravity and radius independently, derives mass from gR²/G and leaves cooling age unavailable.

`whitedwarf_prior` supports temperature, mass, gravity and radius constraints. `companion_prior` also supports age and metallicity constraints. The dwarf defaults to 5 Gyr and solar metallicity; None releases either parameter.

Results include `hypotheses`, `preferred`, `delta` above the best objective, `detection_statistic`, the shared data mask and hydrogen-layer setting. `detection_statistic` compares the composite against the best requested single-star hypothesis. `fractions["beta_G"]` is the WD fraction of total G light. See [WD conventions](../whitedwarf.md).

### Conditional WD light envelopes

```python
from spbin import whitedwarf_light_limit

limit = whitedwarf_light_limit(
    sed, masses=[0.6, 0.8, 1.0],
    temperatures=[8000, 12000, 20000], delta=9.0,
)
print(limit["beta_G_upper"], limit["intervals"])
```

Scan the supplied masses and temperatures, or supply `cooling_ages_gyr`. The primary and shared nuisance parameters are refitted at each point. Disconnected allowed intervals are retained. Fields include `beta_G_upper`, `flux_ratio_G_upper`, `profile`, `intervals`, `single_primary` and optional `luminous_companion`. An empty allowed WD set returns a NaN upper envelope.

`galex_upper_limits` accepts measured total-system UV flux caps in `SED` units. `delta=9` defines the operational objective envelope; the result has `confidence_level=None`.


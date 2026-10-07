# API

## Observations

`download(source_id, cache_dir="data", refresh=False)` returns an `SED`.
Alternatively use `download(ra=..., dec=..., radius_arcsec=2)`.

`SED(flux, error, mask, parallax_mas, parallax_error_mas=0, source_id="")`
accepts a user's own observations. Arrays have 168 channels:
XP61, J/H/Ks/W1/W2, SPHEREx102. Flux and error are in
`1e-18 W m^-2 nm^-1` at the source's actual distance. Missing channels
are NaN and masked. Calibrate XP on the model's 392--992 nm grid,
with 10 nm spacing; do not substitute another 61-channel grid.

`sed.save(path)` and `SED.load(path)` preserve measurements and metadata.
`sed.with_spherex(wavelength_um, flux, error, valid=None)` returns a new
SED with an already extracted SPHEREx spectrum. Wavelengths must match
`StellarModel().wavelength_um[66:]`. It does not extract images or resample.

`download_spherex(sed, cache_dir="data", refresh=False, radius_arcmin=0.8)`
returns a copy with a QR2 aperture spectrum attached. It requires the optional
`spherex` extra. `load_spherex(sed, path, method=None)` imports channel-binned
XphereX aperture/PSF CSVs in Jy. See [SPHEREx acquisition](spherex.md) for
quality selection, native units, cache products and supported channels.

`query_gaia(query, cache_dir=..., tap="esa")` returns a catalogue as an
Astropy Table from an asynchronous ADQL job. `download_gaia(source_ids,
products=["XP_CONTINUOUS", "RVS"], cache_dir=...)` returns records pointing
to native FITS ZIP batches, including delivered and unavailable IDs. Both
support resuming the same request. See [Gaia batch downloads](gaia.md) for
all arguments, examples and cache semantics.

## Prediction

```python
from sedkit import StellarModel
model = StellarModel()
pair = model.evaluate(m1=0.75, q=0.8, age_gyr=5, feh=0)
```

Masses are in solar masses. `q=0` gives one star, `0<q<=1` a binary.
Unsupported components return `None`. The dictionary contains
`flux_10pc`, `components`, `masses`, `labels`, `teff`, `logg`, `radius`,
`hot_weight`, `M_G`, `beta_g`, `a_phot_over_a1`, `age_gyr`, `feh` and
`route`. Labels are `[Teff, M_Ks, G-Ks, [M/H]]`; `radius` is in Rsun.

`StellarModel(hot=True)` enables the [hot-star route](model.md#hot-star-route):
components above 7000 K use the hot table, ages start at 10**6.6 yr and
primary masses reach 20 solar masses. Channels beyond J/H/Ks are NaN and
`model.support` marks the predicted channels. `model.predict_hot(teff, logg,
radius)` evaluates the table directly.

`model.predict_labels(labels)` provides direct four-coordinate prediction.
`model.in_domain(labels)` checks training coverage. Neither operation
establishes real-data accuracy.

## Composable likelihood

```python
from sedkit import loglike_sed
ll_sed = loglike_sed(sed, m1=0.75, q=0.8, parallax_mas=20,
                    age_gyr=5, feh=0, model=model)
```

`loglike_sed` evaluates one shared parameter proposal without priors or a
catalogue parallax constraint. It includes model covariance and its
determinant, preserving absolute observed fluxes and errors. Reuse `model`
inside a sampler. W1/W2 are excluded unless `use_wise=True`.
`extinction=E` attenuates the model in native ZGR23 units without a dust prior.
Unsupported stellar components or nonpositive parallax return `-inf`.
The fixed `-N/2 log(2pi)` constant is omitted; compare on the same data mask.
See [orblet](orblet.md) for composing SED, SB2 RV and astrometry likelihoods.

## Fitting

`fit(sed, kind="both", model=None, age_gyr=5, feh=0, q=None,
use_wise=False, fit_parallax=False, extinction=0, dust_prior=None,
age_prior=None, logg_prior=None, teff_prior=None)` uses multi-start Nelder--Mead with a
fixed initial step in each fitted parameter.
Reuse a `StellarModel` across sources through `model=`.

- Age and metallicity are fixed unless set to `None`.
- `age_prior=(mean, sigma)` constrains a free age with a Gaussian in
  log10(age/yr), for example from a host cluster. `logg_prior=(mean, sigma)`
  constrains the primary's PARSEC log g, for example from Balmer-line
  spectroscopy, and `teff_prior=(mean, sigma)` the primary's Teff in K, for
  example from its spectral type. They enter the single and binary
  objectives alike.
- Free binary q covers 0.1--1, further restricted by component support.
- `q=` fixes q for the binary hypothesis.
- Parallax is fixed at the catalogue value by default. Set
  `fit_parallax=True` to fit within three catalogue standard deviations,
  with one Gaussian constraint. Zero/absent uncertainty keeps it fixed.
- W1/W2 are held out unless `use_wise=True`.
- The fitting mask keeps only `model.support` channels; with
  `StellarModel(hot=True)` that is XP and J/H/Ks. Hot-star fits need a
  young `age_gyr=` or `age_gyr=None`.
- `extinction=None` fits nonnegative ZGR23 E with an `EdenhoferPrior`;
  a number fixes E. See [extinction](extinction.md) for map setup and assumptions.

`kind="single"` or `"binary"` returns one result dictionary.
`kind="both"` returns `single`, `binary`, `delta` and `source_id`.
Each result includes masses, q, age, metallicity, parallax, model flux,
components, light ratio, `chi2`, `m2lnl`, `objective`, `n_fit`, `mask`,
`converged` and `at_bounds`, plus `extinction_e`, `dust_prior_mean`,
`dust_prior_sigma` and `dust_prior_penalty`. `objective` includes parallax and
dust constraints and `label_prior_penalty`, the age, log g and Teff prior terms;
`logg` is the primary's PARSEC log g. `m2lnl` includes the model covariance
determinant without priors.

## Giants with a hot companion

```python
from sedkit import download, fit_giant_companion
sed = download("465986123215151616", cache_dir="data")
result = fit_giant_companion(sed, teff=(4895, 150), logg=(2.54, 0.3), feh=(-1.22, 0.2),
                             threshold=10)
print(result["best_m2"], result["detection"], result["m2_excluded"])
```

`fit_giant_companion(sed, teff, logg, feh, *, m2_grid=M2_GRID,
companion_age_gyr=0.01, extinction_prior=None, tilt_sigma=0.15,
use_wise=True, template=None, model=None, threshold=None)` fits an
empirical giant template plus a main-sequence companion of each grid mass
(1.5--15 solar masses by default) at the catalogue parallax; see the
[giant route](model.md#giant-route). `teff`, `logg` and `feh` are
`(mean, sigma)` Gaussian priors on the APOGEE scale; for LAMOST LASP labels
use widths of about 150 K, 0.3 and 0.2. `extinction_prior=(mean, sigma)`
constrains E. Channels are XP and J/H/Ks, with W1/W2 unless
`use_wise=False`. Reuse `GiantTemplate()` and `StellarModel(hot=True)`
across sources through `template=` and `model=`.

The giant's scale is free by default, so its luminosity is not tied to the
parallax. Three keywords add that constraint:

- `luminosity="parsec"` fits the parallax within 3 sigma (penalty z^2) and
  ties the giant's luminosity to it. The PARSEC M_Ks density at the giant's
  labels enters as a prior, and the implied mass (from M_Ks, BC_Ks, log g and
  Teff) is held in 0.2--10 solar masses.
- `luminosity="massfree"` keeps only the mass bounds, which allows a
  stripped giant.
- `dust_prior=EdenhoferPrior(...)` adds the map E at the trial distance.
  Within 1.25 kpc it is Gaussian with width sqrt(sigma^2 + 0.04^2); beyond,
  E is held above the value at 1.2 kpc minus 0.04. It needs
  `sed.metadata["ra"]` and `["dec"]` and replaces `extinction_prior`.

`parallax=(mean, sigma)` replaces the SED's parallax and error for both
stars, for example by a zero-point-corrected or non-single-star parallax;
`download` stores the uncorrected Gaia DR3 value.

```python
from sedkit import EdenhoferPrior
result = fit_giant_companion(sed, teff=(4895, 150), logg=(2.54, 0.3), feh=(-1.22, 0.2),
                             luminosity="parsec", parallax=(0.374, 0.020),
                             dust_prior=EdenhoferPrior(), threshold=10)
```

`result["rows"]` holds one row per mass, M2 = 0 being the giant alone:
`objective`, its `delta` from the giant alone, its -2 ln L part `minus2lnL`,
`chi2`, the giant's labels, `extinction_e`, `tilt` and `scale`, and the
companion's Teff, radius and share of the observed 0.40--0.45 micron flux.
With `luminosity` or `dust_prior`, rows also hold the fitted `z` and
`parallax_mas`, the giant's `mks`, `luminosity`, `radius` and `mass`, the map
`e_map` and `distance_pc`, and the penalties `label_penalty`,
`luminosity_penalty` (`parsec_penalty` plus `mass_penalty` for "parsec"),
and `dust_penalty`. `detection` is the objective
of the giant alone minus the profile minimum and `best_m2` the mass at the
minimum. With `threshold=t`, `m2_excluded` is the lowest grid mass above
`best_m2` whose objective exceeds the minimum by more than t. The
thresholds calibrated on control giants are 10 for the default fit and for
`luminosity="parsec"` with or without the dust prior, and 11 for
`luminosity="massfree"` with the dust prior
([giant validation](validation-giant.md)). `dust_prior=` without
`luminosity=` has no calibrated threshold.

## Plotting

`plot(sed, result=None, path=None, *, axes=None, components=True, title=None)`
returns a Matplotlib Figure in the blue/coral [sedkit palette](appearance.md).
Single-star fits are orange and binary fits blue. Save it with `path=` or `fig.savefig(...)`.
The paper-style upper panel shows linear lambda F_lambda in physical units;
the lower panel shows ln(F/F_single), or ln(F/F_binary) for a binary-only
result. Measurement errors are unchanged; log-panel error bars use the
first-order error/flux approximation. Nonpositive measurements appear only
in the upper panel. Open points identify available but excluded channels.

Use `components=False` to hide component spectra, `title=` for a custom
caption, or `axes=(sed_axis, ratio_axis)` for a multi-source layout. With
external axes, supply the shared legend and save their owning figure.
The plotting style is scoped to the call; spectra are not normalized.

## Photocentre orbits

`sedkit.orbit.solve_orbit(a0_mas, parallax_mas, period_day, m1, age_gyr=5,
feh=0, model=None)` returns the solutions of a Gaia photocentre orbit for a
coeval main-sequence companion; `solve_amrf(a_obs, m1, ...)` takes the
astrometric mass-ratio function directly. Each solution has `kind`, `q`,
`m2`, `beta_G`, `delta_G` and `delta_Ks`.
`rank_roots(sed, roots, parallax_mas=None, model=None, **fit_kwargs)` fits
the SED at each solution and sorts them by the fit objective, with `delta`
above the best. See [Photocentre orbits](orbit.md).

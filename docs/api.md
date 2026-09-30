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
`flux_10pc`, `components`, `masses`, `labels`, `teff`, `M_G`, `beta_g`,
`a_phot_over_a1`, `age_gyr`, `feh` and `route`.
Labels are `[Teff, M_Ks, G-Ks, [M/H]]`.

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
Unsupported stellar components or nonpositive parallax return `-inf`.
The fixed `-N/2 log(2pi)` constant is omitted; compare on the same data mask.
See the [orblet example](orblet.md) for a composed SED + SB2 RV + astrometry fit.

## Fitting

`fit(sed, kind="both", model=None, age_gyr=5, feh=0, q=None,
use_wise=False, fit_parallax=True)` uses multi-start Nelder--Mead.
Reuse a `StellarModel` across sources through `model=`.

- Age and metallicity are fixed unless set to `None`.
- Free binary q covers 0.1--1, further restricted by component support.
- `q=` fixes q for the binary hypothesis.
- Parallax is fitted within three catalogue standard deviations, with one
  Gaussian constraint. Zero/absent uncertainty fixes it. Set
  `fit_parallax=False` for a fixed-distance experiment.
- W1/W2 are held out unless `use_wise=True`.

`kind="single"` or `"binary"` returns one result dictionary.
`kind="both"` returns `single`, `binary`, `delta` and `source_id`.
Each result includes masses, q, age, metallicity, parallax, model flux,
components, light ratio, `chi2`, `m2lnl`, `objective`, `n_fit`, `mask`,
`converged` and `at_bounds`. `objective` includes the parallax constraint;
`m2lnl` includes the model covariance determinant but not that constraint.

## Plotting

`plot(sed, result=None, path=None, *, axes=None, components=True, title=None)`
returns a Matplotlib Figure. Save it with `path=` or `fig.savefig(...)`.
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


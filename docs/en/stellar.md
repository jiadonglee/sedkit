# Stellar prediction and fitting

[中文](../zh/stellar.md) · [GitHub](https://github.com/jiadonglee/spbin)


### `StellarModel`

```python
from spbin import StellarModel

model = StellarModel()
pair = model.evaluate(m1=0.75, q=0.8, age_gyr=5.0, feh=0.0)
labels = model.labels([0.75, 0.60], age_gyr=5.0, feh=0.0)
inside = model.in_domain(labels)
```

`evaluate` derives stellar parameters from mass, age and metallicity, then sums absolute component fluxes. `q=0` describes one star; `0<q<=1` describes a coeval pair. An unsupported component returns None.

Main fields are `flux_10pc`, `components`, `masses`, `labels`, `teff`, `logg`, `radius`, `M_G`, `hot_weight`, `beta_g`, `a_phot_over_a1`, `age_gyr`, `feh` and `route`. Each label row is `[Teff, M_Ks, G-Ks, [M/H]]`.

`predict_labels(labels)` predicts directly from these coordinates; `predict_hot(teff, logg, radius)` evaluates the hot table. `wavelength_um`, `channels` and `support` provide wavelengths, channel names and the supported-channel mask.

The observed-scale model is:

```text
F_observed = F_10pc × (parallax_mas / 100)^2 × exp(-E × k_lambda)
```

### `fit` parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `kind` | `"both"` | `single`, `binary` or both |
| `model` | `None` | Creates `StellarModel()`; reuse an instance across sources |
| `age_gyr` | `5.0` | Fixed age; None fits age |
| `feh` | `0.0` | Fixed [M/H]; None fits metallicity |
| `q` | `None` | Free ratio, fixed numeric ratio or callable constraint |
| `use_wise` | `False` | Include W1/W2 |
| `fit_parallax` | `False` | Fit within catalogue ±3σ with one Gaussian constraint |
| `extinction` | `0.0` | Fixed E; None fits E≥0 |
| `dust_prior` | `None` | 3D extinction prior; free E creates EdenhoferPrior if absent |
| `age_prior` | `None` | `(mean, sigma)` on log10(age/yr), requires free age |
| `logg_prior` | `None` | `(mean, sigma)` on the primary's log g |
| `teff_prior` | `None` | `(mean, sigma)` on the primary's Teff in K |

The free binary ratio spans 0.1–1 subject to component support. A callable uses `q(m1, age_gyr, feh, parallax_mas)`. Parallax remains fixed if its uncertainty is absent or nonpositive. Fits use multistart Nelder–Mead local optimization.

### `fit` return values

`kind="both"` returns `single`, `binary`, `delta` and `source_id`. The other kinds return one fit dictionary directly.

| Field | Meaning |
| --- | --- |
| `m1`, `m2`, `q` | Component masses and mass ratio |
| `age_gyr`, `feh`, `logg` | Age, metallicity and primary gravity |
| `parallax_mas`, `extinction_e` | Parallax and extinction used by the fit |
| `flux`, `components` | Total/component fluxes at source distance after attenuation |
| `objective` | −2 ln L plus parallax, extinction and label-prior penalties |
| `m2lnl` | SED −2 ln L with model-covariance determinant and no priors |
| `chi2` | Residual quadratic term with measurement and model errors |
| `n_fit`, `mask` | Fitted-channel count and 168-channel mask |
| `converged` | Whether optimizer convergence criteria were met |
| `at_bounds` | Names of parameters close to search boundaries |
| `dust_prior_mean`, `dust_prior_sigma`, `dust_prior_penalty` | Dust-prior moments and penalty |
| `label_prior_penalty` | Combined age, gravity and temperature penalty |
| `beta_g`, `M_G`, `a_phot_over_a1`, `route`, `kind` | Light ratio, magnitudes, photocentre factor, route and hypothesis |

### HR-diagram routing

```python
from spbin import locate, fit_star

location = locate(sed)
print(location["region"], location["reason"])
result = fit_star(sed, age_gyr=None, feh=0.0)
print(result["preferred"])
```

`locate` requires positive parallax, Ks, and metadata keys `phot_g_mean_mag`, `phot_bp_mean_mag`, `phot_rp_mean_mag`. Regions are `ms`, `hot_ms`, `giant`, `sdb`, `wd` and `unsupported`. It also returns dereddened colours, M_G, hypotheses and the selection reason.

`fit_star` returns `location`, `region`, the route-specific `result`, `objectives`, `delta` and `preferred`. It prefers the simplest hypothesis within `tie=1.0` of the lowest objective. `options` supplies route keywords with precedence over common settings. Giants require label priors:

```python
result = fit_star(
    sed, region="giant",
    labels={"teff": (4895, 150), "logg": (2.54, 0.3), "feh": (-1.22, 0.2)},
)
```

Unsupported regions or missing giant labels return `result=None`, with the reason in `location`.


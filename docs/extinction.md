# Extinction fitting with a 3D dust prior

```python
from sedkit import EdenhoferPrior, download, fit

sed = download("858860697467058688", cache_dir="data")
prior = EdenhoferPrior()  # load once, reuse across sources
result = fit(sed, extinction=None, dust_prior=prior)
print(result["binary"]["extinction_e"])
```

Install `pip install -e ".[dust]"`. `EdenhoferPrior()` uses
`dustmaps.edenhofer2023.Edenhofer2023Query(integrated=True, load_samples=True)`.
The map files must already be in the configured dustmaps data directory.
The constructor and fitting do not download files. The posterior-sample
map is about 19 GB: prepare and use it on a machine with enough memory.
Prepare the map once on the server:

```python
from dustmaps.edenhofer2023 import fetch
fetch(fetch_samples=True)
```

For an explicit sample FITS path, pass
`EdenhoferPrior(Edenhofer2023Query(map_fname="samples_healpix.fits",
integrated=True, load_samples=True))`; import the query from
`dustmaps.edenhofer2023`. Bulk downloads and full-map loading belong on the server.

The fitted parameter is nonnegative **ZGR23 E**, the native map unit,
not E(B-V) or A_V. Foreground attenuation is `exp(-E * k_lambda)`, using
the published Zhang, Green & Rix (2023) optical-depth curve. Both binary
components share E. The same attenuation multiplies the model flux and
low-rank error columns; model variance is multiplied by its square.
Observed fluxes, measurement errors and the shared fitting mask are unchanged.

Parallax is fixed at the catalogue value by default. `fit_parallax=True`
fits it with its catalogue Gaussian constraint and updates the dust prior
at each trial distance.

At distance `1000 / parallax_mas` pc, the prior uses the
map mean and standard deviation at `sed.metadata["ra"]`, `["dec"]`
(ICRS degrees, already included by `download`). It approximates the
map posterior by a Gaussian truncated at E=0. Its width and truncation
normalization are retained as distance varies. Both hypotheses use the
same prior prescription. Unsupported trial distances have zero prior support;
an unsupported catalogue distance raises an error. No extrapolation or
zero-extinction fallback is applied to missing map values.

## Mean-only maps and fixed extinction

The mean/std file contains **density** uncertainties, not uncertainties
of integrated extinction. To use a mean-only integrated map, explicitly
choose an E prior width:

```python
from dustmaps.edenhofer2023 import Edenhofer2023Query
query = Edenhofer2023Query(integrated=True)
prior = EdenhoferPrior(query, sigma=0.03)  # chosen E width, not map uncertainty
result = fit(sed, extinction=None, dust_prior=prior)
```

`fit(sed, extinction=0.1)` instead fixes E without loading a dust map.
The default `extinction=0` preserves the zero-extinction experiment.
`loglike_sed(..., extinction=E)` includes attenuation but no dust prior.

Results include `extinction_e`, `dust_prior_mean`, `dust_prior_sigma` and
`dust_prior_penalty`. `objective` includes dust and parallax constraints;
`m2lnl` remains the SED likelihood alone. `delta` is the difference of
penalized best fits, not a marginalized Bayes factor.

## Real-data example

The executed [extinction and parallax notebook](../examples/05_extinction_parallax.ipynb)
compares fixed E/distance, fitted E at fixed distance, and jointly fitted E/parallax.
It shows the absolute SEDs, fitted parameters and input priors.

Install `pip install -e ".[download,dust]"` and run
`python examples/04_extinction_prior.py --cache-dir data --output-dir data/extinction_example`.
Use `--map /path/to/samples_healpix.fits` for an explicit map path.
The [example script](../examples/04_extinction_prior.py) fits the Gaia DR3
SB2 `858860697467058688` with E=0 and with the map prior. It saves unchanged
observations, fitted fluxes/components/masks, a JSON parameter summary,
and PNG/PDF SED and prior plots. See the [real-map check](extinction-validation.md).

## Limitations

The curve shape is fixed; R_V is not fitted. Broadband attenuation uses
the catalogue-equivalent coefficients at the nominal J/H/Ks/W1/W2 wavelengths,
rather than SED-dependent passband integration. SPHEREx uses log interpolation
between the curve anchors and extends the W1--W2 slope from 4.60 to 4.98 µm.
That infrared approximation and the dust-prior width need real-data validation.
The map itself uses Gaia XP estimates; this prior is not independent of all
SED data. These are exploratory constrained fits, not calibrated posteriors.

Sources: [Edenhofer map](https://doi.org/10.5281/zenodo.8187943),
[dustmaps interface](https://dustmaps.readthedocs.io/en/latest/modules.html#module-dustmaps.edenhofer2023),
[ZGR23 extinction curve](https://doi.org/10.5281/zenodo.7811871).

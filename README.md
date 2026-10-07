<p align="center">
  <img src="docs/assets/sedkit-logo.png" width="280" alt="sedkit logo">
</p>
<p align="center">Download and fit stellar spectral energy distributions.</p>

sedkit combines public Gaia DR3 XP spectra with Gaia-linked 2MASS and
AllWISE photometry. A compact empirical stellar model predicts absolute
fluxes for a single star or a coeval binary, and tells whether a Gaia
astrometric orbit comes from a faint companion or a hidden near-equal-mass
twin. The package runs on NumPy and
SciPy, with no J-CAPS installation, JAX, GPU or separate model download.

## Capabilities and scope

### Data

- Gaia DR3 XP spectra on 61 channels at 392--992 nm, calibrated with
  GaiaXPy from the public continuous spectra.
- 2MASS J/H/Ks and AllWISE W1/W2 from Gaia's best-neighbour tables. A band
  enters the fit only for a unique point-source match with A-quality
  photometry. W1/W2 are held out of `fit` by default.
- Optional SPHEREx QR2 spectra ([SPHEREx downloads](docs/spherex.md)).
- Gaia G/BP/RP are kept as metadata, not fit channels. No ultraviolet data
  and no XP below 392 nm enter a fit, so the Balmer jump is not used.
- Models predict absolute fluxes at the parallax, with no free
  normalisation. The parallax is fixed at the catalogue value or fitted
  within three catalogue standard deviations under its Gaussian constraint
  (`fit_parallax=True` in `fit`, `luminosity=` in `fit_giant_companion`).

### Models

| Entry point | Stars | Support | Hypotheses |
| --- | --- | --- | --- |
| `fit` with `StellarModel()` | FGKM dwarfs: empirical J-CAPS network on PARSEC tracks | Teff 2800--7498 K, age 0.5--10 Gyr, [M/H] -1.0 to +0.5; primary mass to 1.5--1.8 solar masses at 0.5--2 Gyr and 1.38 at 3 Gyr (solar [M/H]) | single star, coeval binary |
| `fit` with `StellarModel(hot=True)` | adds A and B dwarfs: CK04/TLUSTY table with an empirical XP correction | 7000--30000 K, log g 3--4.75, [M/H] -0.3 to +0.3, age from 4 Myr, mass to 20 solar masses; XP and J/H/Ks only | single star, coeval binary |
| `fit_giant_companion` | red giant (empirical APOGEE template) plus a hot main-sequence companion | giant Teff 3600--6800 K, log g 0--3.8, [M/H] -2.6 to +0.6; companion 1.5--15 solar masses at 10 Myr and solar [M/H] | giant alone, giant plus companion profiled in mass |
| `orbit.solve_orbit`, `rank_roots` | Gaia photocentre orbits | as `StellarModel()` | faint companion, luminous twin |

Fits stay inside these ranges; the models do not extrapolate.
See [model and limitations](docs/model.md).

### What the fits measure

Single stars:

- **FGK dwarfs.** Hyades, Praesepe and Coma Ber members at 6000--7500 K
  fit with chi2/N of 0.7--0.9 at free age, with masses 0.02--0.06 solar
  masses below the isochrone mass. Free ages come out older than the
  literature cluster ages (1.0--2.0 against 0.6--0.7 Gyr)
  ([warm-star validation](docs/validation-warm.md)).
- **A and B dwarfs.** 44 holdout anchors at 7.5--30 kK fit to 1.2--1.4 per
  cent in XP shape, with Teff 0.2 kK below to 0.05 kK above the
  spectroscopic values. XP and J/H/Ks do not constrain extinction for hot
  stars, so E must be fixed or given a dust prior. The 1-sigma Teff width
  is then 0.6--1.0 kK at 15 kK and 1.6--2.8 kK at 25 kK
  ([hot-star validation](docs/validation-hot.md)).

Unresolved coeval binaries. `result["delta"]` is the single-star minus
binary objective, a model-preference diagnostic rather than a probability:

- **Known age.** In mocks of 1.2--1.55 solar-mass primaries with age and
  [M/H] fixed at the truth, q >= 0.5 companions give Delta 25--400 and q
  within 0.03. A Gaia DR3 SB2 gives q = 0.840 against the RV ratio 0.868.
  For 20 warm SB2s, fits at the RV mass ratio lie within 4 of the best
  objective.
- **Free age, warm primaries (1.2--1.6 solar masses).** An older single
  star reproduces the companion's light: Delta stays below about 15 and q
  is unconstrained. Detection needs an independent age checked against the
  same cluster's main sequence, or an asteroseismic log g (0.02 dex
  detects q >= 0.7). For a 0.8 solar-mass primary with a q = 0.7
  companion, no single age comes within 449 of the binary objective.
- **F-type cluster members at the cluster age.** With the parallax moved to
  the cluster's RV-single sequence, Delta > 25 is 2.7 times more likely for
  a star with a luminous companion than for an RV-single star. It
  separates the two classes no better than the height above the cluster
  sequence.
- **Hot primaries.** Delta does not identify an individual hot binary. For
  near-ZAMS B and A cluster members (7--13 kK) fitted the same way, with a
  20--30 per cent prior binary fraction, Delta <= 25 makes a q >~ 0.65
  companion unlikely (7--11 per cent). Delta > 25 marks a candidate
  (50--65 per cent) for RV or imaging follow-up.

Giants with a hot companion:

- **Detection threshold.** `fit_giant_companion` detects a companion that
  supplies more than about 20 per cent of the 0.40--0.45 micron light: 87
  per cent of injections at 20--30 per cent, all above 30 per cent, none
  below 10 per cent. The threshold of 10 lies above every one of 238
  reddened control giants (maximum 8.8).
- **Recovery by mass.** Injected 2 and 3 solar-mass companions are
  detected in 92.9 and 99.2 per cent of cases. On the controls, the fit
  excludes companions from 2 solar masses upward (median).
- **Luminosity and dust.** By default the giant's luminosity is a free
  scale at the catalogue parallax. `luminosity="parsec"` fits the parallax
  and ties the luminosity to it through a PARSEC M_Ks prior and bounds on
  the implied mass; `dust_prior=` adds the Edenhofer map E. With both, and
  zero-point-corrected parallaxes passed through `parallax=`, the threshold
  stays at 10 and injected 2 solar-mass companions are detected in 97.1 per
  cent of cases.
- **Assumptions.** The fit needs Teff, log g and [M/H] priors on the
  APOGEE scale. Giant and companion share only extinction and, with
  `luminosity=`, the parallax ([giant validation](docs/validation-giant.md)).

Other measurements:

- **Photocentre orbits.** `solve_orbit` returns the faint-companion and
  luminous-twin solutions of a Gaia astrometric orbit, and `rank_roots`
  compares their SEDs ([photocentre orbits](docs/orbit.md)).
- **Extinction.** E defaults to zero. `extinction=None` fits nonnegative
  ZGR23 E with an [Edenhofer 3D dust prior](docs/extinction.md) that follows
  the trial parallax. Distances outside the map have no prior support.
  A numeric `extinction=` fixes E.

### Not covered

- Giants and subgiants in `fit`: the network is trained on dwarfs, the hot
  table stops at log g 3, and stars near the turn-off are outside the
  tables. The giant route models only a hot main-sequence companion.
- White dwarfs, brown dwarfs and ultracool atmospheres, triples, blends and
  variable stars.
- Stars above 30 kK (O stars are untested), hot stars with |[M/H]| > 0.3,
  supergiants, Be and emission-line stars, chemically peculiar stars and
  fast rotators.
- Binaries of different ages, except the giant route.
- Posterior uncertainties, calibrated binary probabilities and population
  inference: fits return constrained best fits and objective differences.
- Correlations between XP channels: GaiaXPy inter-channel covariance is
  omitted.
- SPHEREx above about 6400 K: these predictions are extrapolated.

## Install

```bash
pip install "sedkit[download] @ git+https://github.com/jiadonglee/sedkit.git"
```

For a local checkout, use `pip install -e ".[download]"`. Plain
`pip install .` supports offline fitting and plotting. Notebooks additionally
need Jupyter: `pip install -e ".[download,notebook]"`. For optional SPHEREx
acquisition, add the `spherex` extra; see [SPHEREx downloads](docs/spherex.md).

## One source

```python
from sedkit import download, fit, plot

sed = download("858860697467058688", cache_dir="data")
result = fit(sed, age_gyr=5.0, feh=0.0)
print(result["binary"]["m1"], result["binary"]["q"])
fig = plot(sed, result, path="sed.png")
```

This example fixes age and metallicity. Use `age_gyr=None, feh=None` to fit
them. Parallax is fixed by default; `fit_parallax=True` fits it with its
catalogue Gaussian constraint. Use `kind="single"`, `kind="binary"`, or the default `kind="both"`;
`q=0.7` fixes the binary mass ratio. `result["delta"]` is the single minus
binary objective, including the parallax constraint when fitted. It is a model
preference diagnostic, not a binary probability.

Coordinates are also accepted: `download(ra=..., dec=...)`, in ICRS degrees
at Gaia's reference epoch. Coordinate lookup requires exactly one Gaia
match within 2 arcsec. Source IDs avoid coordinate-epoch ambiguity.

## Fit extinction and parallax together

Install `pip install -e ".[download,dust,notebook]"` for this tutorial.
Prepare the Edenhofer posterior-sample map once on a server with enough
memory; the full map is about 19 GB and is not downloaded by the fit.

```python
from sedkit import EdenhoferPrior

prior = EdenhoferPrior()  # load the existing map once and reuse it
result = fit(sed, extinction=None, dust_prior=prior, fit_parallax=True)
print(result["binary"]["extinction_e"], result["binary"]["parallax_mas"])
```

[Notebook 05](examples/05_extinction_parallax.ipynb) compares fixed E/parallax,
fitted E at fixed parallax, and jointly fitted E/parallax on the same real
SB2. It includes fitted parameters, absolute SEDs and input-prior plots.
The results are constrained best fits, not posterior samples.

![Single and binary SED fits](docs/assets/sed-example.png)

Gaia DR3 SB2 `858860697467058688`: the single-star fit is orange and the
coeval-binary fit blue, with extinction and parallax constraints.
[Figure PDF](docs/assets/sed-example.pdf) · [Plotting script](scripts/plot_readme.py).

![Extinction and parallax priors](docs/assets/extinction-parallax-priors.png)

Input priors and best-fit locations; these curves are not posteriors.
[Figure PDF](docs/assets/extinction-parallax-priors.pdf) · [Plotting script](scripts/plot_readme.py).

## Gaia batch downloads

`query_gaia` queries catalogues asynchronously; `download_gaia` downloads
native FITS products in batches and resumes completed work. See
[Gaia batch downloads](docs/gaia.md) for runnable examples, 300 pc selection,
epoch photometry and cache behaviour.

## Is a Gaia substellar candidate a hidden twin?

A Gaia photocentre orbit constrains mass ratio and light ratio together.
`sedkit.orbit.solve_orbit` finds the faint and luminous solutions;
`rank_roots` compares their absolute SEDs. XP and 2MASS prefer the luminous
solution for a followed-up binary and the dark solution for LP 769-9.
See [the method and worked example](docs/orbit.md) and
[notebook 02](examples/02_gaia_orbit_twin.ipynb).

## Examples

Four executed notebooks, run from `examples/`:

- [01: fit an observed SED](examples/01_sed_fit.ipynb): download a Gaia DR3 SB2,
  compare single and coeval-binary fits, check q against the RV ratio, and scale up.
- [02: Gaia orbit, hidden twin](examples/02_gaia_orbit_twin.ipynb): both solutions of
  two astrometric substellar candidates and the one their SEDs prefer.
- [03: warm primaries](examples/03_warm_binaries.ipynb): binary mocks with known
  and free age, real SB2 checks, and the warm-star age/companion degeneracy.
- [05: extinction and parallax priors](examples/05_extinction_parallax.ipynb):
  fit E at fixed distance or jointly with catalogue-constrained parallax;
  compare the SEDs, parameters and input priors on a real SB2.

[04: extinction prior](examples/04_extinction_prior.py) is a runnable script
comparing zero-extinction and Edenhofer-prior fits of a real SB2.

## Documentation

- [API](docs/api.md): observations, predictions and fitting options.
- [Model and limitations](docs/model.md): physical assumptions and support.
- [Extinction](docs/extinction.md): fitting E with an Edenhofer 3D dust prior.
- [Data and provenance](docs/data.md): units, quality masks and cache products.
- [Gaia batch downloads](docs/gaia.md): catalogue queries and native products.
- [SPHEREx downloads](docs/spherex.md): append QR2 spectra or import XphereX results.
- [Photocentre orbits](docs/orbit.md): faint companion or hidden twin.
- [orblet interface](docs/orblet.md): composing SED, RV and astrometry likelihoods.
- [Validation](docs/validation.md): installation and example checks.
- [Warm-star validation](docs/validation-warm.md): warm network, cluster members, binary mocks and log g priors.
- [Hot-star validation](docs/validation-hot.md): calibration, holdout and CALSPEC checks, binary injections and cluster tests.
- [Giant validation](docs/validation-giant.md): giant template, controls and injected companions.
- [Visual identity](docs/appearance.md): logo, plotting palette and reproducible homepage figures.

## Development

```bash
pip install -e ".[test]"
pytest -q
```

## Credits

The bundled empirical model is extracted from
[J-CAPS](https://github.com/jiadonglee/J-Caps), using PARSEC stellar tracks.
Public XP spectra are calibrated with
[GaiaXPy](https://gaia-dpci.github.io/GaiaXPy-website/).
SPHEREx aperture extraction uses [XphereX](https://github.com/jiadonglee/XphereX),
[TallTable](https://github.com/cmhainje/talltable) and
[SPExPI](https://github.com/fkiwy/spexpi).
Catalogue data are retrieved through
[astroquery](https://astroquery.readthedocs.io/en/latest/gaia/gaia.html).
Please acknowledge Gaia, 2MASS, WISE, PARSEC and J-CAPS when using these
data and models in research; see [provenance](docs/data.md).

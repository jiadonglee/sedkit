<p align="center">
  <img src="docs/assets/sedkit-logo.png" width="280" alt="sedkit logo">
</p>
<p align="center">Download and fit stellar spectral energy distributions.</p>

sedkit combines public Gaia DR3 XP spectra with Gaia-linked 2MASS and
AllWISE photometry. Its stellar model is a hybrid SED emulator: a network
trained on real spectra where stars with good labels exist, synthetic
atmospheres calibrated on real stars where they do not, and PARSEC stellar
evolution tying both to mass, age and metallicity. It predicts absolute
fluxes at the Gaia parallax for a single star or a coeval binary, tests red
giants for hot companions, fits hot subdwarfs with or without a cool
companion, and tells whether a Gaia astrometric orbit comes from a faint
companion, a hidden near-equal-mass twin or a luminous star of another kind. The package runs
on NumPy and SciPy, with no J-CAPS installation, JAX, GPU or separate model
download.

## How the model is built

The model predicts absolute F_lambda at 10 pc on 168 channels: 61 Gaia XP
points at 392--992 nm, 2MASS J/H/Ks, WISE W1/W2 and 102 SPHEREx channels,
in 1e-18 W m^-2 nm^-1. Each part of the HR diagram comes from the source
best constrained there:

| Stars | Teff | Spectral shape | Absolute flux | Teff scale | Model error, XP 0.4--1 um |
| --- | --- | --- | --- | --- | --- |
| Cool dwarfs (M, K, G) | 2800--6250 K | empirical network trained on Gaia stars within 100 pc | PARSEC M_Ks | ASPCAP below 4500 K, IRFM above | 9% below 3800 K, 2% above 4200 K |
| Warm dwarfs (F, early A) | 6250--7498 K | the same network, extended with 1661 dwarfs within 1 kpc | PARSEC M_Ks | IRFM | 2% |
| Hot dwarfs (A, B) | 7000--30000 K | CK04 and TLUSTY spectra through the XP forward model, with an empirical correction | PARSEC radius x model surface flux | spectroscopic or spectral type; IRFM at 7000--7500 K | 1.2% |
| Red giants (`fit_giant_companion`) | 3600--6800 K | empirical template of APOGEE giants | free scale, or a PARSEC M_Ks prior | APOGEE ASPCAP | 3--13% at 392 nm, 3--10% at 442 nm |
| DA white dwarfs (`fit_whitedwarf_companion`) | 6--80 kK, log g 7--9.5 | Koester spectra calibrated to real single-DA spectral labels and Gaia absolute XP fluxes | thick/thin-H C/O cooling tracks, or free WD radius | 294 calibration stars and 289 independent single DAs | independent Teff bias +1.8%, scatter 3.6%; absolute-flux bias +1.9%, source scatter about 14% |
| Hot subdwarfs (`fit_subdwarf_companion`) | 20--45 kK, log g 5--6.5 | TMAP NLTE spectra through the XP forward model, with an empirical correction | free radius at the parallax | Dawson et al. (2026) spectroscopy | 1.3% (0.5--1.4% at 0.45--0.9 um) |

### Cool and warm dwarfs: an empirical network

- **Network.** The J-CAPS v2.1 network is a 16-by-16 GELU MLP whose stored
  weights NumPy evaluates. From Teff, G-Ks and [M/H] it predicts the SED
  shape relative to Ks on all 168 channels. M_Ks then sets the absolute
  level: F = shape x F_Ks(0) x 10**(-0.4 M_Ks).
- **Cool stars.** The training stars are Gaia DR3 dwarfs within 100 pc with
  APOGEE labels, observed in XP, 2MASS, unWISE and SPHEREx. Teff follows
  the infrared-flux method (Casagrande et al. 2021) at 4500 K and above and
  ASPCAP below. Below 3400 K, a three-mode spectral calibration, switched
  off smoothly between 2800 and 3400 K, corrects the coolest M dwarfs.
- **Warm stars.** The network was retrained with 1661 LAMOST/APOGEE dwarfs
  at 6250--7500 K within 1 kpc and Edenhofer et al. (2023) E < 0.05, on the
  IRFM Teff scale. These stars have no SPHEREx spectra, and 7250--7500 K
  holds only 137 of them.
- **Model error.** A fractional covariance (four eigen-columns plus a
  diagonal) is built from training residuals: a cold term below 3800 K and
  a warm term above 4200 K, blended in between. No synthetic spectra enter
  this part, and the network does not extrapolate beyond its training
  coverage: 2800--7498 K at G-Ks >= 0.51, plus a sparse ultracool box at
  2313--2929 K, M_Ks 8.86--10.58, that holds the lowest PARSEC masses
  (0.1 solar masses at 5 Gyr is 2451 K) without validated accuracy.

### Hot dwarfs: calibrated synthetic spectra

- **Synthetic spectra.** CK04 (Castelli & Kurucz 2003) spectra cover Teff
  below 15 kK and TLUSTY BSTAR2006 (Lanz & Hubeny 2007) spectra 15--30 kK,
  both at solar abundance. The two grids differ by about 7 per cent at
  15 kK, so over 11--15 kK CK04 is moved onto TLUSTY by their 15 kK ratio
  and the table is continuous.
- **Table.** A forward model of the Gaia XP external calibration passes
  each spectrum onto the XP channels, and J/H/Ks are sampled directly. The
  result is tabulated on Teff 7000--30000 K (250 K steps) by log g 3--4.75
  (0.05 dex) for R = 1 Rsun at 10 pc.
- **Empirical correction.** Each channel is multiplied by exp(a + W b + s c):
  - a is a per-channel offset of the XP operator;
  - b scales with the Balmer line index W of the synthetic spectrum;
  - c is a cool-edge term that fades out between 7000 and 9000 K.

  The 192 coefficients are fitted to 186 hot anchors and 348 IRFM dwarfs at
  7000--7500 K. The anchors are radial-velocity-constant stars and SB1s,
  without giants, Be or peculiar stars. On 46 held-out anchors and 86 held-out
  dwarfs, the correction lowers the XP shape rms from 4.6--5.1 to
  0.9--1.3 per cent; on 10 independent CALSPEC stars, from 4.7 to 1.8 per
  cent.
- **Absolute flux.** PARSEC gives Teff, log g and radius, and the flux is
  the table times R**2. Over 7000--7498 K the network and the table are
  blended smoothly, with their error terms, so the Teff scale is continuous
  across the handover.
- **Coverage.** The hot route predicts no W1/W2 or SPHEREx and is
  calibrated for |[M/H]| <= 0.3.

### Stellar evolution: PARSEC

- **Isochrones.** PARSEC v1.2S isochrones cover [M/H] -1.0 to +0.5 in
  0.1 dex and log age 6.6--10.0 in 0.05 dex. They hold the pre-main
  sequence and main sequence, end before the contraction hook, and are
  interpolated linearly in mass, log age and [M/H].
- **What they supply.** For a mass, age and [M/H], PARSEC supplies the
  network's Teff, M_Ks and G-Ks, and the hot table's Teff, log g and
  radius.
- **Binaries.** The two components of a binary share age and [M/H], and
  their fluxes are added.
- **Distance.** The observed model is flux_10pc x (parallax / 100 mas)**2,
  with no free normalisation.

### Red giants: an empirical template

- **Template.** `fit_giant_companion` does not emulate PARSEC giants. Its
  `GiantTemplate` is built from 76210 APOGEE DR17 giants with Gaia XP, 2MASS
  and AllWISE, each dereddened by 0.86 SFD E(B-V). A kernel-weighted
  local-linear regression gives the template and its residual covariance on
  a 100 K x 0.2 dex x 0.2 dex grid in Teff, log g and [M/H].
- **Luminosity.** The giant's luminosity is a free scale, or, with
  `luminosity="parsec"`, it is tied to the fitted parallax by the M_Ks
  density of PARSEC giants near its labels.
- **Companion.** The companion is the hot-dwarf model above.

See [model and limitations](docs/model.md) for the full description and
[data and provenance](docs/data.md) for the build scripts of each asset.

## Capabilities and scope

### Data

- Gaia DR3 XP spectra on 61 channels at 392--992 nm, calibrated with
  GaiaXPy from the public continuous spectra.
- 2MASS J/H/Ks and AllWISE W1/W2 from Gaia's best-neighbour tables. A band
  enters the fit only for a unique point-source match with A-quality
  photometry. W1/W2 are held out of `fit` by default.
- Optional SPHEREx QR2 spectra ([SPHEREx downloads](docs/spherex.md)).
- Gaia G/BP/RP are kept as metadata, not fit channels. Outside the
  subdwarf route no ultraviolet data and no XP below 392 nm enter a fit, so
  the Balmer jump is not used; the subdwarf route takes optional XP at
  332--382 nm and GALEX FUV/NUV ([subdwarf validation](docs/validation-subdwarf.md)).
- Models predict absolute fluxes at the parallax, with no free
  normalisation. The parallax is fixed at the catalogue value or fitted
  within three catalogue standard deviations under its Gaussian constraint
  (`fit_parallax=True` in `fit`, `luminosity=` in `fit_giant_companion`).

### Models

| Entry point | Stars | Support | Hypotheses |
| --- | --- | --- | --- |
| `fit` with `StellarModel()` | FGKM dwarfs: empirical J-CAPS network on PARSEC tracks | Teff 2800--7498 K (sparse ultracool support to 2313 K), age 0.5--10 Gyr, [M/H] -1.0 to +0.5; primary mass to 1.5--1.8 solar masses at 0.5--2 Gyr and 1.38 at 3 Gyr (solar [M/H]) | single star, coeval binary |
| `fit` with `StellarModel(hot=True)` | adds A and B dwarfs: CK04/TLUSTY table with an empirical XP correction | 7000--30000 K, log g 3--4.75, [M/H] -0.3 to +0.3, age from 4 Myr, mass to 20 solar masses; XP and J/H/Ks only | single star, coeval binary |
| `fit_giant_companion` | red giant (empirical APOGEE template) plus a hot main-sequence companion | giant Teff 3600--6800 K, log g 0--3.8, [M/H] -2.6 to +0.6; companion 1.5--15 solar masses at 10 Myr and solar [M/H] | giant alone, giant plus companion profiled in mass |
| `fit_subdwarf_companion` | hot subdwarf (TMAP NLTE table with an XP correction) plus a cool dwarf or subgiant of another age | subdwarf 20--45 kK, log g 5--6.5, three helium tiers (He-rich from 32 kK); companion as `StellarModel()`, or the giant template at log g 3.2--3.8 | single FGK star, single subdwarf, subdwarf plus companion |
| `fit_whitedwarf_companion`, `whitedwarf_light_limit` | DA white dwarf plus a dwarf; conditional WD light envelope for a luminous primary | DA table 6--80 kK, log g 7--9.5; C/O cooling tracks with thick or thin H; strongest label validation below 40 kK | single dwarf, single DA, DA plus dwarf |
| `orbit.solve_orbit`, `rank_roots` | Gaia photocentre orbits | as `StellarModel()` | faint companion, luminous twin |
| `orbit.solve_luminous_pair`, `branch_from_rv`, `solve_dark_companion` | photocentre orbits of two luminous stars of different kinds, e.g. sdB + FGK | any masses | both branches of a0 = a \|B - beta_G\| |

Fits stay inside these ranges; the models do not extrapolate.
See [model and limitations](docs/model.md).

### What the fits measure

White dwarfs: the default emulator is anchored to spectroscopic Teff/log g
and absolute Gaia fluxes of 294 single DAs. On 289 independent stars,
with extinction and parallax constraints, the cooling-relation fit gives
1.8% temperature bias and 3.6% robust scatter. Its absolute-flux bias is
1.9%, with about 14% source scatter. Masses remain conditional on the
C/O cooling relation and reference gravity scale. See the
[single-star calibration](reports/wd_single_scale_20261010/report.md) and
[composite/orbit benchmarks](docs/validation-whitedwarf.md).

Single stars:

- **FGK dwarfs.** Hyades, Praesepe and Coma Ber members at 6000--7500 K
  fit with chi2/N of 0.7--0.9 at free age, with masses 0.02--0.05 solar
  masses below the isochrone mass. Free ages come out older than the
  literature cluster ages (1.0--2.1 against 0.6--0.7 Gyr)
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
  reddened control giants (maximum 8.8), the sample that also set
  `tilt_sigma`; its false-positive rate on other giants awaits an
  independent control set.
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

Hot subdwarfs and their companions:

- **Single subdwarfs.** For 157 Dawson et al. (2026) subdwarfs fitted out of
  fold with their spectroscopic log g, Teff lies 0.4 kK above the
  spectroscopic value (scatter 1.7 kK with XP at 332--382 nm, 2.6 kK
  without) and the radius 1.3 per cent above their SED radii (scatter
  4 per cent). Teff is compressed towards 30 kK (+1.1 kK below 28 kK,
  -1.0 kK above 36 kK) and follows Dawson's scale, 1.05 kK below Luo et
  al. (2021). The XP channels at 332--382 nm are validated for single
  subdwarfs, not for composites; GALEX pulls Teff 5 kK low and is not used
  by default.
- **Compact companions.** For 5 subdwarfs with a compact-companion orbit
  and 16 sdB+WD systems the composite is never preferred (Delta <= 2.8);
  dwarf companions above 0.12--0.45 solar masses are excluded.
- **Light fraction.** In injections with real noise and model mismatch,
  beta_G has a scatter of 0.005 for K/M companions, 0.025--0.031 for G and
  0.04 for F, and an E prior 0.03 too high biases it by +0.02. Through a
  photocentre orbit this gives M_sdB to 0.09--0.10 solar masses for G and
  F companions; for K/M companions, with M_sdB = 0.47 +- 0.05, the orbit
  gives the companion's dynamical mass to 0.04. A companion 0.1 solar
  masses below its isochrone mass biases M_sdB by +0.07 (G) to +0.18 (K/M).
- **Detection.** Requiring Delta > 25 and beta_G > 0.2 flags none of 230
  RV-constant FGK dwarfs and recovers composites in which the subdwarf
  gives more than half of the 400--450 nm light
  ([subdwarf validation](docs/validation-subdwarf.md)).

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
- Brown dwarfs and dedicated ultracool atmospheres, triples, blends and
  variable stars.
- Stars above 30 kK (O stars are untested), hot stars with |[M/H]| > 0.3,
  supergiants, Be and emission-line stars, chemically peculiar stars and
  fast rotators.
- Binaries of different ages, except the giant, subdwarf and DA routes.
- Hot subdwarfs outside 20--45 kK and log g 5--6.5, He-rich subdwarfs below
  32 kK, HW Vir-type close binaries with a reflection effect, pulsating
  subdwarfs' light variations, non-DA/magnetic white dwarfs, ELM WDs,
  and double-degenerate systems.
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
- [White-dwarf fitting](docs/whitedwarf.md) and [validation](docs/validation-whitedwarf.md): DA labels, composite light fractions and conditional dark-companion bounds.
- [Subdwarf validation](docs/validation-subdwarf.md): single and composite subdwarfs, light fractions, orbits, injections and false positives.
- [Visual identity](docs/appearance.md): logo, plotting palette and reproducible homepage figures.

## Development

```bash
pip install -e ".[test]"
pytest -q
```

## Credits

The bundled stellar network and hot-star table are extracted from
[J-CAPS](https://github.com/jiadonglee/J-Caps), using PARSEC stellar tracks,
CK04 (Castelli & Kurucz 2003) and TLUSTY BSTAR2006 (Lanz & Hubeny 2007)
model atmospheres. The giant template is built from APOGEE DR17. Hot-subdwarf spectra are
Tuebingen TMAP NLTE models (Werner et al. 2003; Rauch & Deetjen 2003) from
the TheoSSA service of the German Astrophysical Virtual Observatory.
Public XP spectra are calibrated with
[GaiaXPy](https://gaia-dpci.github.io/GaiaXPy-website/).
SPHEREx aperture extraction uses [XphereX](https://github.com/jiadonglee/XphereX),
[TallTable](https://github.com/cmhainje/talltable) and
[SPExPI](https://github.com/fkiwy/spexpi).
Catalogue data are retrieved through
[astroquery](https://astroquery.readthedocs.io/en/latest/gaia/gaia.html).
Please acknowledge Gaia, 2MASS, WISE, APOGEE, PARSEC, CK04, TLUSTY, TMAP
and TheoSSA, GALEX and J-CAPS when using these data and models in research; see
[provenance](docs/data.md).

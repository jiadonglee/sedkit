# Stellar model and limitations

The bundled model uses the J-CAPS v2.1 Ks-anchored empirical network,
retrained with infrared-flux-method Teff (Casagrande et al. 2021) at 4500 K
and above and with warm dwarfs added to 7500 K, and PARSEC tables. The network is a 16-by-16 GELU MLP with optical and infrared
output heads. NumPy evaluates the stored weights and spectral calibration.
Its 168 output channels are absolute fluxes at 10 pc, in
`1e-18 W m^-2 nm^-1`.

PARSEC maps mass, age and metallicity to Teff, M_Ks and G-Ks. The tables
are PARSEC v1.2S isochrones at [M/H] -1.0 to +0.5 in 0.1 dex and log age
8.50 to 10.00 in 0.05 dex, pre-main sequence and main sequence only
([data](data.md)). They are interpolated linearly in mass, log age and
[M/H]. Each isochrone ends before the overall-contraction hook, and a mass
is supported up to the lower turn-off of the two neighbouring ages.
Against PARSEC isochrones at the intermediate ages, for supported stars
below 1.5 solar masses at [M/H] = -0.5, 0 and +0.3, the 99th-percentile
differences are at most 15 K in Teff and 0.02 mag in Ks and G.

Two components share age and metallicity. Their predicted fluxes are added
before distance scaling. The observed-scale model is `flux_10pc * (parallax_mas / 100)**2`.
No median normalization or unconstrained amplitude enters the fit.

The five broadband channels are learned catalogue-equivalent fluxes.
They are not obtained by interpolating the model spectrum at five
wavelengths. The model does not provide general synthetic photometry for
arbitrary filters or establish independent passband closure.

The likelihood includes measurement variance and the bundled fractional
model covariance, with a low-rank basis plus a diagonal term. The primary's
Teff selects the error term for both components: the cold term below
3800 K, the warm term above 4200 K, and a smoothstep-weighted sum of the
two covariances in between, so the likelihood is continuous in mass. The
components are fully correlated. The covariance determinant is retained. Parameters are fitted
from multiple supported grid starts.

The G-band ratio uses the same PARSEC component magnitudes:
`beta_g = 10**(-0.4 * (M_G[1] - M_G[0]))`. It is model-dependent, rather
than an independent observation of resolved component light.

## Limitations

- Fits are exploratory local optima. No posterior uncertainty, calibrated
  binary probability or population inference is provided.
- Extinction defaults to zero. Use `extinction=None` to fit it with an
  [Edenhofer dust prior](extinction.md), or a number to fix ZGR23 E.
- Age covers 0.5--10 Gyr, [M/H] -1--0.5. Both stars must lie inside the
  original network coverage. There is no atmosphere/BD fallback. Coverage
  of sparse ultracool training points is not a validated accuracy range.
- The stellar network is trained on dwarfs: Gaia stars within 100 pc with
  APOGEE labels and, at 6250--7500 K, 1661 LAMOST/APOGEE dwarfs within
  1 kpc at Edenhofer et al. (2023) E < 0.05. Teff follows the IRFM scale at
  4500 K and above and ASPCAP below. Coverage is Teff 2800--7498 K and G-Ks
  down to 0.51; 7250--7500 K holds only 137 stars. The supported primary
  mass at solar metallicity reaches 1.5--1.8 solar masses at 0.5--2 Gyr
  and 1.38 at 3 Gyr; 1.4 solar masses at [M/H] = -0.5 (about 8100 K) is
  outside. Stars near the turn-off are outside the tables; gaps and support
  boundaries remain explicit, without extrapolation.
- For warm primaries (1.2--1.6 solar masses), a free-age single-star fit
  reproduces a companion's light by main-sequence evolution: with age free,
  XP and 2MASS neither detect the companion nor constrain q. Binary
  inference there needs a known age, checked against main-sequence stars of
  the same cluster ([validation](validation-warm.md)).
- The 1 kpc training stars have no SPHEREx spectra. SPHEREx predictions
  above about 6400 K are extrapolated, and the warm model-error term for
  SPHEREx channels comes from cooler stars.
- GaiaXPy inter-channel measurement covariance is omitted; XP marginal
  errors are used. The model covariance does not replace that information.
- Variable sources, blends, triples and white dwarfs are not modeled.
  Catalogue flags are preserved but do not establish a clean binary sample.
- Fixed age/metallicity experiments are conditional on those choices.
  A matched mock checks the algorithm, not real-data model calibration.

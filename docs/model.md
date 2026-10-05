# Stellar model and limitations

The bundled model uses the J-CAPS v2.1 Ks-anchored empirical network,
retrained with infrared-flux-method Teff (Casagrande et al. 2021) at 4500 K
and above and with warm dwarfs added to 7500 K, and PARSEC tables. The network is a 16-by-16 GELU MLP with optical and infrared
output heads. NumPy evaluates the stored weights and spectral calibration.
Its 168 output channels are absolute fluxes at 10 pc, in
`1e-18 W m^-2 nm^-1`.

PARSEC maps mass, age and metallicity to Teff, M_Ks and G-Ks. The tables
are PARSEC v1.2S isochrones at [M/H] -1.0 to +0.5 in 0.1 dex and log age
6.60 to 10.00 in 0.05 dex, pre-main sequence and main sequence only
([data](data.md)). They are interpolated linearly in mass, log age and
[M/H]. Each isochrone ends before the overall-contraction hook and, above
its Teff maximum, before the first star with log g < 3, and a mass is
supported up to the lower terminal mass of the two neighbouring ages. The
default model uses ages of 0.5 Gyr and older.
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

## Hot-star route

`StellarModel(hot=True)` adds components hotter than the network. Above
7000 K a component's flux comes from a channel table: CK04 (Castelli &
Kurucz 2003) below 15 kK and TLUSTY BSTAR2006 above, solar abundances,
passed through a forward model of the Gaia XP external calibration and
tabulated as XP61 and J/H/Ks at 10 pc for R = 1 Rsun. PARSEC supplies Teff,
log g and radius, and the flux scales as R**2. An empirical correction
multiplies the table: per channel, `exp(a + W b + s c)`, with W the Balmer
(H-gamma + H-beta) line-strength index of the synthetic spectrum and s a
cool-edge weight, 1 at 7000 K falling smoothly to 0 at 9000 K. a and b
correct mainly the XP operator, not the synthetic spectra; c is set by
1 kpc IRFM dwarfs at 7000--7500 K held at their IRFM Teff, the PARSEC log g
of their network fit and the Edenhofer map E, the conditions of a sedkit
fit, so the table shares the Teff scale of the network at the seam
([hot validation](validation-hot.md)).

Over 7000--7498 K, to the end of the network's training range, the
component flux is a smoothstep-weighted sum of the network and the table. The
primary's Teff hands the model-error term from v2.1_warm to the hot term
over the same range. The hot route predicts no W1/W2 or SPHEREx channels,
so `hot=True` fits use XP and J/H/Ks for every hypothesis, including
cool-star fits. Ages start at 10**6.6 yr and primary masses reach
20 solar masses.

## Limitations

- Fits are exploratory local optima. No posterior uncertainty, calibrated
  binary probability or population inference is provided.
- Extinction defaults to zero. Use `extinction=None` to fit it with an
  [Edenhofer dust prior](extinction.md), or a number to fix ZGR23 E.
- Age covers 0.5--10 Gyr (10**6.6 yr--10 Gyr with `hot=True`), [M/H]
  -1--0.5. Both stars must lie inside the original network coverage or,
  with `hot=True`, inside the hot support: 7000--30000 K, log g 3--4.75 and
  [M/H] -0.3--0.3. There is no atmosphere/BD fallback. Coverage
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
- The hot table is solar and its correction is calibrated to 30 kK on
  anchors with spectral-type Teff above 15 kK. Out-of-fold residuals are
  1.1--1.3 per cent above 9 kK and 1.2 per cent at 7.5--9 kK.
- At the same PARSEC star in 7000--7498 K, the solar table and the
  metallicity-dependent network differ in XP shape by 0.8 per cent at
  [M/H] = 0 and 1.6 per cent at -0.3 and +0.3, and in XP level by 0, 2 and
  4 per cent. Fits of 7000--7500 K dwarfs give Teff 3--21 K above IRFM.
  The cool-edge correction carries the map E of its calibration dwarfs, so
  map E errors enter the table at 7000--9000 K. Between 7.5 and 9 kK there
  is no IRFM-quality Teff reference.
- With `hot=True`, isochrones with |[M/H]| > 0.3 end at 7000 K, where the
  network alone reaches 7498 K. The 10**6.6 yr age floor applies to every
  star: about half of 6250--7000 K field dwarfs fitted with free age reach
  pre-main-sequence solutions below 0.5 Gyr, with the objective within 2
  of the network-only fit for 89 per cent of the 481.
  Rotation, emission, pulsation and chemical peculiarity are not modelled.
- XP and J/H/Ks alone do not constrain extinction for hot stars: fixed or
  dust-prior E carries the constraint. In noiseless injections with a
  Gaussian E prior of width sqrt(0.03**2 + (0.1 E)**2), the 1-sigma Teff
  width is 0.6--1.0 kK at 15 kK and 1.6--2.8 kK at 25 kK (E = 0--0.6).

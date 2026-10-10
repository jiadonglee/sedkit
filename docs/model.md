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

## Giant route

`fit_giant_companion` tests a red giant for a hot main-sequence companion,
the configuration of a stripped giant with a B-star companion. The giant
is not a PARSEC star: `GiantTemplate` is an empirical flux template in
Teff, log g and [M/H] with a free scale, built from 76210 APOGEE DR17
giants with Gaia XP, 2MASS and AllWISE (ASPCAP S/N > 70, SFD
E(B-V) < 0.1, RV scatter below 1 km/s, RUWE < 1.4). Each training star is
dereddened by E = 0.86 SFD E(B-V) on the ZGR23 curve; 0.86 is the slope of
per-star E fitted against SFD. At every node of a 100 K x 0.2 dex x
0.2 dex grid, a kernel-weighted local-linear regression in the three
labels gives the template, and the weighted residuals, less the
measurement variance, give its fractional covariance as six columns plus a
diagonal. Templates are trilinear between nodes, and the covariance is the
weighted sum of the eight corner covariances. The grid covers Teff
3600--6800 K, log g 0--3.8 and [M/H] -2.6 to +0.6 where the kernel holds
at least 25 effective stars and the template is positive with scatter
below 30 per cent on every XP channel. Typical template scatter is 4--7
per cent at 392--402 nm and 3--4 per cent at 440 nm.

The companion is `StellarModel(hot=True)` of mass M2 at 10 Myr and solar
metallicity, at the parallax; W1/W2 follow the Rayleigh--Jeans
tail of its Ks flux. Giant and companion share ZGR23 E >= 0 on a curve
multiplied by (lambda / 0.55 micron)**tilt, with a Gaussian tilt prior
(default width 0.15, the scatter of free tilts of reddened control
giants). They share nothing else: no common age and no q <= 1. Label
priors on the template's APOGEE scale are required. The fit profiles the
objective, -2 ln L with the template and companion covariances plus the
priors, over a grid of M2. For each M2, the scale, E and tilt are fitted
at every template node within 3.5 prior widths, and the best three nodes
are refined in all six parameters.

With `luminosity=`, the parallax becomes a seventh parameter shared by both
stars, within 3 sigma of its input value with penalty z^2. The giant's M_Ks
follows from its scale, the template Ks and the parallax; its luminosity
from the PARSEC BC_Ks at its labels; its mass from log g and Teff. Gaussian
walls of 0.1 dex hold the mass in 0.2--10 solar masses. For "parsec", a
prior from PARSEC v1.2S giants (subgiant branch to TP-AGB, log g < 3.8)
adds -2 ln(p / p_mode) of M_Ks near the labels:

- Isochrones are at 0.05 dex in log age from 10**7.5 yr and 0.1 dex in
  [M/H] over -1.0 to +0.5, with a 0.5 dex table outside that range.
- Points are weighted by the IMF, the linear age width and
  (age / 1 Gyr)**(4 max(0, -[M/H])), which gives metal-poor isochrones old
  ages. The exponent is fitted to template training giants with
  parallax/error > 10.
- The kernel is 200 K in Teff, 0.12 in log g and 0.15 dex in [M/H].
- The M_Ks histogram, smoothed by 0.15 mag, is mixed with a uniform floor
  so that any M_Ks PARSEC produces near the labels costs at most 6.
- Nodes without PARSEC giants in the kernel (1142 of 6603) carry no M_Ks
  constraint; there the mass walls alone bound the luminosity.

With `dust_prior=`, E follows the Edenhofer et al. (2023) map at the trial
distance, with the map width widened by 0.04 in quadrature. Beyond
1.2 kpc, near the map's 1.25 kpc edge, E is held above the 1.2 kpc value
minus 0.04 by a one-sided wall.

## Subdwarf route

`fit_subdwarf_companion` fits a hot subdwarf and, optionally, a cool
companion of a different age and evolutionary stage. The subdwarf is a
Tuebingen TMAP NLTE spectrum ([data](data.md#hot-subdwarf-atmospheres))
passed through the forward model of the Gaia XP external calibration that
built the hot table, and tabulated for R = 1 Rsun at 10 pc on Teff x
log g in three helium tiers:

| Tier | Atmosphere | log(He/H) | Teff | log g |
| --- | --- | --- | --- | --- |
| `H` | pure hydrogen | -- | 20--45 kK | 5.0--6.5 |
| `mid` | H+He+C | -1.9 | 32--45 kK | 5.0--6.5 |
| `He` | H+He+C | -0.1 | 32--45 kK | 5.0--6.5 |

The `H` tier stands for He-poor subdwarfs (log(He/H) <~ -2): at
32--45 kK it differs from the `mid` tier by 0.3--0.5 per cent in XP shape
and 4--6 per cent in surface flux, that is 2--3 per cent in radius. Inputs
outside a tier are rejected, not extrapolated. Eleven TheoSSA models that
sit 2--11 per cent off their neighbours are replaced by their log g
neighbours; node-to-node roughness of up to 1.2 per cent remains in the
32 kK row of the `He` tier.

Each channel carries the hot-table operator correction (per-channel
offset and Balmer-index term) and a subdwarf correction exp(a + W b), with
W the Balmer index of the TMAP spectrum, fitted per channel to 157 single
subdwarfs of Dawson et al. (2026) at their spectroscopic Teff and log g,
with Edenhofer E and free radius. It covers XP, J/H/Ks and the six XP
channels at 332--382 nm; its median size over XP is 1.3 per cent and it
reaches 5--6 per cent at the XP edges and H-alpha. W1/W2 and SPHEREx are
the TMAP spectrum averaged over each channel. The model error is the hot
term on XP and J/H/Ks, 3 per cent on W1/W2 and SPHEREx, 5 per cent on the
blue XP channels and on GALEX.

The radius is free and the flux scales as R^2 at the parallax; the mass is
M = g R^2 / G and the luminosity R^2 (Teff / Teff_sun)^4. XP and 2MASS
constrain log g weakly: without a spectroscopic prior the fit drifts to the
table edge, so the mass needs a spectroscopic log g. In a composite the
subdwarf's Teff and radius trade against the companion, and a log g prior
alone can leave it at 1--1.4 solar masses; `subdwarf_prior` also takes
`mass` and `radius`.

The companion is a dwarf from `StellarModel()` (PARSEC mass, age and [M/H]
through the network, supported Teff 2800--7498 K) or a subgiant from
`GiantTemplate` with log g 3.2--3.8 and a free scale; its luminosity and
radius follow from its Ks flux, the parallax and the PARSEC BC_Ks, and its
mass g R^2 / G is held to 0.7--3 solar masses by walls of 0.1 dex. The two
stars share the parallax and ZGR23 E, nothing else. The single-FGK
hypothesis is one `StellarModel()` star. All hypotheses use one data
vector: XP and J/H/Ks (W1/W2 with `use_wise`, SPHEREx with `use_spherex`
and without the subgiant), plus optional blue XP and GALEX points. Below
392 nm the extinction curve is Gordon et al. (2023) R_V = 3.1 scaled to
ZGR23 over 392--550 nm. A companion's flux below 392 nm is a blackbody
joined to its 392--412 nm flux, with a 50 per cent error.

Light fractions use a 2 nm spectrum of each component on the XP channel
scale: the channels in 392--992 nm, the model spectrum beyond, joined at
the edge channels (for the companion a blackbody at its Teff). beta_G,
beta_BP and beta_RP are the subdwarf's share of the photon-weighted,
reddened flux in the Gaia DR3 passbands.

The fit profiles the objective over the subdwarf Teff: nodes every 2 kK in
each tier, every 0.5 kK within 3 kK of the best node, with all other
parameters refitted at each node, then a free polish. `ranges` spans each
reported quantity over the profile points and the interpolated crossings
within 1 of the minimum: a profile interval in Teff, not a marginal
posterior.

## Limitations

- Fits are exploratory local optima. No posterior uncertainty, calibrated
  binary probability or population inference is provided.
- With `hot=True`, Delta measures only the luminosity excess over the
  single-star model; it does not identify an individual hot binary (see
  [hot-star validation](validation-hot.md)).
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
  down to 0.51, plus a sparse ultracool box at 2313--2929 K and M_Ks
  8.86--10.58 that holds the lowest PARSEC masses (0.1 solar masses at
  5 Gyr is 2451 K); 7250--7500 K holds only 137 stars. The supported primary
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
- Variable sources, blends and triples are not modeled. DA white dwarfs
  use the separate [WD route](whitedwarf.md).
  Catalogue flags are preserved but do not establish a clean binary sample.
- Fixed age/metallicity experiments are conditional on those choices.
  A matched mock checks the algorithm, not real-data model calibration.
- The hot table is solar and its correction is calibrated to 30 kK on
  anchors with spectral-type Teff above 15 kK. Out-of-fold residuals are
  1.1--1.3 per cent above 9 kK and 1.2 per cent at 7.5--9 kK.
- At the same PARSEC star in 7000--7498 K, the solar table and the
  metallicity-dependent network differ in XP shape by 0.8 per cent at
  [M/H] = 0 and 1.6 per cent at -0.3 and +0.3, and in XP level by 0, 2 and
  4 per cent. Fits of 7000--7500 K dwarfs give Teff 2--21 K above IRFM.
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
- The giant route uses XP from 392 nm, so the Balmer jump is not used. Its
  companion is a non-rotating main-sequence star, and a greyer extinction
  curve also raises the blue end; the tilt prior carries that distinction
  ([giant validation](validation-giant.md)).
- The PARSEC M_Ks prior of the giant route sits 0.1--0.17 mag brighter than
  training giants and controls at [M/H] < -0.5, and outside [M/H] -1.0 to
  +0.5 its isochrones are 0.5 dex apart in age. `download` stores the Gaia
  DR3 parallax without the Lindegren et al. (2021) zero point; the validated
  luminosity fits used corrected parallaxes passed through `parallax=`.

## DA white-dwarf route

The [DA route](whitedwarf.md) combines Koester surface spectra with
Bédard C/O-core cooling tracks. It compares a dwarf, a DA and their
physical flux sum at the shared distance/extinction. The WD-only XP
calibration retains absolute flux at spectral Teff/log g and Gaia parallax.
Its log-temperature spline and gravity term are trained on real single
DAs with source-grouped folds. Empirical WD errors include a correlated
normalization term and diagonal shape variance; stellar errors retain their
low-rank covariance. The likelihood includes both covariance determinants.

The default WD radius follows its temperature and mass, with thick H.
Thin-H tracks and a free WD radius are available for sensitivity and
spectroscopic comparisons. Companion age/metallicity can be fitted.
A temperature/mass profile gives a conditional WD G-light envelope for
orbit calculations. Its operational threshold has no calibrated coverage.
The [validation](validation-whitedwarf.md) measures temperature,
light-fraction and mass sensitivity separately.

# DA white dwarfs and their companions

`WhiteDwarfModel`, `fit_whitedwarf_companion` and
`whitedwarf_light_limit` provide DA spectra, dwarf/DA/composite fits,
and conditional G-band light envelopes for a WD companion.

```python
from sedkit import WhiteDwarfModel, fit_whitedwarf_companion

result = fit_whitedwarf_companion(sed, companion_feh=0.)
composite = result["hypotheses"]["wd+dwarf"]
print(composite["whitedwarf"], composite["companion"])
print(composite["fractions"]["beta_G"])
```

The hypotheses share observed channels, extinction and parallax.
The WD has temperature and mass; radius and cooling age come from
Bédard C/O-core tracks. The dwarf uses `StellarModel()` and has its own
age and metallicity, fixed to 5 Gyr and solar composition by default.
Setting either to `None` fits it. `stellar=StellarModel(hot=True)` enables
the warm/hot stellar route where needed.

`free_radius=True` fits WD temperature, gravity and radius independently.
The reported mass then follows `g R²/G`; cooling age is unavailable.
For example, `whitedwarf_prior={"logg": (8.0, 0.1)}` supplies a
spectroscopic gravity constraint. A prior can constrain WD `teff`,
`mass`, `logg` or `radius`, and dwarf `teff`, `mass`, `logg`, `radius`,
`age_gyr` or `feh`.

Temperatures are in K, masses/radii in solar units, ages in Gyr,
parallaxes in mas and extinction in native ZGR23 E.
`extinction=None` needs either `extinction_prior=(mean, sigma)` or
`dust_prior=EdenhoferPrior(...)`. `fit_parallax=True` fits the catalogue
Gaussian parallax constraint within three sigma and updates the dust
prior at the trial distance. Both components receive the same attenuation.

## Models and data

The bundled Koester DA table contains 858 atmosphere nodes at
6000--80000 K and log g 7--9.5. It predicts absolute 10-pc fluxes
through the same XP forward operator as the hot-star tables.
Its wavelength coverage is approximately 90--3000 nm:
XP61, J/H/Ks and 59 short SPHEREx channels are supported.
W1/W2 and the longer SPHEREx channels are masked.
SPHEREx is included only with `use_spherex=True`.

The default `WhiteDwarfModel()` applies a WD-only `exp(a + W b)`
shape correction and empirical diagonal model error. W uses the DA
Balmer width, with a maximum of 10.4357 nm. It inherits no hot-star
or sdB correction. `WhiteDwarfModel(calibration=None)` uses raw spectra.
`WhiteDwarfModel(hydrogen_layer="thin")` selects the thin-H cooling
tracks for a direct sensitivity comparison; the default is thick H.

Optional `blue=(flux6, error6)` supplies XP at 332--382 nm, in SED flux
units. Optional `galex={"FUV": (AB_mag, mag_error, usable), ...}`
adds FUV/NUV measurements. The bundled UV passband correction and
model errors come from 71 local DA anchors at 6.8--41.4 kK. They affect
GALEX predictions, not the returned coarse continuum. Outside this
temperature range, the UV likelihood uses an adopted fractional model-error
floor of 0.5. GALEX is absent
unless explicitly supplied. Match it with proper motion and reject
saturated, contaminated or ambiguous measurements.

`objective` is minus twice the likelihood plus parameter-prior penalties,
including the model covariance determinant. `chi2` and `minus2lnL`
are also returned. `preferred` names the smallest objective;
`detection_statistic` is the composite improvement over the better
single-star model. This statistic is a diagnostic, not a probability.
The XP-only validation used a threshold of 55 fixed on its training
controls; changing the data, error model or priors changes its scope.

For a detected WD+dwarf composite, `photocentre` reports the WD as star 1,
the dwarf as star 2, and the signed coefficient `B_WD-beta_G` along the
WD-to-dwarf relative vector. This can have either sign. For such pairs,
`solve_luminous_pair(..., m2=dwarf_mass, beta=beta_G)` retains both branches
and solves the WD mass. `solve_dark_companion` is appropriate for the
faint-WD mode with the luminous dwarf as star 1.

## Conditional WD light limits

```python
from sedkit import whitedwarf_light_limit
from sedkit.orbit import solve_dark_companion

limit = whitedwarf_light_limit(
    sed, masses=[0.6, 0.8, 1.0], delta=9.,
    companion_age_gyr=None, companion_feh=None,
    extinction=None, dust_prior=dust, fit_parallax=True,
)
zero_light = solve_dark_companion(a0, parallax, period, primary_mass)
with_light = solve_dark_companion(
    a0, parallax, period, primary_mass,
    beta=limit["flux_ratio_G_upper"],
)
```

At each supplied WD mass and temperature, the fit profiles the luminous
primary and shared nuisance parameters. By default it scans the atmosphere
temperature grid. `temperatures=[...]` supplies another grid;
`cooling_ages_gyr=[...]` instead scans supported cooling ages.
Temperature crossings of the allowed-set boundary are refined.
`profile` retains the fitted points and convergence flags; `intervals`
retains disconnected allowed regions. `beta_G_upper` is the largest
allowed WD share of total G light.

The fraction and orbit ratio differ:
`beta_G = F_WD/(F_primary+F_WD)` and
`flux_ratio_G_upper = beta_G_upper/(1-beta_G_upper)`.
`solve_dark_companion` takes the latter. Use the same primary mass,
parallax and orbit convention when assessing the resulting mass shift.

`luminous_mass=...` additionally fits a coeval MS companion of that
specified mass and returns its objective difference. It tests that
hypothesis, rather than excluding every possible luminous companion.

`galex_upper_limits={"FUV": flux_cap, "NUV": flux_cap}` optionally
requires the WD's UV contribution to fit below measured total system
light. It does not need a primary-star UV template. The primary, extinction
and parallax are refitted under this constraint, and disconnected optical/UV
allowed regions are retained within the supplied temperature range.

For an accepted GALEX AB magnitude `mag` with uncertainty `mag_error`,
construct an FUV total-flux cap as follows (including the 0.05-mag FUV
calibration term; NUV uses 0.03 mag):

```python
import numpy as np

wd = WhiteDwarfModel()
pivot_nm = wd.galex_pivot_nm["FUV"]
flux = 3631. * 10**(-0.4 * mag) * 2997924580. / pivot_nm**2
error = flux * np.log(10) / 2.5 * np.hypot(mag_error, 0.05)
light = whitedwarf_light_limit(
    sed, model=wd, masses=[0.6],
    galex_upper_limits={"FUV": flux + 3 * error},
)
```

The bound compares the cap with `WD_UV * exp(-3*sigma_model)`;
`sigma_model` is the bundled UV fractional-error term (0.127 FUV,
0.149 NUV), with an adopted log-flux floor of 0.5 outside the empirically
tested 6812--41430 K range. This is a conservative model allowance,
not calibrated probability coverage. A missing GALEX measurement supplies
no bound; catalogue absence is not a measured nondetection.

`delta=9` defines an operational envelope. `confidence_level=None`
is returned because coverage has not been calibrated. The envelope is
conditional on the supplied mass/temperature or cooling-age range,
composition, cooling model, primary model and nuisance priors. An empty
allowed WD set returns NaN. Inspect fit quality and convergence before
interpreting a bound. See the [real-data validation](validation-whitedwarf.md).

## Joint orbit likelihoods

`loglike_whitedwarf_sed(sed, teff=..., mass=m_wd, primary_mass=m_ms,
parallax_mas=..., age_gyr=..., feh=..., extinction=...)` is a likelihood
atom for joint SED/RV/astrometry models. The same WD mass sets its cooling
radius and enters the external orbit. It includes model variance and its
determinant, but no parameter priors or catalogue parallax penalty; add
those once in the joint model. Omitting `primary_mass` models one DA.
Unsupported components return minus infinity. Fixed-orbit mass shifts
from `whitedwarf_light_limit` do not replace that joint likelihood.

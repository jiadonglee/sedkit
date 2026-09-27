# Connecting to orblet

sedkit supplies the stellar likelihood and light ratio; orblet supplies
orbital predictions and likelihoods. A shared parameter proposal supplies
M1, q, age, metallicity, parallax and orbital elements.
The [executable example](../examples/orblet_20260927/run.py) composes
`loglike_sed + rv_loglike(primary) + rv_loglike(secondary) + loglike_along_scan`.

## Parameter mapping

`StellarModel.evaluate` returns the true component masses, absolute spectra,
G magnitudes and `beta_g = F_G,2/F_G,1`. For a primary-frame orbit, the signed
photocentre scaling is `(q - beta_g) / (q * (1 + beta_g))`.
sedkit returns this as `a_phot_over_a1`; it agrees with orblet's
`signed_photocentre_axis_ratio(q, beta)` convention.

Scale the primary orbit returned by `campbell_xy` by this factor before
passing it to `along_scan_model`. Equal-mass, equal-light stars have zero
photocentre motion even though their RVs vary. Keep the sign when beta > q.

For `rv_model`, pass `mass_msun=M2*sin(i)` for the primary and
`mass_msun=M1*sin(i)` for the secondary. Both use the true total mass
`M_msun=M1+M2`; the secondary argument of periastron is primary omega + pi.
`campbell_xy` instead takes the true companion mass, with inclination explicit.

In these functions, periods are in years, angles in radians, distances in
mas and RVs in km/s. Times are TCB MJD. The example converts its 120-day
period using orblet's `DAYS_PER_KEPLER_YEAR`. Sky-plane RA is delta-alpha-star;
scan angle is counterclockwise from north. No extra cos(dec) is needed.

Use `loglike_sed(..., parallax_mas=shared_parallax)` so the SED flux scale
and astrometry use the same distance. This likelihood adds no catalogue
parallax Gaussian. When epoch astrometry is included, do not independently
reuse the catalogue solution derived from those epochs. Likewise, do not
use a fitted SED posterior as a prior and add its original likelihood again.

orblet's current `loglike_joint` assumes a dark companion. The luminous SB2
example composes individual likelihoods with the corrected photocentre.

## Reproduce the test

Install sedkit and the tested orblet revision in Python 3.11 or 3.12:

```bash
pip install .
pip install "orblet @ git+https://github.com/saharsh1/orblet.git@7aa297df4520d1e93c8a7a5a763a1f2d928bfa51"
python examples/orblet_20260927/run.py
```

On 2026-09-27, seed 20260927 generated 166 fitted SED channels, 48 primary
and 53 secondary RVs, and 100 along-scan measurements. The experiment
recovers M1=0.75045 from 0.75 solar masses, q=0.79977 from 0.8,
inclination=59.882 from 60 degrees, parallax=19.9898 from 20 mas, and
systemic RV=18.0208 from 18 km/s. Three initial points are tried.

Omitting the photocentre correction and refitting gives q=0.5844 and a
worse objective by 24752.9. Equal-light cancellation and the sign reversal
are also checked. [Results](../examples/orblet_20260927/summary.json),
[orbit diagnostics](../examples/orblet_20260927/joint.png) and
[SED diagnostics](../examples/orblet_20260927/sed.png) are included, together
with the simulated observations and PDF figures.

## Limitations

This is a conditional, same-template mock with measurement noise only.
Orbital shape, age, metallicity and astrometric offsets are fixed; five
parameters are fitted. Scan times and parallax factors are idealized,
not an actual Gaia observation sequence. The likelihood retains bundled
model covariance, but the mock does not draw model-error realizations.
Recovery does not validate real-data accuracy or posterior uncertainty.

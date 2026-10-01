# Connecting to orblet

sedkit supplies the stellar likelihood and light ratio; orblet supplies
orbital predictions and likelihoods. A shared parameter proposal supplies
M1, q, age, metallicity, parallax and orbital elements.
A joint fit composes
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

orblet's current `loglike_joint` assumes a dark companion. For a luminous
companion, compose the individual likelihoods with the corrected photocentre.

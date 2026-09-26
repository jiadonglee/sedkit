# Connecting to orblet

sedlet and orblet can be composed without importing one into the other.
A shared parameter set supplies M1, q, age, metallicity, parallax and the
orbital elements. `StellarModel.evaluate` returns the component masses,
absolute spectra, G magnitudes and `beta_g = F_G,2/F_G,1`.

For a primary-frame orbit, the signed photocentre scaling is
`a_phot/a1 = (q - beta_g) / (q * (1 + beta_g))`.
sedlet returns this as `a_phot_over_a1`; it agrees with orblet's
`signed_photocentre_axis_ratio(q, beta)` convention.

Use the true component masses and inclination for the RV model. Scale
the primary's sky-plane orbit by the signed photocentre factor before
projecting it along Gaia's scan direction. Equal-mass, equal-light stars
have zero photocentre motion even though their RVs can vary.

orblet's current `campbell_xy` and `loglike_joint` assume zero companion
light. A luminous-pair fit needs an externally composed forward model.
Combine the SED and orbit likelihoods for a shared parameter proposal.
Do not use the fitted SED posterior as a prior and then add its original
SED likelihood again. When using epoch astrometry, do not independently
reuse the catalogue astrometric solution derived from those same epochs.

This package supplies the stellar-side quantities; it does not fit orbits.

# Validation

Validated on 2026-09-26 with Python 3.11 in an isolated environment.

- Installed package runs without J-CAPS or JAX available.
- NumPy predictions agree with the source J-CAPS v2.1 implementation
  within 2e-6 relative flux on four supported stellar label sets (the
  current network is checked under [warm-network validation](validation-warm.md#warm-network)).
- Seven focused tests cover absolute component sums, the equal-light
  photocentre null, unsupported models, covariance likelihood versus a
  dense solve, mock recovery, unchanged observations, serialization and
  SPHEREx grid matching and Gaia coordinate-query field names.
- Live public acquisition retrieves 61 calibrated XP channels and five
  broadband points for Gaia DR3 1521154374020165376. The independent
  installation also calibrates the downloaded raw XP with GaiaXPy.

These are implementation checks and conditional examples. They do not
measure real-data binary completeness, false positives or mass accuracy.

## SPHEREx acquisition

On 2026-09-27, a live TallTable query and QR2 aperture extraction for
Gaia DR3 858860697467058688 produced 34 valid SPHEREx channels from 80
complete, unflagged apertures. The combined SED fits 98 channels, retaining
all XP and broadband measurements unchanged. Four focused tests check physical unit conversion against
Astropy, unchanged inputs, missing/mismatched channel handling, proper-motion
arguments, cache reuse and explicit refresh.

## Example notebooks

On 2026-10-08, the first three [example notebooks](../examples) executed top to bottom
with Python 3.11, the warm network and the 2MASS training scale; the
cached SEDs were rebuilt from the cached catalogue products. The SB2 fit
gives q=0.840 against the Gaia NSS RV ratio 0.868 (binary chi2/N 1.02).
The orbit notebook prefers the luminous solution for Gaia DR3
1916454200349735680 (objective difference 659) and the dark solution for
LP 769-9 (929).

The [giant validation](validation-giant.md) records the giant template, control
giants and companion injections.

The [warm-star validation](validation-warm.md) records the warm-network,
cluster, binary-mock and vertical-action checks in
[notebook 03](../examples/03_warm_binaries.ipynb).
The [hot-star validation](validation-hot.md) records the hot-route holdout,
CALSPEC and end-to-end anchor fits and the network seam.

## Extinction and parallax

On 2026-10-03, [notebook 05](../examples/05_extinction_parallax.ipynb)
ran top to bottom on Garching with Python 3.12 and the full Edenhofer
posterior-sample map. It compares fixed E/parallax, fitted E at fixed
parallax and jointly fitted E/parallax, showing parameters, absolute SEDs
and priors. The joint binary fit gives E=0.007298, q=0.924271 and a parallax
shift of 0.284 catalogue sigma. Observation arrays and shared masks are
unchanged; the objective includes the SED likelihood and both prior terms.
See [the real-map check](extinction-validation.md) for the installation,
map and scientific limits.

## Error-term blend

On 2026-10-03 the cold/warm model-error switch at a primary Teff of 4000 K
was replaced by a blend over 3800--4200 K. In mocks with primaries at
3600--4400 K (5 Gyr, [M/H] = 0, 50 pc, XP and JHKs, q = 0, 0.5, 0.7 and 0.9,
fitted at the true age; J-CAPS experiment `error_term_seam_20261003`), the
switch changed the single-minus-binary objective by up to a factor of 1.9
between primaries 10 K apart. It also left 27 of 240 noisy fits with a
solution at exactly 4000 K although the true primary was elsewhere. With the
blend the largest change between neighbours is a factor of 1.15 and no fit
sits at 4000 K. The objective still rises by a factor of three to four from 3800
to 4200 K, because the warm term is smaller. The objective differences in the first three notebooks
were obtained with the blend.

## PARSEC tables

On 2026-10-02, `scripts/build_parsec_tracks.py` rebuilt the tables from
CMD 3.8. At the four age nodes of the previous 0.5-dex tables, all 64
tables agree more than 0.05 solar masses below the turn-off within 0.001
dex in logTe, 0.004 dex in logL and 0.01 mag in Ks and G. Separate CMD
isochrones at the 26 intermediate ages from log age 8.725 to 9.975, at
[M/H] = -0.5, 0 and +0.3 (`data/stellar_model/parsec_fine_age/midpoint_check`
outside the repository), test the interpolation. For supported stars
below 1.5 solar masses, the 99th-percentile Teff difference is 5--15 K
(maximum 56 K), and Ks and G differ by at most 0.02 mag at the 99th
percentile. The previous tables differ from the same isochrones by up to
261 K, and by 0.11--0.16 mag in Ks and G at the 99th percentile.

## Gaia batch API

On 2026-09-30, all 22 package tests passed. Six batch tests cover exact
19-digit identifiers, VOTable arrays/units/masks, rejection of truncated
catalogues, incomplete-product detection, cached batches and resumed TAP
jobs. A live ARI TAP query followed by ESA DataLink downloads for Gaia DR3
30343944744320 and 1521154374020165376 returned two XP_CONTINUOUS spectra,
one RVS spectrum and one EPOCH_PHOTOMETRY product in 12.6 seconds overall.
A repeated call reused all three ZIP files. The unavailable RVS and epoch
photometry for the second source were reported separately. This two-source
check verifies DR3 acquisition and reuse, not bulk throughput or DR4 support.

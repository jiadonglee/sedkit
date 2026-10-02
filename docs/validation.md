# Validation

Validated on 2026-09-26 with Python 3.11 in an isolated environment.

- Installed package runs without J-CAPS or JAX available.
- NumPy predictions agree with the source J-CAPS v2.1 implementation
  within 2e-6 relative flux on four supported stellar label sets.
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

On 2026-10-02, both [example notebooks](../examples) executed top to bottom
with Python 3.11 and the 0.05-dex PARSEC tables. Repeated runs reuse the
cache. The SB2 fit gives q=0.855 against the Gaia NSS RV ratio 0.868. The
orbit notebook prefers the luminous solution for Gaia DR3
1916454200349735680 (objective difference 645) and the dark solution for
LP 769-9 (1118).

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

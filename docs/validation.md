# Validation

Validated on 2026-09-26 with Python 3.11 in an isolated environment.

- Installed package runs without J-CAPS or JAX available.
- NumPy predictions agree with the source J-CAPS v2.1 implementation
  within 2e-6 relative flux on four supported stellar label sets.
- Seven focused tests cover absolute component sums, the equal-light
  photocentre null, unsupported models, covariance likelihood versus a
  dense solve, mock recovery, unchanged observations, serialization and
  SPHEREx grid matching and Gaia coordinate-query field names.
- The reproducible 3%-noise mock injects M1=0.75, q=0.8 and recovers
  M1=0.751, q=0.796 at fixed 5 Gyr and solar metallicity.
- Live public acquisition retrieves 61 calibrated XP channels and five
  broadband points for Gaia DR3 1521154374020165376. The independent
  installation also calibrates the downloaded raw XP with GaiaXPy.
- The quickstart notebook executes all six code cells and displays its
  SED figure. The real-source example reaches the parallax constraint
  boundary under its fixed stellar assumptions; structured residuals
  remain visible.

These are implementation checks and conditional examples. They do not
measure real-data binary completeness, false positives or mass accuracy.

## Observed SB2 examples

On 2026-09-27, four public Gaia DR3 SB2 systems were fitted on unchanged
XP+JHKs observations. All four prefer the binary model, with differing
agreement between SED and RV mass ratios. See the [SB2 experiment](sb2.md)
for results, selection, reproducible inputs and limitations. Individual
and combined paper-style plots were inspected after export.

## SPHEREx acquisition

On 2026-09-27, a live TallTable query and QR2 aperture extraction for
Gaia DR3 858860697467058688 produced 34 valid SPHEREx channels from 80
complete, unflagged apertures. The combined SED fits 98 channels, retaining
all XP and broadband measurements unchanged. The [example](spherex.md)
includes native Jy spectra, all exposure measurements and the extraction
configuration. Four focused tests check physical unit conversion against
Astropy, unchanged inputs, missing/mismatched channel handling, proper-motion
arguments, cache reuse and explicit refresh.

## Joint orblet mock

On 2026-09-27, all 13 package tests passed, including the public
`loglike_sed` interface, unchanged measurements and absence of a duplicate
catalogue parallax constraint. The [joint mock](orblet.md) recovers a shared
five-parameter SED + SB2 RV + along-scan astrometry model from three starts.
The same observations give a strongly biased q when the photocentre light
correction is omitted. This conditional test fixes orbital shape, age,
metallicity and astrometric offsets; it does not establish real-data accuracy.

## Observed joint fit

On 2026-09-27, the [HD 195987 experiment](real-orblet.md) fits 61 observed
XP channels and 52 double-lined RV epochs. Two distinct starts recover
the same best conditional joint solution. Raw observations are unchanged,
the SED-only control is no worse than its joint SED-plus-distance term,
and the profiled RV amplitudes reproduce the independent RV-only fit.
Age reaches the support boundary and secondary RV residuals exceed the
published errors; parameter accuracy and uncertainty remain unvalidated.

## Group-meeting notebooks

On 2026-09-27, notebooks 01--03 of the [group meeting](group-meeting.md) executed
top-to-bottom with Python 3.11, network downloads disabled and the pinned
orblet revision. Seven figures are embedded in the saved outputs. The SED
checks preserve original measurements; the joint likelihood reproduces its
stored objective and the photocentre factor matches orblet. Scientific
limitations are carried into the notebook explanations.

On 2026-10-01, notebook 01 re-executed on the current package with unchanged
fit values, and notebook 04 executed offline from the two included orbit
snapshots. Its solutions and objective differences match the
[orbit example](orbit.md): luminous for the equal-mass binary (637),
dark for LP 769-9 (1118).

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

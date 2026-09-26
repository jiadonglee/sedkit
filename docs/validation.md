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

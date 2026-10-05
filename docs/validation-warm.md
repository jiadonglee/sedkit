# Warm-star validation

## Warm network

On 2026-10-02 the bundled network was replaced by the J-CAPS retraining
with IRFM Teff and 1 kpc dwarfs to 7500 K. NumPy predictions agree with
the J-CAPS implementation within 2.3e-6 relative flux on six label sets
from 3000 to 7400 K. Downloaded 2MASS fluxes of four Gaia DR3 sources
agree with the training data within 2e-6.

Single-star fits of Hyades, Praesepe and Coma Ber members with IRFM Teff
6000--8000 K, at the cluster age and [Fe/H] and with XP and 2MASS
dereddened by the literature E(B-V), compare with the previous network:

| IRFM Teff (K) | Stars | Isochrone mass | chi2/N previous | chi2/N warm | Fitted minus isochrone mass, warm |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6000--6500 | 31 | 1.22 | 2.4 | 0.8 | +0.001 |
| 6500--7000 | 26 | 1.39 | 566 | 1.2 | +0.003 |
| 7000--7500 | 15 | 1.70 | 3395 | 73 | -0.061 |
| 7500--8000 | 13 | 1.86 | 4589 | 502 | -0.204 |

With the age free (0.5--10 Gyr) at the cluster [Fe/H], chi2/N of the warm
network is 0.69, 0.67, 0.88 and 1.54 in the four bins (previous network
1.7, 116, 2394, 3894), and the fitted mass lies 0.016, 0.035, 0.056 and
0.061 solar masses below the isochrone mass. At the cluster age the
1.4--1.6 solar-mass fits leave a residual slope of about 8 per cent across
0.4--2.2 um that disappears when the age is free. The free fits prefer
ages older than the literature values: medians of 1.03, 1.85 and 2.0 Gyr
for Praesepe, the Hyades and Coma Ber against 0.70, 0.65 and 0.60 Gyr.
Their PARSEC Teff lies 7 and 51 K below IRFM at 6000--6500 and 6500--7000 K
and 135 K below at 7000--7500 K. The network reproduces the SED shapes;
the PARSEC isochrone at the literature cluster age is hotter than these
stars at their luminosity. An unresolved companion gives the same free-age
signature (see the [warm-primary notebook](../examples/03_warm_binaries.ipynb)), but freeing the age lowers chi2
by more than 20 per cent for 19 of 26 stars at 6500--7000 K and all 15 at
7000--7500 K, with fitted ages within 1.1--1.6 Gyr (interquartile): a
common offset rather than a binary subset. Below 7000 K the offset comes
from the Hyades and Coma Ber; for Praesepe, single-minus-binary objectives
at the cluster age stay below 2 for 22 of 24 main-sequence members and
reach 36--491 for members 0.2--0.8 mag above the sequence. Hyades
main-sequence members give 15--44 (q 0.42--0.56) also with [Fe/H] free,
so a known age needs a zero-point check on the same cluster's
main-sequence stars (J-CAPS experiment `warm_cluster_binary_20261003`).
At fixed age and parallax the mass follows the luminosity, so the mass
agreement mainly confirms a supported fit; chi2/N measures the SED shape.
Above 7000 K most fits reach the parallax bound: at these cluster ages the
isochrone Teff of 1.7 solar masses exceeds the 7500 K coverage. These are
consistency checks on three clusters, not an accuracy calibration.

## Warm-primary notebook

The warm-primary notebook fits 108 emulator mocks (primaries of 1.2, 1.4
and 1.55 solar masses at 1.2 Gyr, q 0--1, measurement noise plus a
model-covariance draw). With age and [M/H] fixed at the truth, q >= 0.5
companions give single-minus-binary objectives of 25--400 and median fitted
q within 0.03 of the truth. With both free, the objective stays below about
15 and fitted q scatters over 0.1--1: the single-star fit moves to an older,
brighter star. In a noise-free mock, a 1.4 solar-mass primary with a q = 0.7
companion at 1.2 Gyr is matched by a single 1.4 solar-mass star at 2.0 Gyr
(objective 6 above the binary); for a 0.8 solar-mass primary no single age
comes within 449. On 20 Gaia DR3 SB2s at BP-RP 0.3--0.5, chi2/N is 0.6--2.6,
fits at the RV mass ratio lie within 4 of the best free objective, and
three of four eclipsing systems have M1 sin^3 i 1.05--1.10 times the SED
primary mass (the fourth 0.80).
For 2210 Gaia DR3 warm stars within 100 pc with RVs and G >= 5
(`examples/warm_jz_100pc.csv`, from the J-CAPS experiment
`warm_jz_test_20261003`), the vertical action ranks better by the
single-star age than by the binary-solution age for stars without a
`non_single_star` flag (Spearman difference -0.08, 95 per cent interval
-0.13 to -0.03) and the reverse for the 179 flagged binaries (+0.18, 0.00
to +0.34).

## F-type cluster members and log g priors

In 494 Hunt & Reffert clusters (J-CAPS `warm_cluster_truth_large_20261005`;
3702 members at 1.1--1.4 solar masses, multiplicity from WDS, SB9, MSC,
Kervella et al. 2022, Gaia NSS and Gaia DR3 RV variability), fits with
`StellarModel(hot=True)` at [M/H] = +0.1 give, at a main-sequence age
fraction <= 0.4:

| Fit | Luminous companion, Delta > 25 | Gaia RV single, Delta > 25 | LR, Delta > 25 | LR, Delta <= 25 |
|---|---:|---:|---:|---:|
| Age free | 18 per cent | 6 per cent | 3.1 | 0.87 |
| Cluster age, RV-single ridge and shared width | 54 per cent | 20 per cent | 2.7 | 0.58 |

The fixed-age single-star sequence of the RV singles is 0.07--0.11 mag
wide, so three widths correspond to q >~ 0.75 for a 1.2 solar-mass primary.
Delta separates the classes no better than the luminosity excess over that
sequence (ROC AUC 0.68 against 0.72): XP and J/H/Ks do not resolve the
colour of the cooler companion. A second sequence 0.5--0.8 mag above the
ridge holds 15 per cent of the RV singles, likely long-period twins that
Gaia RV does not detect.

In free-age mocks of a 1.2 solar-mass primary with measurement noise only,
the single-star solution moves by -0.05, -0.13, -0.18 and -0.27 dex in
log g for q = 0.5, 0.7, 0.85 and 1. A `logg_prior` of 0.1 dex detects
nothing; 0.05 dex detects about half of the q >= 0.85 binaries; 0.02 dex,
the precision of asteroseismic log g, detects q >= 0.7. Survey log g values
derived from the parallax and an isochrone already contain the companion's
light and are not independent priors.

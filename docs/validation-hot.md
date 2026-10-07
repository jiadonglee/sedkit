# Hot-star validation

`StellarModel(hot=True)` uses the J-CAPS hot-emulator run
`hot_emulator_v5_20261004` above 7000 K ([model](model.md#hot-star-route)).
The numbers below come from that run and from fits with this package.

## Calibration and holdout

The calibration sample holds 232 hot anchors and 434 dwarfs; the operator
correction was fitted to 186 anchors and 348 dwarfs, and the remaining 46
anchors and 86 dwarfs form the holdout.
The anchors come from the hot anchor table: radial-velocity-constant stars
and single-lined binaries, without giants or supergiants, Be and peculiar
stars, and without stars whose Ks absolute magnitude contradicts a
main-sequence star at their spectroscopic Teff. The dwarfs are J-CAPS 1 kpc
IRFM dwarfs at 7000--7500 K with |[Fe/H]| <= 0.3, held to the conditions
of a sedkit fit: Teff to IRFM (20 K), log g to the PARSEC value of their
network fit (0.05) and E to the Edenhofer map (0.002). Each star was fitted
with Teff, log g, E and flux scale under these priors. The anchor holdout is
a random 20 per cent per Teff bin; the dwarf holdout is the J-CAPS
validation split.

| Teff (kK) | Holdout stars | Median rms, 0.4--1 um |
| --- | ---: | ---: |
| 7.0--7.5 (dwarfs) | 86 | 0.88% |
| 7.5--11.5 | 27 | 1.25% |
| 11.5--15 | 10 | 1.05% |
| 15--30 | 8 | 1.20% |

The same channel correction applied to 10 CALSPEC hot single stars, with
the Balmer index measured on their STIS spectra, lowers the median rms of
the XP operator from 4.7 to 1.8 per cent; the median absolute offset is
+0.7 per cent. No CALSPEC star is an anchor.

Out-of-fold shape residuals of the hot anchors over 0.4--1 um are
1.1--1.3 per cent above 9 kK and 1.2 per cent at 7.5--9 kK. The
model-error term, from these residuals, holds three eigen-directions with a
median width of 0.7 per cent over 0.4--1 um, 2.9--4.3 per cent at J/H/Ks,
and a common 0.9 per cent column for the absolute scale.

## End-to-end fits

All 46 holdout anchors were fitted with `fit(sed, "single",
model=StellarModel(hot=True), age_gyr=None, feh=0.0, extinction=E)`, with
E from the Edenhofer map and absolute fluxes at the Gaia parallax. The
table bins the 44 stars of the supported ranges by spectroscopic Teff:

| Teff (kK) | Stars | Median shape rms, 0.4--1 um | Median Teff minus spectroscopic Teff |
| --- | ---: | ---: | ---: |
| 7.5--11.5 | 26 | 1.32% | -193 K |
| 11.5--15 | 10 | 1.22% | -229 K |
| 15--30 | 8 | 1.38% | +50 K |

With mass and age free, the PARSEC radius scales the flux freely, so these
fits do not test the absolute scale; the CALSPEC offset above does.
Spectroscopic Teff above 15 kK mostly come from spectral types
(1.5--2.5 kK uncertainty); at 8--9 kK the seven holdout anchors are
composite spectral types fitted 0.8 kK below their spectral-type Teff.

## Network seam

The 86 dwarf holdout stars were fitted on XP and J/H/Ks with free mass and
age, their spectroscopic [Fe/H] and the map E, by the network alone and by
`hot=True` with the 7000--7498 K handover:

| IRFM Teff (K) | Stars | Shape rms, network | Shape rms, hot=True | Fitted minus IRFM Teff, network | Fitted minus IRFM Teff, hot=True |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7000--7250 | 61 | 1.16% | 1.06% | -16 K | +3 K |
| 7250--7400 | 21 | 1.42% | 1.02% | -48 K | +21 K |
| 7400--7500 | 4 | 1.55% | 1.01% | -67 K | +8 K |

Below 7000 K the two models give the same median Teff offsets (+50, +31
and +15 K at 6250--6500, 6500--6750 and 6750--7000 K, 481 stars). On the
86 dwarfs, the table alone, with free flux scale and measurement errors at
the PARSEC log g of their `hot=True` fits and the map E, gives Teff 4 K
below IRFM. Above the seam, holdout anchors exist only on spectral-type
Teff: one at 7.5--8 kK and seven composites at 8--9 kK.

At fixed IRFM Teff and log g 4.2, the table alone reproduces these stars
to 1.0--1.3 per cent and the handover to 1.1--1.3 per cent, against
1.2--1.9 per cent for the network. At the same PARSEC star at [M/H] = 0,
the table and the network differ by a common 2 per cent in level and
0.8 per cent in XP shape; at [M/H] = -0.3 and +0.3 the shape difference is
1.6 per cent.

## Binary injection-recovery

Matched mocks in J-CAPS `hot_binary_mock_20261004` inject coeval binaries
with hot primaries (2--8 solar masses, 30 and 200 Myr) at the anchors'
measurement errors, with and without a draw from the hot model-error term,
and fit them with `fit(kind="both")`. No single-star mock reaches
Delta > 25. With age fixed at the truth, unevolved 2--5 solar-mass
primaries with q >= 0.5 give Delta 50--1150 and q within 0.04 up to
q = 0.7; primaries near the end of the main sequence (3 solar masses at
200 Myr, 8 at 30 Myr) are detected only for some q. With age free,
median Delta is below 25 for every cell: a single star evolved along the
main sequence reproduces the companion's light, as for warm primaries.

The same mocks fitted with free age and an `age_prior` or `logg_prior`
centred one width off the truth (J-CAPS `hot_binary_prior_mock_20261004`)
show that a prior adds roughly (shift/sigma)**2 to Delta, where the shift is
how far the single-star solution moves to absorb the companion: 1.25, 0.8
and 0.3 dex in age for 2, 3 and 5 solar masses at 30 Myr, and 0.1--0.25 dex
in log g for q = 0.5--1. A cluster age to 0.1--0.2 dex restores detection
of q >= 0.5 for young 2--3 solar-mass primaries; 5 solar masses needs
0.05 dex. A spectroscopic log g at 0.10--0.15 dex hardly helps; 0.05 dex
detects most q >= 0.85 binaries with 2 solar-mass primaries.

On real data (J-CAPS `hot_cluster_binary_20261004`), 49 hot anchors that
are Hunt & Reffert (2024) cluster members were fitted with free age, a
Gaussian prior at the cluster age (0.10--0.35 dex wide) and the age fixed
at the cluster age. With free age or the prior, one of 44 valid stars
reaches Delta > 25 and none of the 12 composite-spectrum stars does. Free
single-star ages are 0.22 dex older than the cluster ages (median; 77 per
cent older), so at the fixed cluster age 23 of 43 stars, nine of them RV
constant, reach Delta > 25 with single-star chi2 six times the free-age
value: the binary absorbs the age offset, not a companion. Fixed cluster
ages are therefore not usable until the hot route and the cluster ages
agree.

The cluster members are about 0.13 mag brighter than the [M/H] = 0 PARSEC
sequence at the cluster age from G to B stars (J-CAPS
`hot_cluster_zams_20261004`). With the model at [M/H] = +0.1 the
fixed-age Delta drops (median 16 to 3 for 297 G-to-B members, 33 to 13
for the hot anchors), but eight of 14 RV-constant hot anchors still reach
Delta > 25 (J-CAPS `hot_cluster_binary_feh_20261004`). At fixed age,
Delta tracks the height above the cluster's own sequence (rank
correlation 0.87; Delta = 25 near +0.1 mag), so it separates binaries
only where single stars scatter by less than that: F and A members
(6500--10000 K), not pre-main-sequence G stars or B stars.

Against literature multiplicity (SIMBAD, WDS, SB9; J-CAPS
`hot_cluster_truth_20261005`), 61 B and early-A members of the same seven
clusters, fitted at fixed age and [M/H] = +0.1, reach Delta > 25 for five
of seven stars with a double-lined orbit or a WDS companion within 2
arcsec and 2.5 mag, against four of 36 with no known companion (Fisher
p = 0.002). A 5 per cent spectral-type `teff_prior` barely moves the fits:
eight late-B stars prefer solutions more than 20 per cent hotter than their
spectral type, and they form the faint tail of the height distribution.

In 698 Hunt & Reffert clusters (J-CAPS `hot_cluster_truth_large_20261005`;
3427 B/A members, multiplicity from WDS, SB9, MSC, Chini et al. 2012,
Kervella et al. 2022 and Gaia NSS), stars with a luminous companion reach
Delta > 25 more often than stars with none (65 against 44 per cent), but
Gaia DR3 RV-constant stars with no known companion also reach 43 per cent.
These false positives come from the width of the fixed-age single-star
sequence: its faint-side half-width grows from 0.08 mag near the ZAMS to
0.54 mag at 0.8 of the turnoff Teff, while its ridge sits at the F-member
zero point. Moving the parallax to the ridge and fitting it
(`fit_parallax=True`) with the ridge width as its error, one luminosity
nuisance shared by both hypotheses, gives:

| Turnoff ratio | Luminous companion | RV single | LR, Delta > 25 | LR, Delta <= 25 |
|---|---:|---:|---:|---:|
| <= 0.55 | 12/16 | 59/336 | 4.3 | 0.30 |
| > 0.55 | 7/31 | 56/618 | 2.5 | 0.85 |

For one hot star this means:
- with free age and photometry only, Delta does not tell a single star
  from a binary;
- for a near-ZAMS B/A cluster member (7--13 kK), Delta <= 25 makes a
  q >~ 0.65 companion unlikely (about 7--11 per cent for a 20--30 per
  cent prior), while Delta > 25 marks a candidate (about 50--65 per cent)
  that needs RV or imaging;
- for evolved members, Delta carries little information;
- the fitted q does not track the literature q.

Delta adds no information to the luminosity excess. Against the RV
singles, the luminous-companion stars give a ROC AUC of 0.78 for the
shared-width Delta and 0.84 for the offset above the ridge near the ZAMS,
and 0.64 for both in evolved stars. At a fixed excess, Delta is the same
for both classes. Companions with q >~ 0.5 to a hot primary are themselves
hot, so XP and J/H/Ks see their light but not their colour. The hot route
therefore serves single-star Teff, extinction and fixed-age luminosity,
and its Delta is not a binary test for an individual hot star.

Only 14 RV-constant stars are hotter than 13 kK, and O stars are untested.

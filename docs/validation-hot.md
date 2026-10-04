# Hot-star validation

`StellarModel(hot=True)` uses the J-CAPS hot-emulator run
`hot_emulator_v3_20261004` above 7000 K ([model](model.md#hot-star-route)).
The numbers below come from that run and from fits with this package.

## Calibration and holdout

The operator correction was calibrated on 232 hot anchors and 348 dwarfs.
The anchors come from the hot anchor table: radial-velocity-constant stars
and single-lined binaries, without giants or supergiants, Be and peculiar
stars, and without stars whose Ks absolute magnitude contradicts a
main-sequence star at their spectroscopic Teff. The dwarfs are J-CAPS 1 kpc
IRFM dwarfs at 7000--7500 K with |[Fe/H]| <= 0.3, with Teff held to IRFM
(20 K) and E to the Edenhofer map (0.01). Each star was fitted with Teff,
log g, E and flux scale under these priors. A random 20 per cent of the
anchors per Teff bin was held out; the 86 dwarfs of the J-CAPS validation
split form the dwarf holdout.

| Teff (kK) | Holdout stars | Median rms, 0.4--1 um |
| --- | ---: | ---: |
| 7.0--7.5 (dwarfs) | 86 | 0.84% |
| 7.5--11.5 | 27 | 1.27% |
| 11.5--15 | 10 | 1.05% |
| 15--30 | 8 | 1.19% |

The same channel correction applied to 10 CALSPEC hot single stars, with
the Balmer index measured on their STIS spectra, lowers the median rms of
the XP operator from 4.7 to 1.7 per cent; the median absolute offset is
+0.7 per cent. No CALSPEC star is an anchor.

Out-of-fold residuals of the hot anchors by Teff are 1.0--1.4 per cent
above 9 kK and 1.6 per cent at 7.5--9 kK. The model-error term, from these
residuals, holds three eigen-directions with a median width of 0.7 per
cent over 0.4--1 um, 2.8--4.3 per cent at J/H/Ks, and a common 0.9 per
cent column for the absolute scale.

## End-to-end fits

All 46 holdout anchors were fitted with `fit(sed, "single",
model=StellarModel(hot=True), age_gyr=None, feh=0.0, extinction=E)`, with
E from the Edenhofer map and absolute fluxes at the Gaia parallax. The
table bins the 45 stars of the supported ranges by spectroscopic Teff:

| Teff (kK) | Stars | Median shape rms, 0.4--1 um | Median Teff minus spectroscopic Teff |
| --- | ---: | ---: | ---: |
| 7.5--11.5 | 27 | 1.32% | -114 K |
| 11.5--15 | 10 | 1.20% | -170 K |
| 15--30 | 8 | 1.38% | +148 K |

With mass and age free, the PARSEC radius scales the flux freely, so these
fits do not test the absolute scale; the CALSPEC offset above does.
Spectroscopic Teff above 15 kK mostly come from spectral types
(1.5--2.5 kK uncertainty).

## Network seam

The 86 dwarf holdout stars were fitted on XP and J/H/Ks with free mass and
age, their spectroscopic [Fe/H] and the map E, by the network alone and by
`hot=True` with the 7000--7498 K handover:

| IRFM Teff (K) | Stars | Shape rms, network | Shape rms, hot=True | Fitted minus IRFM Teff, network | Fitted minus IRFM Teff, hot=True |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7000--7250 | 61 | 1.16% | 1.06% | -16 K | +6 K |
| 7250--7400 | 21 | 1.42% | 1.03% | -48 K | +92 K |
| 7400--7500 | 4 | 1.55% | 1.02% | -67 K | +109 K |

At fixed IRFM Teff and log g 4.2, the table alone reproduces these stars
to 1.7--1.8 per cent and the handover to at most the network's rms
(1.15--1.67 against 1.17--1.89 per cent).

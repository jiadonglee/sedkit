# Giant-route validation

`fit_giant_companion` and `GiantTemplate` ([giant route](model.md#giant-route))
were checked on reddened APOGEE control giants, on companions injected into
them, and on the giant of the earlier twin-template test. The build scripts
(APOGEE selection, XP from the Garching share, crossmatched photometry,
template grid) and the fits ran in `giant_grid_20261006`; the asset comes
from `scripts/build_giant_model.py`.

## Template

The training set holds 76210 APOGEE DR17 giants. Per-star E fitted against
SFD E(B-V) on 3000 giants at 4400--5100 K has slope 0.86 (Theil--Sen;
per-star scatter 0.026), the factor used to deredden them. The template's
fractional scatter, after measurement variance is removed, is 3.1--13 per
cent at 392 nm (5th--95th percentile over nodes, median 6.7) and 2.7--9.7
per cent at 442 nm (median 4.3); at 4700 K, log g 2.2 and [M/H] -0.4 it is
4.9, 3.5 and 0.6 per cent at 392, 422 and 642 nm. The 39 Ks-normalised
GSP-Spec twins of Gaia DR3 465986123215151616 scatter by 27, 21 and 15 per
cent at 392, 402 and 442 nm, and 10 per cent at 642 nm.

## Controls

The controls are 238 APOGEE DR17 giants outside the training cut: SFD
E(B-V) > 0.15, at least three visits with RV scatter below 0.5 km/s, G
10--13.8, Teff 3900--5500 K, log g 1--3.2 and [M/H] -1.6 to +0.3, with
RUWE < 1.4 and no Gaia non-single-star flag. Each was fitted with label
priors of LAMOST LASP width (150 K, 0.3, 0.2) whose means were drawn
around its APOGEE labels at those widths, over the default 1.5--15 solar
mass grid.

| Quantity | 5% | 50% | 95% | 99% |
| --- | ---: | ---: | ---: | ---: |
| Giant-alone chi2/N | 0.34 | 0.59 | 0.96 | 1.37 |
| Free tilt (tilt_sigma = 1) | -0.22 | 0.00 | 0.19 | 0.28 |
| Detection, tilt_sigma = 0.15 | 0 | 0 | 2.1 | 6.3 |

The free tilts have a robust width of 0.14, which sets the default
`tilt_sigma=0.15`. The largest detection among the controls is 8.8; the 45
controls with fitted E > 0.4 and tilt below 0.1 reach 6.0 at the 95th
percentile and 8.7 at most, and the 37 with E <= 0.4 and tilt above 0.1
reach 8.8.
Fitted labels lie within -82/+36 K, -0.17/+0.22 and -0.12/+0.08 dex of
APOGEE (16th--84th percentile).

## Injection-recovery

Main-sequence companions of 2, 3, 4, 5, 6 and 8 solar masses at 20 Myr,
with a draw from the hot-route model error, were added to each control
through its giant-alone extinction and tilt at its parallax, and fitted
with the default 10 Myr companion. q is the objective at the injected
mass above the profile minimum; each mass has 238 injections.

| M2 injected | Detection > 10 | Detection > 25 | Best grid mass within one step | q > 10 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 92.9% | 84.0% | 98.3% | 0 |
| 3 | 99.2% | 97.1% | 99.6% | 0 |
| 4 | 99.6% | 99.6% | 100% | 0 |
| 5 | 100% | 99.6% | 100% | 0 |
| 6 | 100% | 100% | 100% | 0 |
| 8 | 100% | 100% | 100% | 2.1% |

The threshold of 10 is set by the controls, above every one of their
detections. At that threshold a true companion is excluded in 0.35 per
cent of injections, all of them at 8 solar masses, where the 20 Myr
companion is evolved. The injections use the fitted hot table and PARSEC
tracks, perturbed only in age and by the modelled error, so this rate
covers those two mismatches and none of the effects listed under
Limitations. Detection depends on the
companion's share of the 0.40--0.45 micron flux: no injection with a share
below 10 per cent exceeds 10, 35 per cent at 10--20 per cent, 87 per cent at
20--30 per cent and all above 30 per cent. At threshold 10 the controls
exclude companions from 2 solar masses upward (median; at most 4; the
1.5 solar-mass grid point is a pre-main-sequence star at 10 Myr), the
lowest excluded mass supplying 27--55 per cent of their 0.40--0.45 micron
flux (16th--84th percentile). The limit scales with the giant's
luminosity, not with G or E.

## The twin-template giant

Gaia DR3 465986123215151616 (LASP 4895 K, log g 2.54, [M/H] -1.22) has
blue excess against 39 high-latitude GSP-Spec twins, with chi2 lower by 56
for an 8 solar-mass companion. Against the template it gives detection 42,
best mass 6 solar masses and objective 19 lower than the giant alone at 8
solar masses. The best fit leaves chi2/N = 1.47, above the controls' 99th
percentile, and moves a label 3.3 prior widths. Neither a giant alone nor
giant plus companion describes this star. With the tilt free
(`tilt_sigma=1`) the detection is 30.

## Ultraviolet check

Five LAMOST giants of a dark-companion watchlist show blue excess (detection
above 10); four have UV photometry. Both hypotheses were extended below
392 nm with CK04 (below 15 kK) or TLUSTY BSTAR2006 spectra, each scaled to
its component at 0.40--0.55 micron and reddened by F99 (R_V = 3.1) at
E(B-V) equal to the fitted E. For the 19 controls with GALEX AIS NUV errors
below 0.2 mag, observed minus giant-alone NUV is -0.49 to +0.80 (median
+0.26); one of the 64 controls with a GALEX match is detected in FUV.

| Target | Band | AB mag | Minus giant alone | Minus giant + best companion |
| --- | --- | ---: | ---: | ---: |
| J060220.03+352147.9 (2.5 solar masses) | FUV | 20.27 +- 0.14 | -12.5 | +1.7 |
| | NUV | 18.90 +- 0.06 | -2.5 | +0.8 |
| J034958.06+433416.6 (2) | FUV | 19.36 +- 0.12 | -9.9 | +1.8 |
| | NUV | 17.15 +- 0.02 | -2.0 | +0.7 |
| Gaia DR3 465986123215151616 (6) | NUV | 16.63 +- 0.03 | -2.2 | +1.7 |
| TOI-977 (7) | UVOT UVM2 | 17.28 +- 0.03 | -3.9 | +1.1 |

The TOI-977 magnitude is 5 arcsec aperture photometry of one 494 s Swift UVOT
exposure, corrected for coincidence loss (Poole et al. 2008) with the AB zero
point of Breeveld et al. (2011); the NUV of Gaia DR3 465986123215151616 lies at
the edge of its GALEX field. All four have a hot component that the label
priors do not produce: the giant photosphere lies 2--4 mag below the
NUV/UVM2 detections and 10--12 mag below the FUV ones. The best-fitting
companion is 0.7--1.8 mag too bright in the UV in every case. Either the
optical fit overestimates the companion's luminosity or the UV extinction is
steeper than F99 R_V = 3.1, which carries about 1 mag in FUV at these E. For
TOI-977, UVM2 matches 5 solar masses where the optical profile has its
minimum at 7 and allows 6--8.

## XP below 392 nm

A 71-channel version of the fit adds XP at 342--382 nm. The giant's blue
channels come from the same template build; channels where the template is
not positive or its scatter reaches 30 per cent carry no weight (1 per cent
of supported nodes). Below 392 nm the companion is its mean 392--432 nm flux
times an empirical ratio: dereddened XP at each blue channel over
392--432 nm, the median in 0.03 dex Teff bins of 2484 hot main-sequence
stars (literature, spectral-type or spectroscopic Teff, Edenhofer E < 0.15,
G 6--13, no Be, emission-line, peculiar, evolved or SB2 stars; 6.2--26 kK),
with the bin's robust scatter (4--22 per cent) as its error. Extinction below
392 nm continues the ZGR23 curve as a power law. Without the blue channels
the 71-channel fit reproduces the detections above.

The control detections reach 3.2, 8.8 and 12.4 at the 95th and 99th
percentiles and the maximum, against 2.1, 6.3 and 8.8 without the blue
channels; the threshold becomes 14. Injections, as above, at threshold 10
without and 14 with the blue channels:

| M2 injected | Detection > T, without | With | Median detection, without | With |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 92.9% | 90.3% | 88 | 95 |
| 3 | 99.2% | 99.2% | 286 | 321 |
| 4 | 99.6% | 99.6% | 612 | 775 |
| 5 | 100% | 100% | 1117 | 1900 |
| 6 | 100% | 100% | 1803 | 4130 |
| 8 | 100% | 100% | 3835 | 14269 |

A true companion is excluded at its mass in 2.1 and 1.3 per cent of the
8 solar-mass injections and in none below. Injection and fit share the hot-star
ratio, so these rates are optimistic for the blue channels. The blue-excess
targets keep their best masses: detection 36.5 to 38.1 for
J060220.03+352147.9, 25.5 to 24.5 for J034958.06+433416.6, 24.4 to 30.2 for
TOI-977, 14.6 to 15.9 for Gaia DR3 206657917727078400 and 42.1 to 40.1 for
Gaia DR3 465986123215151616. The A-type companions of the first two predict a
Balmer-jump dip at 342--382 nm that the data follow, and TOI-977's B star
the rise at 342 nm that is observed. With XP errors of 0.1--0.15 in ln F, the
blue channels confirm the companion's spectral type; at 2 solar masses the
higher control tail cancels their gain in sensitivity.

## Extinction against the dust map

The fitted E was compared with the integrated Edenhofer et al. (2023) map at
the parallax distance. The map reaches 1.25 kpc; beyond it, its value at
1.2 kpc is a lower limit. For the 20 controls inside the map, E_fit - E_map
is -0.018 in median (16th--84th percentile -0.051 to +0.029), none beyond
3 sigma with a 0.03 floor; the 218 beyond it lie 0.011 above the lower limit
in median. Of the targets inside the map, three agree within 0.04 and
J034958.06+433416.6 does not: 0.431 for the giant alone and 0.369 with its
companion, against 0.528 +- 0.005.

## Luminosity and parallax

By default `fit_giant_companion` leaves the giant's scale free, so its
luminosity is not tied to the parallax. From the fitted scale, the template
Ks and the parallax, the giant's M_Ks, luminosity (PARSEC bolometric
correction), radius and mass (from log g) follow. The keywords `luminosity=`
and `dust_prior=` add these terms ([giant route](model.md#giant-route)):

- **Parallax:** a fitted parameter shared by both stars, within 3 sigma of
  its input value with penalty z^2. The fits below pass, through
  `parallax=`, the Gaia DR3 value with the Lindegren et al. (2021) zero
  point (median -0.031 mas, 0.013 mas added in quadrature) or, where Gaia DR3
  has one, the parallax of the non-single-star solution.
- **`luminosity="parsec"`:** the density of M_Ks among PARSEC v1.2S giants
  (label >= 2, log g < 3.8) at the giant's labels.
  - Isochrones are at 0.05 dex in logAge from 7.5 and 0.1 dex in [M/H] over
    -1.0 to +0.5, with a 0.5 dex table outside that range.
  - Points are weighted by the IMF, the linear age width, and an
    age--metallicity factor (age / 1 Gyr)^(4 max(0, -[M/H])) that gives
    metal-poor isochrones old ages. The factor is fitted on template training
    giants with parallax/error > 10.
  - The kernel is 200 K in Teff (wide because of the PARSEC--APOGEE Teff
    offset), 0.12 in log g and 0.15 dex in [M/H].
  - The M_Ks histogram, smoothed by 0.15 mag, is mixed with a uniform floor so
    that any M_Ks PARSEC produces at the labels costs at most 6.
  - The implied mass is held in 0.2--10 solar masses.
- **`luminosity="massfree"`:** the implied mass held in 0.2--10 solar
  masses without the PARSEC density, which allows a stripped giant.
- **`dust_prior=`:** the Edenhofer et al. (2023) E at the trial parallax
  distance, Gaussian with width sqrt(std^2 + 0.04^2) within 1.25 kpc; beyond
  it, a one-sided wall below E(1.2 kpc) - 0.04.

Without these keywords the fit is the default one of the sections above.

At 5000 K, log g 2.0 and solar metallicity, PARSEC's mode is a 2.5 solar-mass
giant; giants of 5, 6 and 8 solar masses cost 1.75, 2.95 and 4.78. Observed
M_Ks minus the prior median, for the training giants and the controls:

| [M/H] | Training | Controls | Flat star-formation rate, training / controls |
| --- | ---: | ---: | ---: |
| -2.0 to -1.0 | 0.14--0.17 | 0.16 | 0.34--0.36 / 0.33--0.36 |
| -1.0 to -0.5 | 0.09 | 0.11 | 0.26 / 0.29 |
| -0.5 to 0 | 0.04 | -0.08 | 0.14 / 0.07 |

| Quantity | Default | parsec | parsec + dust | massfree + dust |
| --- | ---: | ---: | ---: | ---: |
| Control detection, 95th / 99th percentile / max | 2.1 / 6.3 / 8.8 | 2.2 / 5.5 / 8.9 | 2.2 / 5.6 / 8.9 | 2.1 / 6.7 / 9.7 |
| Threshold | 10 | 10 | 10 | 11 |
| Control implied mass, 16 / 50 / 84% (solar masses) | -- | 0.75 / 1.01 / 1.62 | 0.75 / 1.00 / 1.62 | 0.64 / 1.03 / 2.20 |
| Detection > threshold at 2 / 3 / 4--8 solar masses | 92.9 / 99.2 / 99.6--100% | -- | 97.1 / 99.6 / 100% | 97.1 / 99.6 / 100% |
| True companion excluded | 2.1% at 8 | -- | 0 | 0 |

Of the 238 controls, 26 lie within the map's 1.25 kpc, so the PARSEC prior
alone and with the dust prior give nearly the same controls; injections were
run with the dust prior. The control masses from APOGEE log g are 0.71, 0.97 and 1.55 solar masses at
the same percentiles. The luminosity penalty of the controls has median 0.5
and maximum 6.05; inside the map, E_fit - E_map is -0.012 in median
(16th--84th percentile -0.038 to +0.020). The corrected parallaxes are 3--20
per cent larger than DR3's, so the injected companions are 6--40 per cent
brighter than in the default-fit injections.

On the blue-excess targets (detection, and in brackets its -2 ln L part):

| Target | Default | parsec + dust | massfree + dust | Allowed masses |
| --- | ---: | ---: | ---: | ---: |
| J060220.03+352147.9 | 36.5 | 34.9 (32) | 35.9 (32) | 2.0--2.5 |
| J034958.06+433416.6 | 25.5 | 28.0 (22) | 23.7 (21) | 2.0 |
| TOI-977 | 24.4 | 20.4 (18) | 21.2 (18) | 4--7 |
| Gaia DR3 206657917727078400 | 14.6 | 17.3 (16) | 14.8 (13) | 2.0--2.5 |

For these four targets the packaged keywords give the same detection,
-2 ln L part, best mass, E, implied mass and dust penalty as the fits
above to 0.001, and luminosity penalties within 0.002 (the prior is stored
at float16). The dust prior was evaluated from the map values of those fits.

- **J034958.06+433416.6:** alone, the giant needs 0.48 solar masses
  (luminosity penalty 8.5); with the companion, 2.0. At its NSS distance the
  map gives E = 0.48, and the companion fit's E of 0.39 costs 4.8 against 1.3
  for the giant alone.
- **TOI-977:** the zero point moves its parallax from 0.108 to 0.132 mas, and
  the giant is a bright giant of 5.3--5.6 solar masses and 3700 solar
  luminosities, which the prior accepts. Its best companion pushes the parallax
  to +2.9 sigma.
- **Gaia DR3 465986123215151616:** the detection (21--23) comes from the label
  and luminosity priors, its -2 ln L part being -2.7 to 0; the giant alone
  implies 16 solar masses.
- **Label--parallax inconsistent:** the spectroscopic labels and parallaxes
  of Gaia DR3 349395937424834432 (luminosity penalty 9.6 at every M2),
  TYC 2825-1500-1 (7.3) and Gaia DR3 468681885208015104 (6.5) imply giants of
  11--12 solar masses.

## Limitations

- XP starts at 392 nm in sedkit, the hot table and the ZGR23 curve, so the
  Balmer jump (330--390 nm) is not used. The 71-channel test (XP below
  392 nm) leaves 2 solar-mass sensitivity unchanged.
- The companion is a non-rotating, solar-metallicity star at 10 Myr: on the
  main sequence from about 2 solar masses, pre-main-sequence below. Be
  disks, stripped-star accretors and emission are not modelled.
- A hot companion and a greyer extinction curve both raise the blue end;
  the tilt prior, from disk giants at E 0.1--1, carries that distinction.
- Detection and exclusion assume the label priors are right to their
  widths: a hotter or more metal-poor giant absorbs some companion light.
- The template covers APOGEE giants. Warm metal-poor giants (above about
  5500 K at [M/H] < -1.5) and stars above 6800 K lie at or beyond its edge.
- The controls are low-latitude APOGEE disk giants at G 10--13.8. Results
  are profile optima over a mass grid, not posterior samples.
- Against UV photometry, the best-fitting companion masses of detected
  excesses are biased high (see Ultraviolet check); detection is confirmed,
  the mass is not.
- By default the fit fixes the parallax and leaves the giant's luminosity
  free; `luminosity=` ties them (see Luminosity and parallax). Its PARSEC
  prior remains 0.1--0.17 mag bright at [M/H] < -0.5, and outside [M/H]
  -1.0 to +0.5 its isochrones are 0.5 dex apart in age.

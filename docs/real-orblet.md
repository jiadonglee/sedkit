# HD 195987: observed XP and SB2 velocities

The [example](../examples/real_orblet_20260927/run.py) fits 61 observed Gaia XP
channels and 52 SOPHIE epochs of both components using sedkit and orblet.
All published fluxes, velocities and measurement errors are unchanged.

## Results

On 2026-09-27, the joint fit gives M1=0.83414, M2=0.65603 solar masses,
q=0.786470, parallax=46.2698 mas, age=10 Gyr and [M/H]=-0.0271.
Two distinct initial points recover the same best objective to 1e-8.
Other starts retain local minima; all are recorded in the
[summary](../examples/real_orblet_20260927/summary.json).

The SED-only control gives q=0.78984 and M1=0.83340, using the same distance
constraint. It minimizes only the SED and distance terms, with the joint
solution included among its starting points. The RV-only fit gives q=0.78647
and P=57.32214 days. The joint inclination is 81.756 or 98.244 degrees;
RV and SED do not distinguish these mirror solutions.

Published comparison masses are 0.844 +/- 0.018 and 0.6650 +/- 0.0079
solar masses ([Torres et al. 2002](https://arxiv.org/abs/astro-ph/0205511));
the inclination quoted by BEBOP VI is 99.364 +/- 0.080 degrees.
Neither mass nor inclination is imposed in the fit.

[XP figure](../examples/real_orblet_20260927/sed.png) and
[RV figure](../examples/real_orblet_20260927/rv.png) show the joint solution
with unchanged measurement errors. PDF versions are included.
The blue end of XP retains structured residuals. RV-only residual RMS is
5.64 m/s for the primary and 25.73 m/s for the secondary; chi2=518.97 for
104 measurements and eight parameters. The secondary scatter exceeds the
published errors. No error rescaling or jitter is applied.

## Composition and distance

The RV-only orbit fits P, eccentricity, omega, periastron time, K1, K2 and
two component velocity offsets. Its orbital shape is then fixed for the
conditional joint test. Each proposed q profiles K1 and both offsets by
weighted linear least squares, with K2=K1/q and K1 limited by sin(i)<=1.
The true masses and profiled K1 determine sin(i). Individual `rv_loglike`
atoms use companion mass times sin(i), with the true total mass.
The SED uses `loglike_sed` and the same proposed parallax.

Gaia DR3 reports parallax=45.7485 +/- 0.2012 mas and RUWE=11.775.
This experiment uses the published orbital parallax 46.08 +/- 0.27 mas
from Torres et al. (2002) as one Gaussian distance constraint instead.
The original Gaia parallax remains stored in the SED snapshot and is
not added to the likelihood. Comparison masses and this distance share
the historical interferometric analysis, so their agreement is not a
fully independent validation. There is no epoch astrometry term.

## Reproduce and provenance

Install the [tested orblet revision](orblet.md#reproduce-the-test), then run:

```bash
python examples/real_orblet_20260927/run.py
```

The included SED snapshot and RV CSV make fitting offline. SIMBAD identifies
HD 195987 as Gaia DR3 2067948245320365184; its raw JSON is included.
Raw XP coefficients and Gaia metadata were queried from
[Heidelberg TAP](https://gaia.ari.uni-heidelberg.de/tap/sync) on 2026-09-27.
[prepare_xp.py](../examples/real_orblet_20260927/prepare_xp.py) calibrates those
coefficients with GaiaXPy on sedkit's unchanged wavelength grid.
This test uses XP only; no broadband or SPHEREx observations are included.

The 52 RV rows are table D5 of
[Sairam et al. 2024, BEBOP VI](https://doi.org/10.1093/mnras/stae2317),
reused with attribution under the article's CC BY licence.
[extract_rv.py](../examples/real_orblet_20260927/extract_rv.py) downloads the
published PDF and extracts all seven columns, including original errors
and published O-C values. It additionally requires requests and pypdf.
The fit uses BJD minus 2400000.5 with the corresponding periastron time;
these relative Keplerian times are not Gaia TCB epochs.

## Limitations and next test

This is a Gaussian, Newtonian, conditional fit, with no jitter, relativistic
RV terms, orbit-shape uncertainty, extinction or alpha-enhancement parameter.
The quoted RV q error in JSON is a formal curvature error under that noise
model; excess residual scatter prevents interpreting it as calibrated precision.
XP measurement correlations are omitted, while bundled model covariance
and its determinant are retained. Age reaches the model's 10 Gyr ceiling;
no posterior mass or inclination errors have been measured.

The next small test should free the orbital shape in the joint likelihood
and compare an explicit RV residual model, retaining the original errors.
Older and alpha-enhanced stellar models are needed before interpreting age
or metallicity for this thick-disc benchmark.

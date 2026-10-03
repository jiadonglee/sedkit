# Real-map extinction check

Checked on 2026-10-03 with the installed sedkit package, Python 3.12.9,
and dustmaps 1.0.14 on Garching. The complete Edenhofer main posterior-sample
map was downloaded from the official Zenodo record and loaded with
`integrated=True, load_samples=True`. Its size is 19,516,121,280 bytes.
All 33 package tests pass locally; an extracted wheel also loads the bundled
stellar assets and extinction curve.

The target is Gaia DR3 `858860697467058688`, the SB2 in
[example 01](../examples/01_sed_fit.ipynb), at a catalogue distance of 91.10 pc.
Both hypotheses use the same 64 XP+2MASS channels; W1/W2 are held out.
Age is fixed at 5 Gyr and [M/H] at zero. Both fits explicitly use
`fit_parallax=True`, retaining the catalogue Gaussian constraint. The stellar network and fractional model covariance
are the bundled PARSEC+J-CAPS v2.1 route.

- With E fixed to zero: binary q = 0.92839 and chi2 = 82.01.
- With the distance-dependent map prior: binary E = 0.007298,
  q = 0.92427 and chi2 = 75.75. At the fitted binary distance, the prior
  mean is 0.007219 and its sample standard deviation is 0.000516.
- The dust-constrained single fit has E = 0.008533 and chi2 = 975.62;
  the single-minus-binary penalized objective is 916.44.

All four fits converged. The zero-extinction single fit reaches the upper
parallax bound. Neither dust-constrained fit reaches a fitted-parameter bound.
Assertions confirm unchanged observation arrays and identical single/binary
masks; fitted components sum to the attenuated system flux.

The [runnable script](../examples/04_extinction_prior.py) produces
`summary.json`, `sed.npz`, `fits.npz`, `sed_comparison.png/.pdf` and
`dust_prior.png/.pdf`. The [saved result](../examples/results/extinction_20261003/summary.json)
and [SED comparison](../examples/results/extinction_20261003/sed_comparison.png)
retain this experiment. `dust_prior_sigma` is the map-prior width, not an
uncertainty on the fitted E or q.

## Interpretation and limits

This target has little extinction, so the prior changes q by only -0.0041.
The Gaia DR3 SB2 semi-amplitudes in example 01 give q_RV = 0.86792;
the SED value remains higher by 0.05635. Extinction does not remove that
conditional-model mismatch. The test establishes a working real-map fit,
not externally calibrated mass-ratio accuracy or independent dust validation.
The prior is a truncated-Gaussian approximation to the map samples and
the extinction-curve shape is fixed; see [method limits](extinction.md#limitations).

The server installation source is `/home/jdli/nexus/sedkit_extinction_20261003/package`,
with interpreter `../.venv312/bin/python`. The reusable map is
`/home/jdli/xiasangju/jdli/mpoor/data/dustmaps/edenhofer_2023/samples_healpix.fits`.

```bash
ssh astronode-garching
cd /home/jdli/nexus/sedkit_extinction_20261003/package
../.venv312/bin/python examples/04_extinction_prior.py --cache-dir examples/data --output-dir ../runs/new_example
```

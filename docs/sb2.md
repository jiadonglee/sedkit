# Four observed SB2 systems

Four Gaia DR3 systems with published double-lined spectroscopic orbits
provide a first comparison of SED and radial-velocity mass ratios.

The example uses XP and 2MASS J/H/Ks, with W1/W2 held out. It fits age,
metallicity and catalogue-constrained parallax. RV mass ratios are independent
comparison values, not constraints on the free SED fit.

| Gaia DR3 source_id | q from RV | q from SED | Binary chi2/N |
| --- | ---: | ---: | ---: |
| 3628057800913942016 | 0.806 | 1.000 | 2.21 |
| 858860697467058688 | 0.868 | 0.855 | 1.19 |
| 1352014168153436544 | 0.948 | 0.890 | 0.93 |
| 5690586201229305344 | 0.988 | 0.868 | 1.11 |

All four fits prefer the binary model. Agreement in q varies: the first
source retains structured residuals and reaches q=1. Fixing q to its RV
value worsens the objective by 138; for the other three, the changes are
0.7, 3.9 and 6.1. These differences are conditional on this stellar model
and likelihood, without a posterior uncertainty calculation.

![Four SB2 SED fits](../examples/sb2_20260927/overview.png)

## Reproduce

Install `sedkit[download]`, then run:

```bash
python examples/sb2_20260927/run.py
```

[run.py](../examples/sb2_20260927/run.py) uses the included observed SED
snapshots and NSS orbit table; no network request is needed while they
are present. Removing a source snapshot downloads that source again.
The script saves individual PNG/PDF figures, a combined figure,
[summary.csv](../examples/sb2_20260927/summary.csv) and full fit JSON files.

## Selection and provenance

Targets were selected before fitting from Gaia DR3 NSS SB2/SB2C sources
with continuous XP, G<13, Teff 4000--7000 K, parallax/error>10, RUWE<1.4,
and catalogue low extinction. The four examples additionally require
solution type SB2, 9<G<12, 0.8<BP-RP<1.8, |b|>20 deg and A_G<0.05 mag,
and span RV mass ratios approximately 0.8--1.0. This is a small illustrative
sample, without a selection-function correction.

[targets.csv](../examples/sb2_20260927/targets.csv) records the selection
fields. [nss.ecsv](../examples/sb2_20260927/nss.ecsv) was retrieved from the
public Gaia Archive on 2026-09-27 using:

```sql
SELECT source_id, nss_solution_type, semi_amplitude_primary,
       semi_amplitude_secondary, period
FROM gaiadr3.nss_two_body_orbit
WHERE source_id IN (3628057800913942016, 858860697467058688,
                    1352014168153436544, 5690586201229305344)
```

For a two-body SB2 orbit, M2/M1=K1/K2. We report
q_RV=min(K1/K2,K2/K1), matching the model convention q<=1.
[Data provenance](data.md) describes XP calibration and photometric masks.

## Limitations

The model assumes two coeval stars, no extinction and the bundled stellar
training coverage. The fits use diagonal XP measurement errors plus model
covariance. They do not include RV orbit uncertainties or posterior errors
on q. Age is bounded at 0.5--10 Gyr and metallicity at -1--0.5 dex;
the second and fourth sources reach age boundaries. The first reaches
q=1. chi2/N uses model covariance and is not a reduced chi-squared with
fitted degrees of freedom removed. Binary preference is not a calibrated
binary probability or a validation of mass accuracy.

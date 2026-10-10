# White-dwarf validation

The DA route targets a companion's Gaia G light fraction
`beta_G = F_WD / (F_primary + F_WD)`. The initial scope is DA atmospheres,
Teff 6000--80000 K and log g 7.0--9.5, with thick-hydrogen-layer,
carbon/oxygen-core cooling sequences from Bédard et al. (2020).

## Anchor census

The 2026-10-10 census joins published EDR3 identifiers directly to
`gaiadr3.gaia_source.has_xp_continuous`. For Kepler et al. (2021), the
plate/MJD/fibre key is joined to the proper-motion-aware Gaia--SDSS
crossmatch of Gentile Fusillo et al. (2021). EDR3 and DR3 share source IDs.
No G-magnitude cut substitutes for the published XP flag: DR3 includes
an additional faint white-dwarf selection.

| Reference | Parameter rows | With XP | Pure DA in the model box |
| --- | ---: | ---: | ---: |
| Manser et al. (2024), DESI DA fits | 1958 | 1043 | 974 |
| Kepler et al. (2021), SDSS DR16 | 555 | 425 | 416 |
| Gianninas et al. (2011), via MWDD | 1264 | 1240 | 1210 |
| Kilic et al. (2025), via MWDD | 3146 | 3060 | photometric labels |
| Manser et al. (2024), DESI DB fits | 141 | 65 | outside the DA route |

The spectroscopic DA union contains **2524 distinct Gaia sources**.
Of these, **2064** have Teff >= 13000 K or the catalogue's 3D-corrected
labels. **2005** also satisfy RUWE < 1.4 and parallax/error > 10.
These are calibration candidates; usable spectral quality and residuals
have not yet been measured. Duplicate label measurements are retained in
the matched catalogue, while every fold must group by Gaia source ID.

Kilic et al. (2025) classify the spectra but obtain atmospheric parameters
from SDSS/Pan-STARRS photometry and Gaia parallaxes. Their labels are used
for coverage and comparisons, not independent spectroscopic calibration.
The SDSS DR16 fits also use a parallax constraint and explicitly omit 3D
corrections; their gravity/mass comparison is not independent of Gaia.
Cold 1D labels require correction before parameter validation.

The scripts and counts are in `reports/wd_plan_20261010/`; raw catalogues,
Gaia responses and the candidate list are in `data/whitedwarf/anchors/`.
Three candidates fainter than G = 17.65 have had XP_CONTINUOUS downloaded
successfully through DataLink.

## Criteria fixed before fitting

The census exceeds 200 sources, so a WD-specific wavelength and Balmer
correction can be tested with five folds grouped by source ID. The first
fit uses the uncorrected DA table. Residuals versus Teff and the WD's own
Balmer equivalent width decide whether a constant, low-order Teff term,
or `exp(a + W b)` is justified. The hot-star/sdB correction and its Balmer
normalization are not inherited.

| Test | Criterion |
| --- | --- |
| Single DA, held-out labels | absolute median fractional Teff offset <= 3%; robust scatter <= 8%; robust mass scatter <= 0.08 solar masses, reported as conditional on the adopted cooling model |
| Single DA and RV-constant FGK controls | one-sided 95% binomial upper bound on false positives <= 2%, separately in both samples |
| Dark-companion orbits | compare SED beta_G limits with RV/astrometric light-ratio constraints; the permitted beta_G range changes the inferred WD mass by <= 0.02 solar masses |
| WD+MS, subsequent calibration | WD Teff scatter <= 15%; companion subtype within one subclass; absolute beta_G offset <= 0.05 where an independent decomposition is available |

Robust scatter means `1.4826 * median(abs(residual - median(residual)))`.
At least 149 independent controls with zero false positives are needed
for a 2% one-sided 95% upper bound. A nonzero false-positive count requires
more controls. The detection threshold is determined on WD injections
and FGK controls; a likelihood threshold is not a calibrated confidence
level by itself.

GALEX on/off and XP 332--382 nm on/off are compared on the same held-out
stars. Report offsets and scatter by temperature and magnitude, as well
as the combined sample. Freeze the choice using the training folds before
evaluating the held-out fold. Parameter fitting and beta_G limits have
not yet been run.

## Limitations

- Candidate counts measure data availability, not fitting precision or
  selection completeness. MWDD currently contributes the Gianninas
  spectroscopic subset, not every literature measurement in the database.
- A pure DA catalogue classification does not rule out an unseen
  companion. Known double-degenerate flags are excluded across the joined
  catalogues; residual screening still needs to be done.
- The inspected Koester spectrum covers about 90--3000 nm. W1/W2 and
  SPHEREx channels beyond that range need another spectral source or an
  explicitly validated extension before use.
- The cooling relation assumes a carbon/oxygen core and thick hydrogen
  layer. It is not a validation of helium-core or extremely low-mass WDs.
  UV systematics, extinction and spectroscopic temperature scales remain
  to be tested on real data.

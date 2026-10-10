# Data acquisition and provenance

`download` queries public `gaiadr3.gaia_source`, retrieves XP_CONTINUOUS
through Gaia DataLink, and calibrates it with GaiaXPy on the bundled
61-point grid, 392--992 nm. GaiaXPy fluxes/errors in W m^-2 nm^-1 are
multiplied by 1e18 to match the model units. Values are otherwise unchanged.

2MASS and AllWISE matches use Gaia DR3's published best-neighbour tables,
including the 2MASS PSC/XSC join. Source IDs are retained as exact strings.
Catalogue magnitudes, errors and quality/match flags are stored in metadata
and the cached raw tables. A band enters the quality mask only for a
unique, point-source match with A-quality photometry and valid errors.
WISE contamination flags must be zero. Upper limits remain in the raw
tables and are excluded from the Gaussian flux fit.

Vega magnitudes are converted to Jy, then to the model's catalogue-equivalent
F_lambda convention at nominal wavelengths. Zero points are
1594, 1024, 666.7 Jy for J/H/Ks and 309.540, 171.787 Jy for W1/W2.
The model's nominal wavelengths are 1.235, 1.662, 2.159, 3.35, 4.6 µm.
These are broadband measurements, not monochromatic samples.

Gaia G/BP/RP magnitudes are metadata, not additional fit channels. This
avoids treating photometry overlapping XP as independent measurements.
W1/W2 are downloaded and plotted but held out by default.

Each `cache_dir/source_id/` contains raw Gaia, 2MASS and AllWISE ECSV
tables, raw XP XML, calibrated XP ECSV and `sed.npz`. Existing products
are reused. `refresh=True` explicitly replaces the source's cached products.
The cache contains public measurements, not fitted or corrected fluxes.
The orbit tests read two such `SED` snapshots from `tests/fixtures/`.

SPHEREx spectra can be [downloaded and extracted](spherex.md) through
TallTable/SPExPI, or imported from XphereX CSVs. The downloader retains
per-exposure aperture flags and combines only complete, unflagged apertures.
Jy fluxes/errors are converted to physical F_lambda units; missing channels
remain masked on the 102-channel model grid. Supplied quality masks are
retained; G<9 targets exclude that
segment during fitting because of observed bright-source bias.
The optional extraction worker has its own isolated dependencies.

`query_gaia` and `download_gaia` provide a separate [batch acquisition
route](gaia.md). They retain the original catalogue VOTable and DataLink
FITS ZIPs, including native units and covariance arrays. Batch products are
not converted into calibrated `SED` objects automatically.

## Sources

- [Gaia DR3 archive and acknowledgements](https://gea.esac.esa.int/archive/)
- [GaiaXPy](https://gaia-dpci.github.io/GaiaXPy-website/)
- [2MASS photometric system](https://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec6_4a.html)
- [WISE photometric system](https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html)
- [PARSEC](http://stev.oapd.inaf.it/cgi-bin/cmd)
- [J-CAPS source model](https://github.com/jiadonglee/J-Caps)
- [ZGR23 extinction curve](https://doi.org/10.5281/zenodo.7811871): the bundled
  `models/extinction_curve.txt` tabulates optical depth per native E.

The compact network, normalization, calibration and model-error assets
come from J-CAPS v2.1 / its stellar v2.2 fitting route. The PARSEC tables
are built by `scripts/build_parsec_tracks.py` from CMD 3.8: PARSEC v1.2S,
YBC bolometric corrections, Gaia DR2 (Evans et al. 2018) + Tycho2 + 2MASS
Vega magnitudes, no extinction, labels 0 and 1, requested per [M/H] as
log age 6.60--8.45 and 8.50--10.00. Their columns are mass, logTe, logL, Ks
and G; they are stellar evolutionary models, not observations.

The hot-star assets in `models/hot/` are built by
`scripts/build_hot_model.py` from the J-CAPS run `hot_emulator_v5_20261004`:
the channel table on Teff 7000--30000 K by log g 3.0--4.75 (CK04 and TLUSTY
BSTAR2006 through the XP forward model), the per-channel correction with
its Balmer index and cool-edge weight, and the `hot_v5` model-error term. `summary.json` records
the support box, provenance and validation numbers.

The giant-route assets in `models/giant/` come from the run
`giant_grid_20261006`. `scripts/build_giant_model.py` builds the empirical
template `giant_grid.npz` from APOGEE DR17 giants with Gaia XP, 2MASS and
AllWISE. `scripts/build_giant_prior.py` builds `parsec_prior.npz` on the same
grid: the M_Ks density of PARSEC v1.2S giants near each node, with an
age--metallicity weight fitted to template training giants with precise
parallaxes, and the PARSEC bolometric correction BC_Ks. `summary.json`
records both builds.

### Hot-subdwarf atmospheres

A subdwarf table needs Teff 20--45 kK, log g 5--6.5, a helium axis and
300 nm--5 um, from NLTE or line-blanketed atmospheres. The public grids:

| Grid | Teff (kK) | log g | Helium | Wavelength | Use here |
| --- | --- | --- | --- | --- | --- |
| TMAP via TheoSSA (GAVO), pure H | 20--46 in 1 kK | 4.9--6.6 in 0.01 | none | 0.1 nm--40 um | H tier |
| TMAP via TheoSSA, H+He+C (`HHeC`) | 30--46 in 0.5 kK | 4.9--6.6, irregular | log(He/H) -1.9 to +0.2, 9 values | 115--178 nm and 300 nm--5.5 um | mid and He tiers |
| TMAP via TheoSSA, H+He (`HHe`) | 30--46 | 4.9--6.2 | mostly log(He/H) = -3.1 | 0.5 nm--5.5 um | not used: one He value |
| TMAP grids on the SVO service | 20--150 | 4--9 | pure H, or sparse H+He | to 5.5 um or 40 um | duplicates of the above |
| TLUSTY BSTAR2006 / OSTAR2002 | 15--55 | to 4.75 | solar | UV--IR | below sdB gravities |
| Husfeld et al. NLTE He-rich | 35--80 | 4--7 | He-rich | optical | starts at 35 kK |
| CK04 / ATLAS9 | to 50 | to 5.0 | solar, LTE | UV--IR | LTE, log g edge at 5.0 |
| Koester, Levenhagen (2017) | WD range | 7--9.5 | | | white-dwarf gravities |
| Published sdB grids (XTgrid, Nemeth et al.) | | | | | spectra fitted per star, no public grid |

The table uses the Tuebingen NLTE Model-Atmosphere Package (TMAP; Werner et
al. 2003; Rauch & Deetjen 2003) spectra served by TheoSSA (Ringat & Rauch,
GAVO). TheoSSA holds precomputed H+He models only from 30 kK; below that
it holds pure hydrogen. The table therefore has three tiers: pure hydrogen
for He-poor subdwarfs at 20--45 kK, and H+He+C with log(He/H) = -1.9 and
-0.1 at 32--45 kK (the 30--31.5 kK rows of `HHeC` are incomplete in log g).
A He-rich subdwarf below 32 kK is outside the table.
`scripts/build_subdwarf_model.py` selects one model per table node from the
TheoSSA catalogue (nearest log g within 0.05 dex, the latest computation),
downloads it from the Tuebingen archive, and reduces it through the hot-table
XP forward model (`select`, `fetch`, `table` stages). TMAP tabulates F_lambda
/ pi; the build multiplies by pi and verifies the bolometric flux against
sigma Teff^4 (0.9 per cent at 30 kK, log g 5.6). The `HHeC` spectra have no
flux at 178--300 nm, so the He-rich tiers carry no GALEX NUV and no coarse
spectrum there. `summary.json` lists each tier's helium abundance, model
count and largest log g step bridged by interpolation.

GALEX FUV and NUV come from the GR6/7 AIS catalogue (Bianchi et al. 2017,
VizieR II/335). A band enters a fit only below the 10 per cent local
count-rate roll-off of Morrissey et al. (2007, Table 1), 114 and 303 counts
per second, AB 13.68 (FUV) and 13.88 (NUV), with artifact flag at most 1;
the calibration uncertainty, 0.05 and 0.03 mag, is added in quadrature. XP
below 392 nm is calibrated by GaiaXPy on 332--382 nm from the cached
continuous spectrum.

2MASS fluxes follow the J-CAPS training convention: catalogue zero points at
the model wavelengths times 0.98523, 0.98566 and 0.99105 for J, H and Ks.
The stellar network anchors PARSEC M_Ks on the same Ks scale, and so does
the giant route's luminosity.
AllWISE W1/W2 are converted at the model wavelengths; the training data use
unWISE fluxes, which lie about 5 per cent lower for the example sources, so
W1/W2 stay out of the default fit. The source code license is retained in LICENSE.

### White-dwarf atmospheres and cooling

The [DA anchor census and validation criteria](validation-whitedwarf.md)
use the spectroscopic DESI EDR fits of Manser et al. (2024), SDSS DR16
fits of Kepler et al. (2021), and the Gianninas et al. (2011) subset of
the Montreal White Dwarf Database. Kilic et al. (2025) supplies spectral
classifications and photometric parameters for coverage comparisons.
Catalogue EDR3 IDs are joined to Gaia DR3; SDSS plate/MJD/fibre IDs are
resolved using the Gentile Fusillo et al. (2021) Gaia--SDSS crossmatch.

The SVO `koester2` grid supplies pure-hydrogen LTE DA spectra at air
wavelengths, with surface flux `4 pi H_lambda` in erg s^-1 cm^-2 A^-1.
Selection over Teff 6000--80000 K and log g 7.0--9.5 yields 858 nodes
(78 temperatures, 11 gravities). The inspected 10000 K, log g 8 spectrum
covers 89.923--2999.1793 nm and integrates to 0.99380 times sigma Teff^4
over that finite range. This checks the surface-flux scale; it does not
validate XP agreement. The long infrared channels are outside this
spectrum's support.

The Bédard et al. (2020) thick-hydrogen cooling sequences tabulate radius
in cm and age in years. At 10000 K and 0.6 solar masses, interpolation in
the inspected sequence gives R = 0.012830 solar radii and age = 0.633 Gyr;
the tabulated log g agrees with G M / R^2. The atmosphere table and public
WD fitting API are still under development.

Sources: [SVO Koester grid](https://svo2.cab.inta-csic.es/theory/newov2/index.php?models=koester2),
[Bédard cooling sequences](https://www.astro.umontreal.ca/~bergeron/CoolingModels/),
[DESI catalogue](https://zenodo.org/records/13684288),
[SDSS DR16 catalogue](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/507/4646),
[MWDD](https://www.montrealwhitedwarfdatabase.org/tables-and-charts.html).

The complete table supports 123 of the 168 channels and has no replaced
outlier nodes. `scripts/build_whitedwarf_model.py` builds the atmosphere
and thick/thin-H cooling tables. The bundled XP correction uses 297 local
DA anchors; raw catalogues and validation spectra stay in `data/whitedwarf/`.
The UV passband correction uses a separate 71-source DA subset.
See [WD fitting](whitedwarf.md) for model and passband conventions.

The orbit validation uses [Yamaguchi et al. (2024)](https://arxiv.org/html/2405.06020v1)
Tables 1, 4 and 5 and Gaia DR3 NSS Thiele–Innes elements. WDMS spectral
parameters and angular-normalization distances use the
[author SQL catalogue](https://sdsswdms.upc.edu/query.php), matched by
plate/MJD/fibre. M-subtype comparisons use the
[Pecaut/Mamajek dwarf scale](https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt).

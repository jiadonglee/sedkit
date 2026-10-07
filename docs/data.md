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

2MASS fluxes follow the J-CAPS training convention: catalogue zero points at
the model wavelengths times 0.98523, 0.98566 and 0.99105 for J, H and Ks.
AllWISE W1/W2 are converted at the model wavelengths; the training data use
unWISE fluxes, which lie about 5 per cent lower for the example sources, so
W1/W2 stay out of the default fit. The source code license is retained in LICENSE.

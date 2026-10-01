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

The compact network, normalization, calibration and model-error assets
come from J-CAPS v2.1 / its stellar v2.2 fitting route. PARSEC subset
columns are mass, logTe, logL, Ks and G; they are stellar evolutionary
models, not observations. The source code license is retained in LICENSE.

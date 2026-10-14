# Data model, units and channels

[中文](../zh/data.md) · [GitHub](https://github.com/jiadonglee/spbin)


`SED` stores measured fluxes, errors, valid-channel masks, parallax and catalogue metadata. Measurements are at the actual source distance. Fitting preserves their normalization and measurement errors.

| Quantity | Unit or convention |
| --- | --- |
| `flux`, `error` | `1e-18 W m^-2 nm^-1`, channel-equivalent F_lambda |
| Wavelength | μm; XP uses 392–992 nm at 10 nm spacing |
| `parallax_mas` | mas |
| `m1`, `m2`, `mass` | Solar masses |
| `radius` | Solar radii |
| `teff` | K |
| `logg` | log10(g / cm s^-2) |
| `age_gyr`, `cooling_age_gyr` | Gyr |
| `feh` | Stellar-model [M/H], dex |
| `extinction` | Native ZGR23 E |
| `q` | M2/M1 |

Arrays contain 168 channels in the following zero-based Python slices:

| Slice | Channels |
| --- | --- |
| `0:61` | Gaia XP, 61 points |
| `61:64` | 2MASS J/H/Ks |
| `64:66` | AllWISE W1/W2 |
| `66:168` | SPHEREx, 102 points |

Gaia G/BP/RP are metadata rather than additional fitted channels. J/H/Ks/W1/W2 are broadband measurements represented as catalogue-equivalent fluxes.

```python
import numpy as np
from spbin import SED, StellarModel

model = StellarModel()
prediction = model.evaluate(m1=0.75, q=0.8, age_gyr=5.0, feh=0.0)
flux = prediction["flux_10pc"] * (20.0 / 100.0)**2
error = np.maximum(np.abs(flux) * 0.03, 1e-6)

mock = SED(
    flux=flux,
    error=error,
    mask=np.isfinite(flux),
    parallax_mas=20.0,
    parallax_error_mas=0.1,
    source_id="mock",
)
mock.save("data/mock.npz")
restored = SED.load("data/mock.npz")
```

This creates a noiseless model demonstration. For observed data, provide measured values, uncertainties and quality masks. Missing channels should contain NaN and be masked out.


## Acquisition APIs

### `download`

Retrieve a source by exact ID or search by ICRS coordinates:

```python
sed = download("858860697467058688", cache_dir="data")
sed = download(ra=120.0, dec=30.0, radius_arcsec=2.0, cache_dir="data")
```

Returns `SED`. The source cache stores catalogue tables, native XP, calibrated XP and `sed.npz`. Existing products are reused; `refresh=True` refreshes source data. Photometric selection requires unique point-source matches and A-quality bands, with zero WISE contamination flags. Downloaded parallaxes are the original Gaia DR3 values.

### `query_gaia` and `download_gaia`

```python
from spbin import query_gaia, download_gaia

catalogue = query_gaia(
    "SELECT TOP 20 source_id, parallax, phot_g_mean_mag "
    "FROM gaiadr3.gaia_source "
    "WHERE parallax > 10 AND has_xp_continuous = 'true'",
    cache_dir="data/query_nearby",
    tap="esa",
)
batches = download_gaia(
    catalogue["source_id"],
    products=["XP_CONTINUOUS", "RVS"],
    cache_dir="data/products_nearby",
    batch_size=100,
    workers=2,
)
for batch in batches:
    print(batch["product"], batch["path"], batch["unavailable_source_ids"])
```

`query_gaia` returns an Astropy Table with integers, units, array columns and masks retained. `tap` accepts `esa`, `ari` or a complete TAP URL. A reported server overflow raises an error.

`download_gaia` returns batch/product records containing `product`, `source_ids`, `unavailable_source_ids`, `path` and `cached`. `path` is a Path to the original FITS ZIP, or None when no source has the product. Batch acquisition does not automatically produce calibrated `SED` objects.

IDs must be integers or decimal strings; floating-point IDs are rejected. Reuse a directory to resume the same request. Change directories when changing a query, source ordering, products, release, batch size or data structure. See [Gaia acquisition](../gaia.md) for complete cache semantics.

### SPHEREx

```python
from spbin import download_spherex, load_spherex

sed = download_spherex(sed, cache_dir="data", radius_arcmin=0.8)
# Alternatively, import an existing XphereX PSF extraction:
sed = load_spherex(sed, "psf_spectrum.csv", method="psf")
```

The downloader performs QR2 aperture extraction. Its first invocation prepares Python 3.12 and isolated dependencies through uv. Imported CSVs require `wavelength_um`, `flux_jy` and `error_jy`; subsets of the model channels are accepted and Jy values are converted to `SED` units.

`sed.with_spherex(wavelength_um, flux, error, valid=None)` accepts the complete 102-channel grid and arrays already in `SED` units, returning a new object. Both attachment and CSV import assign channels without interpolation. See [SPHEREx acquisition](../spherex.md).


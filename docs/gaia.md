# Gaia batch downloads

Install `sedkit[download]`. Two APIs cover catalogue selection and native
DataLink products. The existing `download(source_id)` still returns one
calibrated `SED`; `download_gaia` returns raw product files for catalogue-scale
work, including joint XP/RVS/astrometry analysis.

```python
from sedkit import query_gaia, download_gaia

catalogue = query_gaia("""
    SELECT TOP 100 source_id, parallax, phot_g_mean_mag, bp_rp,
           has_xp_continuous, has_rvs, has_epoch_photometry
    FROM gaiadr3.gaia_source
    WHERE parallax >= 3.3333333333333335
      AND phot_g_mean_mag < 19 AND bp_rp BETWEEN 0.35 AND 4.5
""", tap="ari", cache_dir="data/nearby100/catalogue")

batches = download_gaia(
    catalogue["source_id"],
    products=["XP_CONTINUOUS", "RVS", "EPOCH_PHOTOMETRY"],
    cache_dir="data/nearby100/products",
    batch_size=100, workers=2,
)
for batch in batches:
    print(batch["product"], len(batch["source_ids"]), batch["path"])
    print("Unavailable:", batch["unavailable_source_ids"])
```

This is a small query example, not a complete FGKM selection. The parallax
cut is an observed-parallax approximation to 300 pc. The magnitude and colour
cuts affect completeness; no RUWE or binary-solution cut is imposed. Remove
`TOP 100` for the parent catalogue and choose a sufficient `maxrec` or split
large queries. Do not require every source to have every product when building
the parent sample.

## Catalogue API

`query_gaia(query, *, cache_dir="data/gaia-query", tap="esa",
maxrec=1000000, wait=600, timeout=(30, 180))` returns an Astropy `Table`.

- `tap`: `"esa"`, `"ari"`, or a full TAP service URL.
- `maxrec`: server result-row limit. A reported `OVERFLOW` raises an error;
  a truncated result is not cached as complete.
- `wait`: seconds spent polling, excluding individual network requests.
  A timeout retains the server job URL. Repeat the call to resume.
- `timeout`: connect/read inactivity timeouts for the result download.
  PyVO manages the TAP job-control requests separately.

Column names are lowercased. Integers, array columns, units and masks are
retained through Astropy. The original response is saved as `catalogue.vot`;
`job.json` records the asynchronous job URL. `request.json` ties the directory
to the exact ADQL, endpoint and row limit. Repeated calls reuse the local
result. Use a new directory when changing the query or when refreshing a
catalogue. Public server jobs can expire before a delayed resume.

## Product API

`download_gaia(source_ids, *, products=("XP_CONTINUOUS",),
cache_dir="data/gaia-products", data_release="Gaia DR3", batch_size=100,
workers=2, data_structure="RAW", datalink_url=..., timeout=(30, 300))`
returns a list of dictionaries, one per batch and requested product:

- `product`: requested DataLink product name.
- `source_ids`: exact integer IDs delivered in the file.
- `unavailable_source_ids`: IDs with no link advertising that product.
- `path`: local FITS ZIP as a `pathlib.Path`, or `None` if none are available.
- `cached`: whether a completed file was reused.

IDs must be integers or decimal strings. Floating-point IDs are rejected,
since a rounded 19-digit identifier cannot be recovered. Duplicate IDs are
removed in input order. `products` accepts one string or a sequence.

DataLink discovery determines availability for each source. A service error
raises; a product advertised but missing from the downloaded FITS also raises.
The downloader verifies source IDs before marking a batch complete. Product
names must match the service exactly: an unadvertised name is reported as
unavailable, so check names before interpreting an all-missing result.

`RAW` groups data in server-defined FITS tables; `INDIVIDUAL` requests
individual-source files. Both are returned in ZIP archives. The raw bytes,
FITS headers, array columns, units and covariance information are retained.
No XP calibration or resampling is applied, and RVS spectra retain the archive's
normalization. Read a member without extracting it:

```python
from io import BytesIO
from zipfile import ZipFile
from astropy.table import Table

path = next(batch["path"] for batch in batches if batch["path"] is not None)
with ZipFile(path) as archive:
    for name in archive.namelist():
        if name.lower().endswith(".fits"):
            table = Table.read(BytesIO(archive.read(name)), format="fits")
            print(name, table.colnames)
```

The cache stores discovery VOTables, original ZIPs and completion records.
Repeat the same call after an interruption: completed batches are skipped,
and incomplete batches restart from their beginning. This is batch resumption,
not HTTP byte-range resumption. Network failures propagate to the caller.
Use one directory per request and one running call per directory. Changing
IDs, their order, products, release, batch size or structure requires a new
directory; `workers` and network timeouts can change when resuming.

## Releases and limitations

DR3 exposes `XP_CONTINUOUS`, `XP_SAMPLED`, `RVS` and `EPOCH_PHOTOMETRY`,
among other products. Availability is source-dependent. Release and product
names are parameters so the same interface can be tested against future
releases; DR4 service compatibility is not yet validated. Catalogue astrometry
comes from the ADQL query; these APIs do not manufacture epoch astrometry
from catalogue uncertainties or perform a joint fit.

The APIs follow the public [Gaia TAP/DataLink interface](https://astroquery.readthedocs.io/en/latest/gaia/gaia.html).
Download time depends on server preparation and streaming as well as network
bandwidth. Start with a small batch on the machine that will run the full job.

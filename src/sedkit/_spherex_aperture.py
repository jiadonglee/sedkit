# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "talltable @ git+https://github.com/cmhainje/talltable.git@37548cfa6ee45c613252b8369ed034e40e265448#subdirectory=packages/core",
#   "spexpi @ git+https://github.com/fkiwy/spexpi.git@96b9ff95b4497f18b7cfc75dc6b38bcb3aff1b0f",
#   "pandas", "astropy", "healpy", "pyarrow",
# ]
# ///
"""QR2 pixel acquisition and aperture extraction from XphereX (MIT).

Source: jiadonglee/XphereX, revision b313934d84fea9221c9607ba70f259c025faf660.
Copyright (c) 2026 Jiadong Li. See the package LICENSE.
Runs in an isolated Python 3.12 environment through uv.
"""

from pathlib import Path
import argparse
import json
import time
import healpy as hp
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from talltable import PixelQuery
from spexpi.spherex_pipeline import (
    PipelineConfig, aperture_flux_one_radius, build_spectrum_table,
    convert_to_ujy, convert_variance_to_ujy2, load_sapm, process_spectrum_table,
)

DATA_RELEASE = "qr2"
SPECTRAL_CHANNELS_COLLECTION = "cal-sch-v1-2026-106"
APERTURE_RADIUS_PIX = 2.0
QUERY_EPOCH_DEFAULT = 2025.5
QUERY_ATTEMPTS = 3
QUERY_BACKOFF_S = 5.0
_TRANSIENT_MARKERS = (
    "timed out", "timeout", "connection", "closed stream",
    "prematurely", "502", "503", "504", "gateway",
)

def resolve_target(ra: float, dec: float, pmra: float=0.0, pmdec: float=0.0, ref_epoch: float | None=None, query_epoch: float=QUERY_EPOCH_DEFAULT) -> SkyCoord:
    """Return the sky position to query.

    If ``ref_epoch`` is given, propagate the input coordinates with proper
    motion from ``ref_epoch`` to ``query_epoch``. Otherwise return the input
    coordinates unchanged. ``pmra`` is mu_alpha*cos(dec) in mas/yr.
    """
    if ref_epoch is None:
        return SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr, distance=100 * u.pc, obstime=Time(ref_epoch, format='decimalyear'))
    return target.apply_space_motion(Time(query_epoch, format='decimalyear'))

def _is_transient(exc: Exception) -> bool:
    import requests
    if isinstance(exc, requests.exceptions.RequestException):
        return True
    if isinstance(exc, RuntimeError):
        return any((m in str(exc).lower() for m in _TRANSIENT_MARKERS))
    return False

def query_pixels(target: SkyCoord, radius_arcmin: float) -> pd.DataFrame:
    """Fetch pixels, retaining flags for the downstream aperture selection."""
    return _execute_query(lambda q: q.disc(target.ra.deg, target.dec.deg, radius_arcmin))

def _execute_query(region_fn) -> pd.DataFrame:
    """Query TallTable with transient-error retry and attach sky coordinates."""
    import random
    for attempt in range(1, QUERY_ATTEMPTS + 1):
        try:
            q = PixelQuery(web=True)
            region_fn(q)
            table = q.flags(custom_mask=0).with_wavelengths().with_rowcoldet().execute()
            break
        except Exception as exc:
            if attempt == QUERY_ATTEMPTS or not _is_transient(exc):
                raise
            delay = QUERY_BACKOFF_S * 3 ** (attempt - 1) * (0.5 + random.random())
            print(f'  [RETRY {attempt}/{QUERY_ATTEMPTS - 1}] {type(exc).__name__}; waiting {delay:.1f}s', flush=True)
            time.sleep(delay)
    pixels = table.to_pandas()
    pixels['ra'], pixels['dec'] = hp.pix2ang(2 ** 22, pixels['hphigh'].to_numpy(np.int64), nest=True, lonlat=True)
    return pixels

def build_config(outdir: Path) -> PipelineConfig:
    """SPExPI config shared across sources (calibration caches are reused)."""
    cfg = PipelineConfig(photometry_method='aperture', aperture_radii_pix=(APERTURE_RADIUS_PIX,), selected_aperture_radius_pix=APERTURE_RADIUS_PIX, annulus_inner_scale=2.0, annulus_outer_scale=3.0, reduce_oversampling=True, use_spectral_channels_for_binning=True, spectral_channels_collection=SPECTRAL_CHANNELS_COLLECTION, sapm_cache_dir=str(outdir / 'calibration_cache' / 'sapm'), spectral_channels_cache_dir=str(outdir / 'calibration_cache' / 'spectral_channels'), plot_spectrum=False)
    setattr(cfg, '_active_data_release', DATA_RELEASE)
    return cfg

def _measure_exposure_aperture(group, cfg, sapm_cache):
    row = group['row'].to_numpy(int)
    col = group['col'].to_numpy(int)
    detector = int(group['det'].iloc[0])
    design = np.c_[col, row, np.ones(len(group))]
    affine = np.linalg.lstsq(design, np.c_[group['_off_lon_arcsec'].to_numpy(), group['_off_lat_arcsec'].to_numpy()], rcond=None)[0]
    matrix = np.array([[affine[0, 0], affine[1, 0]], [affine[0, 1], affine[1, 1]]])
    try:
        target_col, target_row = np.linalg.solve(matrix, -affine[2])
    except np.linalg.LinAlgError:
        return None
    row0, row1, col0, col1 = (row.min(), row.max(), col.min(), col.max())
    shape = (row1 - row0 + 1, col1 - col0 + 1)
    if len(group) < 30 or max(shape) > 30:
        return None
    image = np.full(shape, np.nan)
    variance = np.full(shape, np.nan)
    zodi = np.zeros(shape)
    flags = np.full(shape, 1 << 6, dtype=np.int32)
    y, x = (row - row0, col - col0)
    image[y, x] = group['flux']
    variance[y, x] = group['variance']
    zodi[y, x] = group['zodi']
    flags[y, x] = group['flags'].to_numpy(np.int32)
    if detector not in sapm_cache:
        sapm_cache[detector] = load_sapm(detector, cfg, DATA_RELEASE)
    sapm, sapm_header = sapm_cache[detector]
    sapm = sapm[np.ix_(np.arange(row0, row1 + 1), np.arange(col0, col1 + 1))]
    image_header = fits.Header({'BUNIT': 'MJy/sr'})
    flux_ujy = convert_to_ujy(image - zodi, image_header, sapm, sapm_header)
    variance_ujy2 = convert_variance_to_ujy2(variance, image_header, sapm, sapm_header)
    flux, error, area, n_bad = aperture_flux_one_radius(flux_ujy, variance_ujy2, flags, target_col - col0, target_row - row0, APERTURE_RADIUS_PIX, cfg)
    wave_model = np.linalg.lstsq(design, group['wavelength'].to_numpy(float), rcond=None)[0]
    wavelength = float(wave_model @ [target_col, target_row, 1.0])
    if not (np.isfinite(flux) and np.isfinite(error) and (error > 0)):
        return None
    return {'imageid': int(group['imageid'].iloc[0]), 'detector': detector, 'wavelength_um': wavelength, 'flux_jy': flux * 1e-06, 'error_jy': error * 1e-06,
            'aperture_area_pix': float(area), 'n_bad_aperture': int(n_bad),
            'valid': bool(n_bad == 0 and np.isclose(area, np.pi * APERTURE_RADIUS_PIX**2, rtol=0, atol=1e-6))}

def aperture_spectrum(pixels, target, cfg, sapm_cache):
    """Return all measured exposures and bin only complete, unflagged apertures."""
    if len(pixels) == 0:
        return (pd.DataFrame(), _empty_spectrum())
    offsets = SkyCoord(pixels['ra'].to_numpy() * u.deg, pixels['dec'].to_numpy() * u.deg).transform_to(target.skyoffset_frame())
    pixels = pixels.assign(_off_lon_arcsec=offsets.lon.arcsec, _off_lat_arcsec=offsets.lat.arcsec)
    measurements = [result for _, group in pixels.groupby('imageid', sort=False) if (result := _measure_exposure_aperture(group, cfg, sapm_cache)) is not None]
    raw = pd.DataFrame(measurements)
    if raw.empty:
        return (raw, _empty_spectrum())
    raw = raw.sort_values('wavelength_um').reset_index(drop=True)
    rows = [{'wavelength': r.wavelength_um, 'flux': r.flux_jy, 'err': r.error_jy, 'detector': r.detector, 'photometry_method': 'aperture', 'aperture_radius_pix': APERTURE_RADIUS_PIX, 'status': 'ok'} for r in raw[raw['valid']].itertuples()]
    if not rows:
        return raw, _empty_spectrum()
    binned = process_spectrum_table(build_spectrum_table(rows, APERTURE_RADIUS_PIX, 'aperture'), cfg).to_pandas()
    binned.columns = ['wavelength_um', 'flux_jy', 'error_jy', 'detector']
    return (raw, binned)

def _empty_spectrum():
    return pd.DataFrame(columns=['wavelength_um', 'flux_jy', 'error_jy', 'detector'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ra", "dec", "pmra", "pmdec", "ref-epoch"):
        parser.add_argument("--" + name, type=float, required=True)
    parser.add_argument("--radius", type=float, default=0.8)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    target = resolve_target(args.ra, args.dec, args.pmra, args.pmdec, args.ref_epoch)
    pixel_path = args.outdir / "pixels.parquet"
    if pixel_path.exists() and not args.refresh:
        pixels = pd.read_parquet(pixel_path)
    else:
        pixels = query_pixels(target, args.radius)
        pixels.to_parquet(pixel_path, index=False)
    print(f"TallTable: {len(pixels):,} pixels", flush=True)
    cfg = build_config(args.cache_root)
    raw, spectrum = aperture_spectrum(pixels, target, cfg, {})
    raw.to_csv(args.outdir / "aperture_exposures.csv", index=False)
    if spectrum.empty or not np.any(np.isfinite(spectrum.error_jy) & (spectrum.error_jy > 0)):
        raise RuntimeError("no usable SPHEREx aperture measurements for this target")
    spectrum.to_csv(args.outdir / "aperture_spectrum.csv", index=False)
    metadata = dict(data_release=DATA_RELEASE, method="aperture",
        spectral_channels_collection=SPECTRAL_CHANNELS_COLLECTION,
        extraction_source="https://github.com/jiadonglee/XphereX",
        extraction_revision="b313934d84fea9221c9607ba70f259c025faf660",
        aperture_radius_pix=APERTURE_RADIUS_PIX, query_radius_arcmin=args.radius,
        ra=args.ra, dec=args.dec, pmra=args.pmra, pmdec=args.pmdec,
        ref_epoch=args.ref_epoch, query_epoch=QUERY_EPOCH_DEFAULT,
        query_ra=float(target.ra.deg), query_dec=float(target.dec.deg),
        n_pixels=len(pixels), n_exposures=len(raw),
        n_valid_exposures=int(raw.valid.sum()),
        aperture_quality="no fatal aperture flags and full unmasked area",
        input_unit="Jy", proper_motion="query-centre propagation to 2025.5")
    (args.outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(spectrum)} channels to {args.outdir}", flush=True)


if __name__ == "__main__":
    main()

"""Gaia-position crossmatches to archival ultraviolet photometry."""

from pathlib import Path
import json

import numpy as np

from .fetch import _row_dict
from .gaia import _ids

# Approximate observing epochs, rather than the J2000 coordinate equinox.
CATALOGUES = {
    "galex": ("II/335/galex_ais", 2007.0, "objid"),
    "swift": ("II/339/uvotssc1", 2007.5, "SrcID"),
    "xmm": ("II/370/xmmom5s", 2010.0, "ID"),
}


def _position(table, epoch):
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    import astropy.units as u

    ra = np.asarray(table["ra"], float).copy()
    dec = np.asarray(table["dec"], float).copy()
    moved = np.zeros(len(table), bool)
    if {"pmra", "pmdec"} <= set(table.colnames):
        pmra = np.asarray(np.ma.filled(table["pmra"], np.nan), float)
        pmdec = np.asarray(np.ma.filled(table["pmdec"], np.nan), float)
        ref = (np.asarray(np.ma.filled(table["ref_epoch"], np.nan), float)
               if "ref_epoch" in table.colnames else np.full(len(table), 2016.0))
        moved = np.isfinite(pmra) & np.isfinite(pmdec) & np.isfinite(ref)
        if moved.any():
            coord = SkyCoord(ra=ra[moved]*u.deg, dec=dec[moved]*u.deg,
                             pm_ra_cosdec=pmra[moved]*u.mas/u.yr,
                             pm_dec=pmdec[moved]*u.mas/u.yr,
                             obstime=Time(ref[moved], format="jyear"))
            coord = coord.apply_space_motion(new_obstime=Time(epoch, format="jyear"))
            ra[moved], dec[moved] = coord.ra.deg, coord.dec.deg
    return ra, dec, moved


def _bands(row, catalogue):
    from .subdwarf import GALEX_BRIGHT

    result = {}
    bands = ("FUV", "NUV") if catalogue == "galex" else ("UVW2", "UVM2", "UVW1", "U", "B", "V")
    for band in bands:
        if catalogue == "galex":
            mag, err = row.get(band+"mag"), row.get("e_"+band+"mag")
            quality = row.get("Fafl" if band == "FUV" else "Nafl")
            clean = quality is not None and quality <= 1
            extended = None
        elif catalogue == "swift":
            mag = row.get(band+"-AB")
            err = row.get("e_"+(band if band.startswith("UV") else band+"mag"))
            quality, extended = row.get("f"+band), row.get("x"+band)
            clean = quality == 0 and extended == 0
        else:
            key = band+"mAB"
            mag, err = row.get(key), row.get("e_"+key)
            quality, extended = row.get("q."+band), row.get("x"+band)
            clean = bool(quality) and set(str(quality).strip()) == {"F"} and extended == 0
        measured = (mag is not None and err is not None
                    and np.isfinite([mag, err]).all() and err > 0)
        usable = bool(measured and clean)
        if catalogue == "galex" and measured:
            usable &= mag > GALEX_BRIGHT[band]
        result[band] = dict(mag_ab=mag, mag_error=err, quality_flag=quality,
                            extended=extended, usable=bool(usable))
    return result


def _summary(matches, source_ids, catalogue):
    id_column = CATALOGUES[catalogue][2]
    result = {sid: dict(status="no_match", n_matches=0, n_sources=0, matches=[]) for sid in source_ids}
    for row in matches:
        raw = _row_dict(row)
        sid = str(raw["source_id"])
        result[sid]["matches"].append(dict(separation_arcsec=float(raw["angDist"]),
                                            bands=_bands(raw, catalogue), raw=raw))
    for entry in result.values():
        records = entry["matches"]
        records.sort(key=lambda r: r["separation_arcsec"])
        identities = {r["raw"].get(id_column) for r in records}
        entry["n_matches"] = len(records)
        entry["n_sources"] = len(identities)
        if records:
            entry["status"] = "ambiguous" if len(identities) > 1 else "matched"
        if entry["status"] == "ambiguous":
            for record in records:
                for band in record["bands"].values():
                    band["usable"] = False
    return result


def crossmatch_uv(sources, *, cache_dir="data/uv", catalogues=("galex", "swift", "xmm"),
                  radius_arcsec=3.0, refresh=False):
    """Crossmatch an Astropy source table; return records keyed by Gaia ID.

    Requires spbin[download]. Input columns are source_id, ra, dec (ICRS
    degrees), optionally pmra, pmdec (mas/yr) and ref_epoch (Julian year,
    default 2016). Proper motion is applied at approximate catalogue epochs.
    All matches, repeat observations, AB magnitudes and native quality flags
    are retained. Multiple distinct catalogue sources are marked ambiguous.
    No match is not an upper limit. These records are UV diagnostics and do
    not extend the 168-channel fitting vector.
    """
    from astropy.table import Table
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    if not np.isfinite(radius_arcsec) or radius_arcsec <= 0:
        raise ValueError("provide a positive finite UV matching radius")
    for name in catalogues:
        if name not in CATALOGUES:
            raise ValueError(f"unknown UV catalogue: {name}")
    source_ids = [str(_ids([value])[0]) for value in sources["source_id"]]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("provide one row per Gaia source_id")
    if not source_ids:
        return {}
    directory = Path(cache_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    result = {sid: {} for sid in source_ids}
    for name in catalogues:
        catalogue, epoch, _ = CATALOGUES[name]
        ra, dec, moved = _position(sources, epoch)
        upload = Table([source_ids, ra, dec], names=["source_id", "ra", "dec"])
        path = directory / f"{name}_matches.ecsv"
        request_path = directory / f"{name}_positions.ecsv"
        if path.exists() and not refresh:
            old = Table.read(request_path, format="ascii.ecsv")
            same = (old.meta.get("radius_arcsec") == radius_arcsec and len(old) == len(upload)
                    and all(np.array_equal(old[key], upload[key]) for key in upload.colnames))
            if not same:
                raise ValueError("UV input changed; use refresh=True or a new cache_dir")
            matches = Table.read(path, format="ascii.ecsv")
        else:
            tables = Vizier(columns=["**", "_r"], row_limit=-1).query_region(
                SkyCoord(ra=ra*u.deg, dec=dec*u.deg), radius=radius_arcsec*u.arcsec,
                catalog=catalogue, frame="icrs", cache=not refresh)
            if tables:
                matches = tables[0]
                matches["source_id"] = [source_ids[int(q)-1] for q in matches["_q"]]
                indices = np.asarray(matches["_q"], int)-1
                centres = SkyCoord(ra=ra[indices]*u.deg, dec=dec[indices]*u.deg)
                positions = SkyCoord(matches["RAJ2000"], matches["DEJ2000"], unit=u.deg)
                matches["angDist"] = centres.separation(positions).arcsec
            else:
                matches = Table(names=["source_id", "angDist"], dtype=["U32", float])
            upload.meta["radius_arcsec"] = radius_arcsec
            upload.write(request_path, format="ascii.ecsv", overwrite=True)
            matches.write(path, format="ascii.ecsv", overwrite=True)
        summary = _summary(matches, source_ids, name)
        for i, sid in enumerate(source_ids):
            result[sid][name] = dict(catalogue=catalogue, match_epoch_jyear=epoch,
                                     proper_motion_applied=bool(moved[i]),
                                     radius_arcsec=float(radius_arcsec), **summary[sid])
    (directory / "uv.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def download_uv(sed, *, cache_dir="data", **kwargs):
    """Crossmatch one SED and attach the returned records to metadata['uv']."""
    from astropy.table import Table

    row = {key: sed.metadata[key] for key in ("ra", "dec", "pmra", "pmdec", "ref_epoch")
           if key in sed.metadata}
    row["source_id"] = str(sed.source_id)
    result = crossmatch_uv(Table(rows=[row]),
                          cache_dir=Path(cache_dir).expanduser()/str(sed.source_id)/"uv", **kwargs)
    sed.metadata["uv"] = result[str(sed.source_id)]
    return sed.metadata["uv"]

"""Public Gaia DR3 XP and Gaia-linked 2MASS/AllWISE measurements."""

from pathlib import Path
from io import BytesIO
import warnings

import numpy as np

from .data import SED
from .model import StellarModel

BANDS = ("J", "H", "Ks", "W1", "W2")
# Catalogue Vega zero points, in Jy. WISE wavelengths follow the checkpoint.
ZERO_JY = (1594.0, 1024.0, 666.7, 309.540, 171.787)


def _row_dict(row):
    result = {}
    for key in row.colnames:
        value = row[key]
        if np.ma.is_masked(value):
            value = None
        elif isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode()
        result[key] = value
    return result


def _tap(query):
    """One-source synchronous TAP query with a bounded network timeout."""
    import requests
    from astropy.table import Table
    response = requests.get("https://gea.esac.esa.int/tap-server/tap/sync",
        params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": query},
        timeout=60)
    response.raise_for_status()
    table = Table.read(BytesIO(response.content), format="votable")
    table.rename_columns(table.colnames, [name.lower() for name in table.colnames])
    return table


def _query_cached(query, path):
    from astropy.table import Table
    if path.exists():
        return Table.read(path, format="ascii.ecsv")
    table = _tap(query)
    table.write(path, format="ascii.ecsv", overwrite=True)
    return table


def _resolve(ra, dec, radius_arcsec):
    ra, dec, radius = float(ra), float(dec), float(radius_arcsec) / 3600
    if not np.isfinite([ra, dec, radius]).all() or not 0 <= ra < 360 or not -90 <= dec <= 90 or radius <= 0:
        raise ValueError("provide finite ICRS coordinates in degrees and a positive radius")
    query = f"""SELECT TOP 2 source_id, ra, dec,
        DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra}, {dec})) AS separation
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(POINT('ICRS',ra,dec), CIRCLE('ICRS',{ra},{dec},{radius}))
        ORDER BY separation"""
    table = _tap(query)
    if len(table) != 1:
        raise ValueError("coordinate search must find exactly one Gaia source; use source_id or a smaller radius")
    return str(int(table[0]["source_id"]))


def _photometry(rows, wave):
    """Catalogue magnitudes -> equivalent F_lambda at nominal wavelengths."""
    flux, error = np.full(5, np.nan), np.full(5, np.nan)
    valid = np.zeros(5, bool)
    metadata = {}
    for label, table, indices, mag_names, err_names in rows:
        metadata[label] = [_row_dict(row) for row in table]
        if len(table) != 1:
            continue
        row = metadata[label][0]
        unique = row["n_mates"] == 0 and row["n_neighbours"] == 1
        quality = str(row.get("ph_qual") or "")
        if label == "2MASS":
            point_source = row.get("ext_key") in (None, 0)
            clean = [True] * 3
        else:
            point_source = row.get("ext_flag", row.get("ext_flg", 0)) == 0
            contamination = str(row.get("cc_flags") or "")
            clean = [len(contamination) > i and contamination[i] == "0" for i in range(2)]
        for position, (i, mag_key, err_key) in enumerate(zip(indices, mag_names, err_names)):
            mag, sigma = row.get(mag_key), row.get(err_key)
            if mag is None or sigma is None:
                continue
            jy = ZERO_JY[i] * 10**(-0.4 * float(mag))
            flux[i] = jy * 299792458.0 * 10.0 / (wave[i] * 1000)**2
            error[i] = flux[i] * np.log(10) / 2.5 * float(sigma)
            valid[i] = (unique and point_source and clean[position]
                        and len(quality) > position and quality[position] == "A"
                        and np.isfinite(flux[i]) and np.isfinite(error[i]) and error[i] > 0)
    return flux, error, valid, metadata


def download(source_id=None, *, ra=None, dec=None, radius_arcsec=2.0,
             cache_dir="data", refresh=False):
    """Download one source; coordinates are ICRS degrees at Gaia's epoch.

    Requires sedlet[download]. Uses Gaia's published crossmatches for
    2MASS/AllWISE. Native fluxes/errors and catalogue quality flags are
    retained; only unambiguous, A-quality photometry enters the default mask.
    Existing source downloads are reused unless refresh=True.
    """
    from astropy.table import Table, vstack

    if source_id is None:
        if ra is None or dec is None:
            raise ValueError("provide a Gaia source_id or ra and dec")
        source_id = _resolve(ra, dec, radius_arcsec)
    source_id = str(int(source_id))
    directory = Path(cache_dir).expanduser() / source_id
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "sed.npz"
    if output.exists() and not refresh:
        return SED.load(output)
    if refresh:
        # These are this function's own cached public catalogue products.
        for name in ("gaia.ecsv", "2mass.ecsv", "allwise.ecsv", "xp.xml", "xp.ecsv"):
            (directory / name).unlink(missing_ok=True)
    gaia = _query_cached(f"""SELECT source_id, ra, dec, ref_epoch, pmra, pmdec,
        parallax, parallax_error, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
        ruwe, phot_bp_rp_excess_factor, has_xp_continuous
        FROM gaiadr3.gaia_source WHERE source_id={source_id}""", directory / "gaia.ecsv")
    if len(gaia) != 1:
        raise ValueError("Gaia source_id was not found in DR3")
    metadata = _row_dict(gaia[0])
    metadata["source_id"] = source_id
    metadata["data_release"] = "Gaia DR3"
    two_mass = _query_cached(f"""SELECT tm.*, x.angular_distance AS match_arcsec,
        x.number_of_neighbours AS n_neighbours, x.number_of_mates AS n_mates
        FROM gaiadr3.tmass_psc_xsc_best_neighbour AS x
        JOIN gaiadr3.tmass_psc_xsc_join AS j
          ON j.clean_tmass_psc_xsc_oid=x.clean_tmass_psc_xsc_oid
        JOIN gaiadr1.tmass_original_valid AS tm ON j.original_psc_source_id=tm.designation
        WHERE x.source_id={source_id}""", directory / "2mass.ecsv")
    wise = _query_cached(f"""SELECT w.*, x.angular_distance AS match_arcsec,
        x.number_of_neighbours AS n_neighbours, x.number_of_mates AS n_mates
        FROM gaiadr3.allwise_best_neighbour AS x
        JOIN gaiadr1.allwise_original_valid AS w ON w.allwise_oid=x.allwise_oid
        WHERE x.source_id={source_id}""", directory / "allwise.ecsv")
    model = StellarModel()
    flux, error, mask = np.full(168, np.nan), np.full(168, np.nan), np.zeros(168, bool)
    phot_flux, phot_error, phot_mask, provenance = _photometry(
        [("2MASS", two_mass, range(3), ("j_m", "h_m", "ks_m"), ("j_msigcom", "h_msigcom", "ks_msigcom")),
         ("AllWISE", wise, range(3, 5), ("w1mpro", "w2mpro"), ("w1mpro_error", "w2mpro_error"))],
        model.wavelength_um[61:66])
    flux[61:66], error[61:66], mask[61:66] = phot_flux, phot_error, phot_mask
    metadata.update(provenance)
    if metadata["has_xp_continuous"]:
        xp_path = directory / "xp.ecsv"
        if xp_path.exists():
            xp = Table.read(xp_path, format="ascii.ecsv")
        else:
            raw = directory / "xp.xml"
            if not raw.exists():
                from astroquery.gaia import Gaia
                products = Gaia.load_data(ids=[source_id], data_release="Gaia DR3",
                                          retrieval_type="XP_CONTINUOUS", data_structure="RAW",
                                          format="votable", verbose=False)
                tables = [item.to_table() for items in products.values() for item in items]
                if not tables:
                    raise RuntimeError("Gaia lists XP data but the DataLink response is empty")
                table = vstack(tables) if len(tables) > 1 else tables[0]
                table.write(raw, format="votable", overwrite=True)
            from gaiaxpy import calibrate
            calibrated, sampling = calibrate(str(raw),
                sampling=model.wavelength_um[:61] * 1000, truncation=False, save_file=False)
            selected = calibrated[calibrated["source_id"].astype(str) == source_id]
            if len(selected) != 1:
                raise RuntimeError("GaiaXPy did not return the requested source")
            row = selected.iloc[0]
            xp = Table([sampling, np.asarray(row["flux"], float), np.asarray(row["flux_error"], float)],
                       names=["wavelength_nm", "flux_W_m2_nm", "error_W_m2_nm"])
            xp.write(xp_path, format="ascii.ecsv", overwrite=True)
        flux[:61] = np.asarray(xp["flux_W_m2_nm"]) * 1e18
        error[:61] = np.asarray(xp["error_W_m2_nm"]) * 1e18
        mask[:61] = np.isfinite(flux[:61]) & np.isfinite(error[:61]) & (error[:61] > 0)
    else:
        warnings.warn("this Gaia source has no continuous XP spectrum", stacklevel=2)
    metadata["xp_error_model"] = "GaiaXPy marginal errors; inter-channel measurement correlations omitted"
    sed = SED(flux, error, mask, metadata["parallax"] or np.nan,
              metadata["parallax_error"] or 0.0, source_id, metadata)
    sed.save(output)
    return sed

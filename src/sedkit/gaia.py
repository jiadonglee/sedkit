"""Resumable public Gaia catalogue queries and batched DataLink products."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from numbers import Integral
from pathlib import Path
import time
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit
from zipfile import ZipFile

import numpy as np

TAP = {
    "esa": "https://gea.esac.esa.int/tap-server/tap",
    "ari": "https://gaia.ari.uni-heidelberg.de/tap",
}
DATALINK = "https://gea.esac.esa.int/data-server/datalink/links"


def _save_json(path, value):
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _bind(directory, settings):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "request.json"
    if path.exists():
        if json.loads(path.read_text()) != settings:
            raise ValueError("request changed; choose a new cache_dir")
    else:
        _save_json(path, settings)


def _ids(values):
    """Reject floating-point IDs before they can silently round."""
    if isinstance(values, (str, Integral)):
        values = [values]
    result = []
    for value in values:
        if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
            raise ValueError("source IDs must be positive integers or decimal strings")
        if not isinstance(value, (str, Integral)):
            raise ValueError("source IDs must be integers or strings, never floats")
        if isinstance(value, str) and not value.isdecimal():
            raise ValueError("source IDs must be decimal strings")
        sid = int(value)
        if not 0 < sid <= np.iinfo(np.int64).max:
            raise ValueError("source ID is outside the positive int64 range")
        result.append(sid)
    return list(dict.fromkeys(result))


def _votable(source):
    from astropy.io.votable import parse

    doc = parse(source)
    for resource in doc.resources:
        infos = list(resource.infos)
        for table in resource.tables:
            infos.extend(table.infos)
        for info in infos:
            if info.name == "QUERY_STATUS" and info.value != "OK":
                raise RuntimeError(f"Gaia {info.value}: {info.content}")
    return doc.get_first_table().to_table(use_names_over_ids=True)


def _stream(url, path, *, params=None, timeout):
    import requests

    temporary = path.with_suffix(path.suffix + ".part")
    with requests.get(url, params=params, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
    return temporary


def query_gaia(query, *, cache_dir="data/gaia-query", tap="esa", maxrec=1000000,
               wait=600, timeout=(30, 180)):
    """Run an asynchronous ADQL query and return an Astropy Table.

    Requires sedkit[download]. ``tap`` is 'esa', 'ari', or a TAP URL.
    The directory stores the job URL and original VOTable for resumption.
    Repeating the same request reuses its result; changed requests require
    a new directory. Server truncation raises rather than returning a
    partial catalogue. ``wait`` bounds polling, excluding network calls;
    on TimeoutError call again with the same directory to resume.
    """
    import pyvo

    if not query.strip() or maxrec < 1 or wait <= 0:
        raise ValueError("provide a nonempty query, positive maxrec and wait")
    directory = Path(cache_dir).expanduser()
    endpoint = TAP.get(tap, tap).rstrip("/")
    _bind(directory, {"query": query, "tap": endpoint, "maxrec": maxrec})
    raw = directory / "catalogue.vot"
    if not raw.exists():
        job_path = directory / "job.json"
        if job_path.exists():
            job = pyvo.dal.AsyncTAPJob(json.loads(job_path.read_text())["url"], delete=False)
        else:
            job = pyvo.dal.TAPService(endpoint).submit_job(query, maxrec=maxrec)
            _save_json(job_path, {"url": job.url})
        if job.phase in ("PENDING", "HELD"):
            job.run()
        deadline = time.monotonic() + wait
        while job.phase not in ("COMPLETED", "ERROR", "ABORTED"):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Gaia job still running: {job.url}; repeat this call to resume")
            time.sleep(min(5, remaining))
        job.raise_if_error()
        temporary = _stream(job.result_uri, raw, timeout=timeout)
        table = _votable(str(temporary))
        temporary.replace(raw)
    else:
        table = _votable(str(raw))
    table.rename_columns(table.colnames, [name.lower() for name in table.colnames])
    return table


def _discover(ids, release, path, endpoint, timeout):
    if not path.exists():
        temporary = _stream(endpoint, path, params={
            "ID": ",".join(f"{release} {sid}" for sid in ids)}, timeout=timeout)
        table = _votable(str(temporary))
        _check_links(table)
        temporary.replace(path)
    else:
        table = _votable(str(path))
        _check_links(table)
    return table


def _check_links(table):
    if "error_message" in table.colnames:
        errors = [str(value) for value in table["error_message"]
                  if not np.ma.is_masked(value) and str(value).strip()]
        if errors:
            raise RuntimeError("Gaia DataLink: " + "; ".join(errors))


def _archive_ids(path):
    """Check the FITS payload and identify the sources actually delivered."""
    from astropy.io import fits

    found = set()
    with ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith((".fits", ".fit")):
                continue
            with fits.open(BytesIO(archive.read(name)), memmap=False) as hdus:
                for hdu in hdus:
                    for key in ("SOURCEID", "source_id"):
                        if key in hdu.header:
                            found.update(_ids([hdu.header[key]]))
                    names = getattr(getattr(hdu, "columns", None), "names", None) or []
                    for column in names:
                        if column.lower() == "source_id":
                            found.update(_ids(hdu.data[column]))
    return found


def download_gaia(source_ids, *, products=("XP_CONTINUOUS",),
                  cache_dir="data/gaia-products", data_release="Gaia DR3",
                  batch_size=100, workers=2, data_structure="RAW",
                  datalink_url=DATALINK, timeout=(30, 300)):
    """Download Gaia products in batches, retaining the original FITS ZIPs.

    Accepts integer/string IDs (duplicates removed in input order).
    ``products`` is a product name or sequence of names, e.g.
    XP_CONTINUOUS, RVS, EPOCH_PHOTOMETRY for DR3. Names must match DataLink.
    ``data_structure`` is RAW or INDIVIDUAL. Requires sedkit[download].

    Returns one dict per batch/product: product, source_ids (delivered),
    unavailable_source_ids (not advertised), path (Path or None), cached.
    A product advertised but not delivered raises an error. Completed
    batches are reused on rerun; an interrupted batch restarts. Use one
    directory per request and do not run concurrent calls in that directory.
    No calibration, resampling or flux/error normalization is applied.
    """
    ids = _ids(source_ids)
    products = [products] if isinstance(products, str) else list(products)
    products = list(dict.fromkeys(products))
    if not products or any(not isinstance(p, str) or not p or
                           not p.replace("_", "").isalnum() for p in products):
        raise ValueError("provide nonempty DataLink product names")
    if (not isinstance(batch_size, Integral) or not 1 <= batch_size <= 5000
            or not isinstance(workers, Integral) or workers < 1):
        raise ValueError("batch_size must be 1..5000 and workers must be positive")
    if data_structure not in ("RAW", "INDIVIDUAL"):
        raise ValueError("data_structure must be RAW or INDIVIDUAL")
    directory = Path(cache_dir).expanduser()
    _bind(directory, {"source_ids": ids, "products": products,
                      "data_release": data_release, "batch_size": int(batch_size),
                      "data_structure": data_structure, "datalink_url": datalink_url})

    def fetch_batch(offset):
        batch = ids[offset:offset + batch_size]
        stem = f"{offset // batch_size:06d}"
        links = _discover(batch, data_release, directory / f"{stem}-links.vot",
                          datalink_url, timeout)
        available = {product: {} for product in products}
        for row in links:
            if np.ma.is_masked(row["access_url"]):
                continue
            url = str(row["access_url"])
            product = parse_qs(urlsplit(url).query).get("RETRIEVAL_TYPE", [""])[0]
            if product in available:
                sid = _ids([str(row["ID"]).split()[-1]])[0]
                available[product][sid] = url
        results = []
        for product in products:
            selected = [sid for sid in batch if sid in available[product]]
            missing = [sid for sid in batch if sid not in available[product]]
            path = directory / f"{stem}-{product}.zip"
            receipt = directory / f"{stem}-{product}.json"
            cached = path.exists() and receipt.exists()
            if selected and not cached:
                parts = urlsplit(available[product][selected[0]])
                params = parse_qs(parts.query)
                params.update(ID=[",".join(f"{data_release} {sid}" for sid in selected)],
                              FORMAT=["fits"], DATA_STRUCTURE=[data_structure],
                              USE_ZIP_ALWAYS=["true"], VALID_DATA=["false"])
                url = urlunsplit(parts._replace(query=urlencode(params, doseq=True)))
                temporary = _stream(url, path, timeout=timeout)
                delivered = _archive_ids(temporary)
                if delivered != set(selected):
                    raise RuntimeError(f"{product} batch {stem}: requested and delivered IDs differ; "
                                       f"missing={sorted(set(selected) - delivered)}, "
                                       f"unexpected={sorted(delivered - set(selected))}")
                temporary.replace(path)
                _save_json(receipt, {"source_ids": selected})
            results.append({"product": product, "source_ids": selected,
                            "unavailable_source_ids": missing,
                            "path": path if selected else None, "cached": cached})
        return results

    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        batches = pool.map(fetch_batch, range(0, len(ids), int(batch_size)))
        try:
            return [item for batch in batches for item in batch]
        except Exception:
            pool.shutdown(wait=False, cancel_futures=True)
            raise

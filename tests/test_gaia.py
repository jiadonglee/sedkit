from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
from astropy.table import Table, MaskedColumn

from sedkit.gaia import _archive_ids, _ids, _votable, download_gaia, query_gaia

IDS = [4318465066420528001, 4318465066420528002]


def vot(table):
    buffer = BytesIO()
    table.write(buffer, format="votable")
    return buffer.getvalue()


def archive(path, ids):
    table = Table({"source_id": ids, "flux": [[1., 2.]] * len(ids)})
    table["flux"].unit = "W m-2 nm-1"
    buffer = BytesIO()
    table.write(buffer, format="fits")
    with ZipFile(path, "w") as output:
        output.writestr("XP.fits", buffer.getvalue())


def test_exact_ids_and_invalid_inputs():
    assert _ids([str(IDS[0]), np.int64(IDS[1]), IDS[0]]) == IDS
    for value in ([float(IDS[0])], [True], [None], [np.ma.masked]):
        with pytest.raises(ValueError):
            _ids(value)


def test_votable_preserves_arrays_units_masks_and_rejects_overflow():
    table = Table({"source_id": IDS, "coefficients": [[1., 2.], [3., 4.]]})
    table["parallax"] = MaskedColumn([10., 20.], mask=[False, True], unit="mas")
    raw = vot(table)
    restored = _votable(BytesIO(raw))
    assert list(restored["source_id"]) == IDS
    assert restored["coefficients"].shape == (2, 2)
    assert str(restored["parallax"].unit) == "mas"
    assert restored["parallax"].mask[1]
    raw = raw.replace(b'</RESOURCE>', b'<INFO name="QUERY_STATUS" value="OVERFLOW"/></RESOURCE>')
    with pytest.raises(RuntimeError, match="OVERFLOW"):
        _votable(BytesIO(raw))


def test_batch_resume_and_unavailable(monkeypatch, tmp_path):
    links = Table({"ID": [f"Gaia DR3 {IDS[0]}"], "access_url": [
        "https://example.org/data?RETRIEVAL_TYPE=XP_CONTINUOUS&ID=unused"], "error_message": [""]})
    monkeypatch.setattr("sedkit.gaia._discover", lambda *args: links)
    calls = []

    def stream(url, path, **kwargs):
        calls.append(url)
        temporary = path.with_suffix(".part")
        archive(temporary, [IDS[0]])
        return temporary

    monkeypatch.setattr("sedkit.gaia._stream", stream)
    rows = download_gaia(IDS, products=["XP_CONTINUOUS", "RVS"], cache_dir=tmp_path)
    assert rows[0]["source_ids"] == [IDS[0]]
    assert rows[0]["unavailable_source_ids"] == [IDS[1]]
    assert rows[1]["path"] is None
    assert rows[1]["unavailable_source_ids"] == IDS
    assert _archive_ids(rows[0]["path"]) == {IDS[0]}
    with ZipFile(rows[0]["path"]) as zipped:
        restored = Table.read(BytesIO(zipped.read("XP.fits")), format="fits")
    assert str(restored["flux"].unit) == "W / (nm m2)"
    resumed = download_gaia(IDS, products=["XP_CONTINUOUS", "RVS"], cache_dir=tmp_path)
    assert resumed[0]["cached"] and len(calls) == 1
    with pytest.raises(ValueError, match="request changed"):
        download_gaia(IDS[:1], cache_dir=tmp_path)


def test_failed_batch_retries_without_marking_complete(monkeypatch, tmp_path):
    links = Table({"ID": [f"Gaia DR3 {sid}" for sid in IDS], "access_url": [
        "https://example.org/data?RETRIEVAL_TYPE=RVS"] * 2})
    monkeypatch.setattr("sedkit.gaia._discover", lambda *args: links)
    delivered = IDS[:1]

    def stream(url, path, **kwargs):
        temporary = path.with_suffix(".part")
        archive(temporary, delivered)
        return temporary

    monkeypatch.setattr("sedkit.gaia._stream", stream)
    with pytest.raises(RuntimeError, match="IDs differ"):
        download_gaia(IDS, products="RVS", cache_dir=tmp_path)
    assert not list(tmp_path.glob("*.zip"))
    assert not (tmp_path / "000000-RVS.json").exists()
    delivered = IDS
    assert download_gaia(IDS, products="RVS", cache_dir=tmp_path)[0]["source_ids"] == IDS


def test_query_resume_cache_and_overflow(monkeypatch, tmp_path):
    import pyvo

    job = SimpleNamespace(url="https://example.org/async/1", phase="COMPLETED",
                          result_uri="https://example.org/result", raise_if_error=lambda: None)
    submitted = []
    def submit(query, **kwargs):
        submitted.append(query)
        return job
    monkeypatch.setattr(pyvo.dal, "TAPService", lambda url: SimpleNamespace(submit_job=submit))
    monkeypatch.setattr(pyvo.dal, "AsyncTAPJob", lambda *args, **kwargs: job)
    raw = vot(Table({"SOURCE_ID": IDS}))
    overflow = True
    def stream(url, path, **kwargs):
        temporary = path.with_suffix(".part")
        data = raw.replace(b'</RESOURCE>', b'<INFO name="QUERY_STATUS" value="OVERFLOW"/></RESOURCE>') if overflow else raw
        temporary.write_bytes(data)
        return temporary
    monkeypatch.setattr("sedkit.gaia._stream", stream)
    with pytest.raises(RuntimeError, match="OVERFLOW"):
        query_gaia("SELECT source_id FROM gaiadr3.gaia_source", cache_dir=tmp_path)
    assert not (tmp_path / "catalogue.vot").exists()
    overflow = False
    result = query_gaia("SELECT source_id FROM gaiadr3.gaia_source", cache_dir=tmp_path)
    assert list(result["source_id"]) == IDS
    assert len(submitted) == 1
    result = query_gaia("SELECT source_id FROM gaiadr3.gaia_source", cache_dir=tmp_path)
    assert list(result["source_id"]) == IDS
    with pytest.raises(ValueError, match="request changed"):
        query_gaia("SELECT TOP 1 source_id FROM gaiadr3.gaia_source", cache_dir=tmp_path)


def test_individual_fits_header_id(tmp_path):
    from astropy.io import fits
    path = tmp_path / "rvs.zip"
    buffer = BytesIO()
    hdu = fits.BinTableHDU(Table({"flux": [1., 2.]}))
    hdu.header["SOURCEID"] = str(IDS[0])
    hdu.writeto(buffer)
    with ZipFile(path, "w") as output:
        output.writestr("rvs.fits", buffer.getvalue())
    assert _archive_ids(path) == {IDS[0]}

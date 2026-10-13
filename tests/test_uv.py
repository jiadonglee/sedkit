import numpy as np
import pytest
from astropy.table import Table

from spbin.uv import _bands, _position, _summary, crossmatch_uv


def test_uv_quality_and_distinct_neighbours():
    rows = Table({"source_id": ["1", "1"], "angDist": [0.3, 0.4], "SrcID": [7, 7],
                  "UVW2-AB": [17., 17.1], "e_UVW2": [.05, .06],
                  "fUVW2": [0, 1], "xUVW2": [0, 0]})
    entry = _summary(rows, ["1", "2"], "swift")
    assert entry["1"]["status"] == "matched"
    assert entry["1"]["n_matches"] == 2
    assert entry["1"]["matches"][0]["bands"]["UVW2"]["usable"]
    assert not entry["1"]["matches"][1]["bands"]["UVW2"]["usable"]
    assert entry["2"]["status"] == "no_match"
    rows["SrcID"][1] = 8
    entry = _summary(rows, ["1"], "swift")["1"]
    assert entry["status"] == "ambiguous"
    assert not any(r["bands"]["UVW2"]["usable"] for r in entry["matches"])
    assert not _bands({"FUVmag": 12., "e_FUVmag": .05, "Fafl": 0}, "galex")["FUV"]["usable"]
    xmm = {"UVM2mAB": 17., "e_UVM2mAB": .05, "q.UVM2": "FFFFFFFFFFFF", "xUVM2": 0}
    assert _bands(xmm, "xmm")["UVM2"]["usable"]
    xmm["q.UVM2"] = "FTFFFFFFFFFF"
    assert not _bands(xmm, "xmm")["UVM2"]["usable"]


def test_motion_uses_pmra_cosdec():
    sources = Table({"ra": [120.], "dec": [60.], "pmra": [1000.], "pmdec": [0.]})
    ra, dec, moved = _position(sources, 2006.)
    assert moved[0]
    assert (120.-ra[0])*3600*np.cos(np.deg2rad(dec[0])) == pytest.approx(10., abs=.001)


def test_full_vizier_rows_and_cache(monkeypatch, tmp_path):
    VizierClass = pytest.importorskip("astroquery.vizier").VizierClass

    calls = []
    def query(self, coordinates, **kwargs):
        calls.append(kwargs)
        return [Table({"_q": [1], "RAJ2000": [120.], "DEJ2000": [0.], "SrcID": [7],
                       "UVW2-AB": [17.], "e_UVW2": [.05], "fUVW2": [0], "xUVW2": [0]})]
    monkeypatch.setattr(VizierClass, "query_region", query)
    sources = Table({"source_id": [1521154374020165376], "ra": [120.], "dec": [0.]})
    first = crossmatch_uv(sources, cache_dir=tmp_path, catalogues=("swift",))
    second = crossmatch_uv(sources, cache_dir=tmp_path, catalogues=("swift",))
    assert first == second
    assert len(calls) == 1
    band = first["1521154374020165376"]["swift"]["matches"][0]["bands"]["UVW2"]
    assert band["usable"] and band["mag_error"] == .05
    sources["ra"][0] += .01
    with pytest.raises(ValueError, match="input changed"):
        crossmatch_uv(sources, cache_dir=tmp_path, catalogues=("swift",))


def test_download_adds_uv_to_cached_sed(monkeypatch, tmp_path):
    from spbin import SED, download
    import spbin.uv

    sed = SED(np.ones(168), np.ones(168), np.ones(168, bool), 10., source_id="1",
              metadata={"ra": 120., "dec": 0.})
    path = tmp_path/"1"/"sed.npz"
    sed.save(path)
    def attach(sed, **kwargs):
        sed.metadata["uv"] = {"swift": {"status": "no_match"}}
    monkeypatch.setattr(spbin.uv, "download_uv", attach)
    assert "uv" not in download("1", cache_dir=tmp_path).metadata
    result = download("1", cache_dir=tmp_path, uv=True)
    assert result.metadata["uv"] == SED.load(path).metadata["uv"]
    np.testing.assert_array_equal(result.flux, sed.flux)

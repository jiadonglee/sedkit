from io import BytesIO
from types import SimpleNamespace

from astropy.table import Table
import requests

from sedkit.fetch import _tap


def test_tap_normalizes_gaia_field_names_and_keeps_exact_id(monkeypatch):
    table = Table({"SOURCE_ID": [1521154374020165376], "ra": [190.1704458396]})
    output = BytesIO()
    table.write(output, format="votable")
    response = SimpleNamespace(content=output.getvalue(), raise_for_status=lambda: None)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)
    result = _tap("SELECT source_id, ra FROM gaiadr3.gaia_source")
    assert result.colnames == ["source_id", "ra"]
    assert int(result[0]["source_id"]) == 1521154374020165376

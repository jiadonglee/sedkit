"""Actual loopback HTTP fault tests. Known unsafe acceptances are strict xfails."""

import hashlib
import json
import threading
import time

import pytest
import requests

from sedkit.gaia import download_gaia, query_gaia
from benchmarks.gaia_service import BASE_ID, PRODUCTS, available, gaia_service


@pytest.fixture
def service():
    with gaia_service() as value:
        yield value


def options(service, tmp_path, n=100):
    return dict(source_ids=list(range(BASE_ID + 1, BASE_ID + n + 1)),
                products="XP_CONTINUOUS", cache_dir=tmp_path, batch_size=50,
                workers=1, datalink_url=service.url + "/links", timeout=(1, 1))


def data_requests(service):
    return [e for e in service.events if e["path"] == "/data"]


@pytest.mark.parametrize("n", [100, 1000, 10000])
def test_scale_partition_and_warm_no_http(service, tmp_path, n):
    opts = options(service, tmp_path, n)
    opts.update(products=PRODUCTS, workers=2, batch_size=100)
    rows = download_gaia(**opts)
    for product in PRODUCTS:
        delivered = {sid for row in rows if row["product"] == product for sid in row["source_ids"]}
        missing = {sid for row in rows if row["product"] == product for sid in row["unavailable_source_ids"]}
        assert delivered == {sid for sid in opts["source_ids"] if available(sid, product)}
        assert missing == set(opts["source_ids"]) - delivered
    count = len(service.events)
    warm = download_gaia(**opts)
    assert all(row["cached"] for row in warm)
    assert len(service.events) == count


@pytest.mark.parametrize("fault", ["429", "503", "disconnect", "corrupt_zip", "missing_id", "unexpected_id"])
def test_failed_batch_resume_preserves_completed_zip(service, tmp_path, fault):
    opts = options(service, tmp_path, 1000)
    service.inject(fault, after=1)
    with pytest.raises(Exception):
        download_gaia(**opts)
    # map() can schedule another batch before cancellation reaches the executor.
    completed = {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                 for p in tmp_path.glob("*.zip")}
    assert "000000-XP_CONTINUOUS.zip" in completed
    assert not (tmp_path / "000001-XP_CONTINUOUS.json").exists()
    assert not (tmp_path / "000001-XP_CONTINUOUS.zip").exists()
    failure_events = [e for e in data_requests(service) if e["fault"] == fault]
    assert len(failure_events) == 1  # API does not automatically retry.
    before = len(data_requests(service))
    rows = download_gaia(**opts)
    assert len(data_requests(service)) - before == 20 - len(completed)
    for name, previous in completed.items():
        p = tmp_path / name
        assert (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) == previous
    assert sum(len(row["source_ids"]) for row in rows) == 800
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("status", ["429", "503"])
def test_retry_after_is_not_honoured(service, tmp_path, status):
    service.inject(status)
    start = time.monotonic()
    with pytest.raises(requests.HTTPError) as exc:
        download_gaia(**options(service, tmp_path, 50))
    assert exc.value.response.headers["Retry-After"] == "1"
    assert time.monotonic() - start < 1
    assert len(data_requests(service)) == 1


@pytest.mark.parametrize("fault", ["overflow", "503", "414"])
def test_discovery_rejects_explicit_failures(service, tmp_path, fault):
    service.inject(fault, route="/links")
    with pytest.raises(Exception):
        download_gaia(**options(service, tmp_path))
    assert not (tmp_path / "000000-links.vot").exists()
    # Already scheduled other batches may finish even after a discovery error.
    assert not (tmp_path / "000000-XP_CONTINUOUS.zip").exists()
    assert download_gaia(**options(service, tmp_path))


@pytest.mark.xfail(strict=True, reason="main accepts silent discovery omission as unavailable, then caches it")
def test_silent_discovery_omission_is_not_scientific_unavailability(service, tmp_path):
    service.inject("silent_omission", route="/links")
    opts = options(service, tmp_path)
    rows = download_gaia(**opts)
    assert BASE_ID + 1 in {sid for row in rows for sid in row["source_ids"]}


@pytest.mark.parametrize("damage", ["zip", "receipt"])
@pytest.mark.xfail(strict=True, reason="main checks cache file existence, not ZIP integrity or receipt content")
def test_damaged_cache_must_not_be_accepted(service, tmp_path, damage):
    opts = options(service, tmp_path)
    download_gaia(**opts)
    if damage == "zip":
        (tmp_path / "000000-XP_CONTINUOUS.zip").write_bytes(b"corrupt")
    else:
        (tmp_path / "000000-XP_CONTINUOUS.json").write_text('{"source_ids": []}')
    with pytest.raises(Exception):
        download_gaia(**opts)


def test_real_tap_http_timeout_resume_and_overflow(service, tmp_path):
    opts = dict(query="SELECT TOP 100 source_id FROM gaiadr3.gaia_source", tap=service.url + "/tap",
                cache_dir=tmp_path / "resume", wait=0.02)
    service.hold = True
    with pytest.raises(TimeoutError):
        query_gaia(**opts)
    assert len(service.jobs) == 1
    job = json.loads((tmp_path / "resume/job.json").read_text())
    service.jobs["1"]["phase"] = "COMPLETED"
    assert len(query_gaia(**opts)) == 100
    assert len(service.jobs) == 1
    assert json.loads((tmp_path / "resume/job.json").read_text()) == job
    calls = len(service.events)
    assert len(query_gaia(**opts)) == 100
    assert len(service.events) == calls
    service.hold = False
    with pytest.raises(RuntimeError, match="OVERFLOW"):
        query_gaia(**dict(opts, cache_dir=tmp_path / "overflow", maxrec=99))
    assert not (tmp_path / "overflow/catalogue.vot").exists()


def test_expired_tap_job_requires_explicit_new_cache(service, tmp_path):
    opts = dict(query="SELECT TOP 100 source_id FROM gaiadr3.gaia_source", tap=service.url + "/tap",
                cache_dir=tmp_path, wait=0.01)
    service.hold = True
    with pytest.raises(TimeoutError):
        query_gaia(**opts)
    service.inject("404", route="/tap/async/")
    with pytest.raises(Exception):
        query_gaia(**opts)
    assert len(service.jobs) == 1  # No accidental duplicate expensive query.


@pytest.mark.xfail(strict=True, reason="TAP job can be created before submit_job returns its URL; retry creates another")
def test_interrupted_tap_submission_does_not_duplicate_job(service, tmp_path):
    opts = dict(query="SELECT TOP 100 source_id FROM gaiadr3.gaia_source", tap=service.url + "/tap",
                cache_dir=tmp_path, wait=1)
    service.inject("503", route="/tap/async/")
    with pytest.raises(Exception):
        query_gaia(**opts)
    assert len(service.jobs) == 1
    query_gaia(**opts)
    assert len(service.jobs) == 1


def test_individual_structure(service, tmp_path):
    rows = download_gaia(**options(service, tmp_path), data_structure="INDIVIDUAL")
    assert sum(len(row["source_ids"]) for row in rows) == 80


def test_product_spelling_can_look_like_all_unavailable(service, tmp_path):
    rows = download_gaia(**dict(options(service, tmp_path), products="RVS_DR4_UNKNOWN"))
    assert sum(len(row["unavailable_source_ids"]) for row in rows) == 100
    assert not data_requests(service)


def test_executor_waits_for_running_batch_after_failure(monkeypatch, service, tmp_path):
    from astropy.table import Table
    both_started = threading.Barrier(2)

    def discover(ids, *args):
        both_started.wait(timeout=2)
        if ids[0] == BASE_ID + 1:
            raise RuntimeError("first batch failed")
        time.sleep(0.25)
        return Table(names=["ID", "access_url"], dtype=["U64", "U64"])

    monkeypatch.setattr("sedkit.gaia._discover", discover)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="first batch failed"):
        download_gaia(**dict(options(service, tmp_path), workers=2))
    assert time.monotonic() - start >= 0.25

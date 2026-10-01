# Gaia batch-I/O acceptance

This harness exercises **unchanged** `query_gaia` and `download_gaia`. It adds
no retry, calibration, transport, or availability policy to the public APIs.
Use Python 3.10+ on Linux (RSS uses Linux `ru_maxrss` units).

```bash
python -m pip install -e '.[test]'
PYTHONPATH=src:. python -m pytest tests/test_gaia_acceptance.py -q -rx
PYTHONPATH=src:. python -m benchmarks.gaia_io --mode local \
  --sizes 100 1000 10000 --out data/bench-local
PYTHONPATH=src:. python -m benchmarks.gaia_io --mode live \
  --sizes 100 1000 10000 --out data/bench-live --deadline 900
```

The live default selects sources advertising all three DR3 products, to avoid
confusing sparse availability with high download throughput. It uses ARI TAP
and ESA DataLink; this is a **DR3 transport baseline**, not a DR4 validation or
a representative scientific parent sample. The flags use quoted `'True'`
because ARI rejects an unquoted ADQL `true`. Sample IDs are saved as exact
integers. Queries use `TOP n`, `MAXREC n+1`, and check count and uniqueness.
No claim about completeness beyond that deliberately bounded sample is made.

Re-run the **same live command and directory** after an interruption: saved
TAP jobs and completed batches are reused. Every attempt gets a separate JSON
and event log. A new directory gives a cold client-cache repetition; it cannot
force a cold server cache. Local fixtures use an ephemeral port, so use a fresh
directory for each local invocation. Do not run two calls in the same cache
directory. `--deadline` is a per-call hard process limit, not a whole-run budget.
Killing a process keeps partial files/job URLs; it does not cancel remote jobs.

The benchmark default product timeout is `(30, 300)`, matching the public API;
`--connect-timeout` and `--read-timeout` allow a bounded diagnostic. TAP result
downloads use `(10, 60)` and job control remains managed by PyVO. A hard deadline
also bounds control requests that lack a requests timeout. Increasing `--wait`
changes the API polling budget. Record timeout differences when comparing runs.

Other useful cases:

```bash
# All three products in ONE API call; discovery is shared across products.
PYTHONPATH=src:. python -m benchmarks.gaia_io --mode live --combined \
  --sizes 100 1000 --out data/bench-combined --deadline 900

# Mixed availability, independent of the all-products cohort.
PYTHONPATH=src:. python -m benchmarks.gaia_io --mode live --combined --mixed \
  --where 'parallax >= 10 AND phot_g_mean_mag < 15' \
  --sizes 100 --out data/bench-mixed --deadline 900

# Batch-size comparison on IDENTICAL source IDs. Use a fresh cache per setting.
PYTHONPATH=src:. python -m benchmarks.gaia_io --mode live --sizes 1000 \
  --ids-file data/bench-live/ids-1000.json --batch-size 200 --workers 2 \
  --out data/bench-b200-w2 --deadline 900

# Force a real server-side overflow (expected error, no completed catalogue).
PYTHONPATH=src:. python -m benchmarks.gaia_io --mode live --query-only \
  --sizes 100 --maxrec 99 --out data/bench-overflow --deadline 180
```

For DR4, explicitly supply a verified `--table`, `--release`, `--where`, and
`--products` after consulting released metadata. Do not merely replace DR3 with
DR4 in an unattended production script. `--mixed` disables the all-available
expectation; it cannot prove that absent discovery links mean unavailable data.

## Recorded metrics and interpretation

Each call produces JSON plus flushed JSONL events, including:

- Mode, operation, IDs path, ADQL, endpoint, release, batch size, workers, cache
  path, timeout and saved environment versions/source hashes.
- API wall time (excludes interpreter startup), rows or delivered/unavailable
  counts, per-product counts, cached batches, complete ZIP bytes, peak client RSS.
- HTTP status, `Retry-After`, requested timeout, URL length, request-to-response
  latency; stream bytes and durations; VOTable parsing and FITS validation times.
- Stage p50/p95, total successful stream bytes, product bytes, sources/s and
  MiB/s, complete ZIP/receipt counts and leftover partial files after failure.

Cold and warm calls each use a fresh Python process. Successful/error call
timings start inside the worker; deadline rows instead report parent-observed
process time, including startup and termination.

`http_headers` is request-to-response time: for `stream=True` this is time to
headers, but PyVO's non-streaming requests may include response body transfer.
It includes proxy, network and service latency; it is **not server CPU time**.
Stage sums overlap/nest and concurrent worker times overlap; do not add them
to estimate wall time. Streaming includes body transfer and disk writes.
`successful_stream_bytes` excludes failed/unfinished streams; partial bytes are
reported separately. No estimate of total wire bytes/retransmissions is made.
For combined runs, `delivered` counts source-product pairs; consult `by_product`.
Failed/deadline runs do not have a successful source throughput. Warm-cache
source rates describe local reuse, not network throughput.

The local service exercises actual Requests, PyVO, VOTable, FITS, ZIP and disk
paths, but generates illustrative synthetic flux arrays in the server process.
Its throughput **must never be presented as Gaia or DR4 throughput**. Client
RSS excludes fixture-server memory. RAW and INDIVIDUAL source-ID extraction
are covered, but the fixture does not reproduce all scientific schemas.

## Fault coverage and gates

The acceptance tests inject HTTP 429/503 with `Retry-After`, 414, disconnects,
corrupt ZIPs, advertised-but-missing and unexpected IDs, explicit DataLink and
TAP overflow, TAP polling interruption, expired jobs, silent discovery omission,
damaged cache/receipt, unknown products and failure while another worker runs.
Recovery checks ZIP hash **and mtime**, so a successful re-download cannot
masquerade as cache reuse. 100/1k/10k cases verify exact delivered/unavailable
partitions and zero HTTP requests on warm reuse. Server throttling is simulated;
we do not deliberately overload the public archive to induce 429 responses.

Strict xfails are **unresolved production risks**, not acceptance passes:
silent discovery omission, damaged ZIP, mismatched receipt, and the interval
between successful remote job creation and local job-URL persistence.
Automatic transport retries/backoff are absent by design in the tested API;
manual repeat is tested. Do not blindly retry all errors: 400/414/overflow,
schema mismatch and invalid IDs require correction, whereas transient failures
need bounded backoff with jitter and `Retry-After` handling in an outer runner
or a separately reviewed internal change.

Before unattended DR4 operation, require all of:

1. Live 100 and 1k delivery for each needed product, no unexplained loss versus
   catalogue availability; a representative 10k run or an explicit larger-scale
   operational limit. Repeat cold runs to quantify run-to-run variability.
2. A verified day-1 schema/product-name contract and source/epoch integrity
   audits. Source-ID equality alone cannot detect missing epochs/columns.
3. Transient retry/backoff, finite total runtime, controlled concurrency and an
   explicit procedure for expired/orphaned TAP jobs.
4. Cache integrity and discovery coverage controls; suspicious all-unavailable
   results must stop science processing instead of becoming non-detections.

There is no universal sources/s pass threshold without a campaign size and
deadline. Estimate runtime from product-specific successful rates and observed
retry overhead on the actual deployment network; do not extrapolate a two-source
check to DR4 bulk capability.

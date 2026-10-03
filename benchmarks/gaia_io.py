"""Run: python -m benchmarks.gaia_io --mode live --out data/gaia-bench.

No retry policy is added to sedkit: every attempt calls the unchanged public API.
Each call runs in a separate process with a hard deadline (including TAP control).
Use the same output directory to resume; a new directory for a cold repetition.
"""

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import platform
import resource
import subprocess
import threading
import time
from unittest.mock import patch
from urllib.parse import urlsplit

import numpy as np
import requests

from sedkit import gaia
from .gaia_service import BASE_ID, PRODUCTS, gaia_service


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


class Metrics:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.events = []

    def emit(self, event):
        with self.lock:
            self.events.append(event)
            with self.path.open("a") as handle:
                handle.write(json.dumps(event) + "\n")

    def stage(self, name, fn):
        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            event = {"stage": name}
            try:
                result = fn(*args, **kwargs)
                if name == "stream":
                    event.update(bytes=Path(result).stat().st_size,
                                 file=Path(args[1]).name)
                return result
            except Exception as exc:
                event["error"] = type(exc).__name__
                raise
            finally:
                event["seconds"] = time.perf_counter() - start
                self.emit(event)
        return wrapped

    def http(self, original):
        def wrapped(session, method, url, **kwargs):
            start = time.perf_counter()
            event = {"stage": "http_headers", "method": method,
                     "endpoint": urlsplit(url).path, "timeout": kwargs.get("timeout"),
                     "url_bytes": len(requests.Request(method, url, params=kwargs.get("params")).prepare().url.encode())}
            try:
                response = original(session, method, url, **kwargs)
                event.update(status=response.status_code, retry_after=response.headers.get("Retry-After"),
                             url_bytes=len(response.request.url.encode()),
                             content_length=response.headers.get("Content-Length"))
                return response
            except Exception as exc:
                event["error"] = type(exc).__name__
                raise
            finally:
                event["seconds"] = time.perf_counter() - start
                self.emit(event)
        return wrapped


def worker(case, result_path):
    result_path = Path(result_path)
    metrics = Metrics(result_path.with_suffix(".events.jsonl"))
    start = time.perf_counter()
    result = {"case": case, "status": "error"}
    with ExitStack() as stack:
        for name, stage in [("_stream", "stream"), ("_votable", "parse_votable"),
                            ("_archive_ids", "validate_fits_zip")]:
            stack.enter_context(patch.object(gaia, name, metrics.stage(stage, getattr(gaia, name))))
        stack.enter_context(patch.object(requests.Session, "request", metrics.http(requests.Session.request)))
        try:
            if case["kind"] == "query":
                table = gaia.query_gaia(case["query"], tap=case["tap"], maxrec=case.get("maxrec") or case["n"] + 1,
                                        wait=case["wait"], timeout=(10, 60), cache_dir=case["cache"])
                ids = [int(v) for v in table["source_id"]]
                save(case["ids_file"], ids)
                result.update(rows=len(ids), distinct_ids=len(set(ids)))
                if len(ids) != case["n"] or len(set(ids)) != len(ids):
                    raise RuntimeError("catalogue count/uniqueness acceptance check failed")
            else:
                ids = json.loads(Path(case["ids_file"]).read_text())
                products = PRODUCTS if case["product"] == "ALL" else [case["product"]]
                rows = gaia.download_gaia(ids, products=products, cache_dir=case["cache"],
                                          batch_size=case["batch_size"], workers=case["workers"],
                                          datalink_url=case["datalink"], data_release=case["release"],
                                          data_structure=case["structure"],
                                          timeout=(case.get("connect_timeout", 10), case.get("read_timeout", 90)))
                delivered = [sid for row in rows for sid in row["source_ids"]]
                missing = [sid for row in rows for sid in row["unavailable_source_ids"]]
                by_product = {}
                for product in products:
                    found = [sid for row in rows if row["product"] == product for sid in row["source_ids"]]
                    absent = [sid for row in rows if row["product"] == product for sid in row["unavailable_source_ids"]]
                    if (set(found) & set(absent) or set(found) | set(absent) != set(ids)
                            or len(found) + len(absent) != len(ids)):
                        raise RuntimeError("delivered/unavailable partition is inconsistent")
                    by_product[product] = {"delivered": len(found), "unavailable": len(absent)}
                result.update(delivered=len(delivered), unavailable=len(missing),
                              by_product=by_product,
                              cached_batches=sum(row["cached"] for row in rows), batches=len(rows),
                              zip_bytes=sum(row["path"].stat().st_size for row in rows if row["path"]))
                if case.get("expect_all_available") and missing:
                    raise RuntimeError("catalogue advertises product but DataLink reports unavailable")
            result["status"] = "ok"
        except Exception as exc:
            # Limit error size: requests exceptions can echo a whole batch URL.
            result["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    result["seconds"] = time.perf_counter() - start
    result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    save(result_path, result)


def run_case(case, out, deadline):
    path = out / f"{case['name']}.json"
    # Keep earlier attempts intact, so resumes do not overwrite failure evidence.
    attempt = 1
    while path.exists():
        attempt += 1
        path = out / f"{case['name']}.attempt-{attempt}.json"
    process = mp.get_context("spawn").Process(target=worker, args=(case, str(path)))
    start = time.perf_counter()
    process.start()
    process.join(deadline)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        save(path, {"case": case, "status": "deadline", "seconds": time.perf_counter() - start,
                    "note": "Local process stopped; saved TAP job and completed batches retained."})
    if not path.exists():
        save(path, {"case": case, "status": "crash", "exitcode": process.exitcode})
    result = json.loads(path.read_text())
    events_path = path.with_suffix(".events.jsonl")
    events, incomplete_events = [], 0
    for line in events_path.read_text().splitlines() if events_path.exists() else []:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            incomplete_events += 1  # A hard kill can interrupt the final log write.
    stages = {}
    for stage in sorted({e["stage"] for e in events}):
        selected = [e for e in events if e["stage"] == stage]
        times = [e["seconds"] for e in selected]
        stages[stage] = {"calls": len(times), "sum_s": sum(times),
                         "p50_s": float(np.percentile(times, 50)), "p95_s": float(np.percentile(times, 95))}
    http = [e for e in events if e["stage"] == "http_headers"]
    transferred = sum(e.get("bytes", 0) for e in events if e["stage"] == "stream")
    payload_bytes = sum(e.get("bytes", 0) for e in events if e.get("file", "").endswith(".zip"))
    seconds = result.get("seconds", 0)
    cached_ids = 0
    for receipt in Path(case["cache"]).glob("*.json"):
        if receipt.with_suffix(".zip").exists():
            cached_ids += len(json.loads(receipt.read_text())["source_ids"])
    result.update(stages=stages, incomplete_event_lines=incomplete_events,
                  http_statuses=dict(Counter(str(e.get("status", "exception")) for e in http)),
                  max_url_bytes=max([e.get("url_bytes", 0) for e in http], default=0),
                  successful_stream_bytes=transferred,
                  payload_stream_bytes=payload_bytes, receipt_source_count=cached_ids,
                  completed_zip_files=len(list(Path(case["cache"]).glob("*.zip"))),
                  leftover_parts=len(list(Path(case["cache"]).glob("*.part"))),
                  partial_file_bytes=sum(p.stat().st_size for p in Path(case["cache"]).glob("*.part")),
                  delivered_per_s=(result.get("delivered", result.get("rows", 0)) / seconds
                                   if seconds and result["status"] == "ok" else None),
                  payload_mib_per_wall_s=payload_bytes / 2**20 / seconds if seconds else None,
                  stream_mib_per_wall_s=transferred / 2**20 / seconds if seconds else None)
    save(path, result)
    print(json.dumps({"case": case["name"], "status": result["status"], "seconds": result.get("seconds"),
                      "delivered": result.get("delivered"), "unavailable": result.get("unavailable"),
                      "error": result.get("error")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "live"), default="local")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--products", nargs="+", default=list(PRODUCTS))
    parser.add_argument("--combined", action="store_true", help="One public-API call for all three products; share discovery")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--deadline", type=float, default=300)
    parser.add_argument("--connect-timeout", type=float, default=30)
    parser.add_argument("--read-timeout", type=float, default=300)
    parser.add_argument("--wait", type=float, default=120)
    parser.add_argument("--maxrec", type=int, help="Override row ceiling; smaller than size tests OVERFLOW rejection")
    parser.add_argument("--tap", default="ari")
    parser.add_argument("--release", default="Gaia DR3")
    parser.add_argument("--table", default="gaiadr3.gaia_source")
    parser.add_argument("--where", default="has_xp_continuous = 'True' AND has_rvs = 'True' AND has_epoch_photometry = 'True'")
    parser.add_argument("--mixed", action="store_true", help="Do not require all products; use --where for parent selection")
    parser.add_argument("--structure", choices=("RAW", "INDIVIDUAL"), default="RAW")
    parser.add_argument("--query-only", action="store_true")
    parser.add_argument("--ids-file", type=Path, help="Use a previously selected master sample; skip queries")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / f"environment-{time.time_ns()}.json", {"mode": args.mode, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "python": platform.python_version(), "platform": platform.platform(),
         "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
         "production_sha256": hashlib.sha256(Path(gaia.__file__).read_bytes()).hexdigest(),
         "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         "dependencies": {p: importlib.metadata.version(p) for p in ["numpy", "astropy", "pyvo", "requests"]},
         "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
    with ExitStack() as stack:
        service = stack.enter_context(gaia_service()) if args.mode == "local" else None
        tap = service.url + "/tap" if service else args.tap
        datalink = service.url + "/links" if service else gaia.DATALINK
        for n in args.sizes:
            ids_file = args.out / f"ids-{n}.json"
            base = {"n": n, "tap": tap, "wait": args.wait, "maxrec": args.maxrec,
                    "ids_file": str(ids_file), "mode": args.mode}
            if args.ids_file:
                ids = json.loads(args.ids_file.read_text())[:n]
                if len(ids) != n:
                    raise ValueError("master sample is shorter than requested size")
                save(ids_file, ids)
            else:
                query = f"SELECT TOP {n} source_id FROM {args.table} WHERE {args.where}"
                case = dict(base, kind="query", query=query, cache=str(args.out / f"query-{n}"), name=f"query-{n}-cold-or-resume")
                result = run_case(case, args.out, args.deadline)
                if result["status"] != "ok":
                    continue
                run_case(dict(case, name=f"query-{n}-warm"), args.out, args.deadline)
            if args.query_only:
                continue
            for product in (["ALL"] if args.combined else args.products):
                case = dict(base, kind="download", product=product, datalink=datalink, release=args.release,
                            structure=args.structure, batch_size=args.batch_size, workers=args.workers,
                            connect_timeout=args.connect_timeout, read_timeout=args.read_timeout,
                            expect_all_available=not service and not args.mixed,
                            cache=str(args.out / f"{product}-{n}"), name=f"{product}-{n}-cold-or-resume")
                result = run_case(case, args.out, args.deadline)
                if result["status"] == "ok":
                    run_case(dict(case, name=f"{product}-{n}-warm"), args.out, args.deadline)


if __name__ == "__main__":
    main()

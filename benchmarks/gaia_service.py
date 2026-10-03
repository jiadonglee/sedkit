"""Loopback TAP/DataLink fault fixture. Synthetic data, never a Gaia speed estimate."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import re
import threading
import time
from urllib.parse import parse_qs, urlsplit
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from astropy.table import Table

PRODUCTS = ("XP_CONTINUOUS", "RVS", "EPOCH_PHOTOMETRY")
BASE_ID = 4318465066420528000


def vot(table, overflow=False):
    out = BytesIO()
    table.write(out, format="votable")
    raw = out.getvalue()
    if overflow:
        raw = raw.replace(b"</RESOURCE>", b'<INFO name="QUERY_STATUS" value="OVERFLOW"/></RESOURCE>')
    return raw


def available(sid, product):
    return (sid - BASE_ID) % {"XP_CONTINUOUS": 5, "RVS": 3, "EPOCH_PHOTOMETRY": 2}[product] != 0


def zip_bytes(ids, product, individual=False):
    # Deliberately synthetic float arrays; sizes are illustrative, not a DR4 schema.
    width = {"XP_CONTINUOUS": 110, "RVS": 2401, "EPOCH_PHOTOMETRY": 64}[product]
    rng = np.random.default_rng(42)
    out = BytesIO()
    with ZipFile(out, "w", compression=ZIP_DEFLATED) as archive:
        for group in ([[sid] for sid in ids] if individual else [ids]):
            table = Table({"source_id": np.array(group, dtype="int64"),
                           "flux": rng.normal(size=(len(group), width)).astype("float32")})
            buffer = BytesIO()
            table.write(buffer, format="fits")
            archive.writestr(f"{product}-{group[0]}.fits", buffer.getvalue())
    return out.getvalue()


class Service(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server_port}"
        self.jobs = {}
        self.events = []
        self.lock = threading.Lock()
        self.fault = None
        self.fault_route = "/data"
        self.fault_count = 0
        self.fault_after = 0
        self.hold = False

    def inject(self, fault, route="/data", count=1, after=0):
        self.fault, self.fault_route, self.fault_count = fault, route, count
        self.fault_after = after


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, body=b"", status=200, headers=None):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        form = parse_qs(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode())
        self.server.events.append({"method": "POST", "path": self.path})
        if self.path == "/tap/async":
            key = str(len(self.server.jobs) + 1)
            n = int(re.search(r"TOP\s+(\d+)", form["QUERY"][0], re.I)[1])
            self.server.jobs[key] = {"n": n, "maxrec": int(form.get("MAXREC", [n])[0]), "phase": "PENDING"}
            return self.reply(status=303, headers={"Location": f"{self.server.url}/tap/async/{key}"})
        if self.path.endswith("/phase"):
            key = self.path.split("/")[-2]
            self.server.jobs[key]["phase"] = "EXECUTING" if self.server.hold else "COMPLETED"
            return self.reply()
        self.reply(status=404)

    def do_GET(self):
        parts = urlsplit(self.path)
        path, params = parts.path, parse_qs(parts.query)
        with self.server.lock:
            fault = None
            if path.startswith(self.server.fault_route) and self.server.fault_count:
                if self.server.fault_after:
                    self.server.fault_after -= 1
                else:
                    fault = self.server.fault
                    self.server.fault_count -= 1
            self.server.events.append({"method": "GET", "path": path,
                                       "url_bytes": len(self.path.encode()), "fault": fault})
        if fault in ("429", "503", "414", "404"):
            return self.reply(b"injected failure", int(fault), {"Retry-After": "1"})
        if fault == "slow":
            time.sleep(0.3)
        if path.startswith("/tap/async/"):
            key = path.split("/")[3]
            job = self.server.jobs[key]
            if path.endswith("/result"):
                ids = np.arange(BASE_ID + 1, BASE_ID + 1 + min(job["n"], job["maxrec"]), dtype="int64")
                return self.reply(vot(Table({"SOURCE_ID": ids}), job["n"] > job["maxrec"]))
            xml = f'''<uws:job xmlns:uws="http://www.ivoa.net/xml/UWS/v1.0" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.0">
              <uws:jobId>{key}</uws:jobId><uws:phase>{job['phase']}</uws:phase>
              <uws:executionDuration>600</uws:executionDuration>
              <uws:results><uws:result id="result" xlink:href="{self.server.url}/tap/async/{key}/result"/></uws:results>
            </uws:job>'''
            return self.reply(xml.encode(), headers={"Content-Type": "application/xml"})
        ids = [int(s.split()[-1]) for s in params.get("ID", [""])[0].split(",") if s]
        if path == "/links":
            rows = [(f"Gaia DR3 {sid}", f"{self.server.url}/data?RETRIEVAL_TYPE={product}", "")
                    for sid in ids for product in PRODUCTS if available(sid, product)]
            if fault == "silent_omission":
                rows = [row for row in rows if row[0] != f"Gaia DR3 {ids[0]}"]
            return self.reply(vot(Table(rows=rows, names=("ID", "access_url", "error_message")),
                                  overflow=fault == "overflow"))
        if path == "/data":
            if fault == "missing_id":
                ids = ids[:-1]
            if fault == "unexpected_id":
                ids = ids + [BASE_ID + 999999]
            if fault == "corrupt_zip":
                return self.reply(b"not a zip")
            raw = zip_bytes(ids, params["RETRIEVAL_TYPE"][0], params.get("DATA_STRUCTURE") == ["INDIVIDUAL"])
            if fault == "disconnect":
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw[:len(raw)//2])
                self.close_connection = True
                return
            return self.reply(raw)
        self.reply(status=404)


@contextmanager
def gaia_service():
    service = Service()
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    try:
        yield service
    finally:
        service.shutdown()
        service.server_close()
        thread.join()

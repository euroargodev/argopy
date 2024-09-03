"""
Fixture for a local HTTP server to be used by all http requests of unit tests

For this to work, we need to catch all the possible URI sent from unit tests and return some data (eg json or netcdf)
we downloaded once before tests.

This requires to download data using fully functional http servers: this is done by the "citests_httpdata_manager" located
in "argopy/cli"

Servers covered by this fixture:
            "https://github.com/euroargodev/argopy-data/raw/master",
            "https://erddap.ifremer.fr/erddap",
            "https://data-argo.ifremer.fr",
            "https://api.ifremer.fr",
            "https://coastwatch.pfeg.noaa.gov/erddap",
            "https://www.ocean-ops.org/api/1",
            "https://dataselection.euro-argo.eu/api",
            "https://vocab.nerc.ac.uk/collection",
            "https://argovis-api.colorado.edu",
            "https://dx.doi.org",
            "https://archimer.ifremer.fr",

The HTTPTestHandler class below is taken from the fsspec tests suite at:
https://github.com/fsspec/filesystem_spec/blob/55c5d71e657445cbfbdba15049d660a5c9639ff0/fsspec/tests/conftest.py

"""
import contextlib
from pathlib import Path
import threading
from collections import ChainMap
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest
import logging
from urllib.parse import unquote
import socket
import json
import importlib


log = logging.getLogger("argopy.tests.mocked_http")
LOG_SERVER_CONTENT = (
    False  # Should we list all files/uris available from the mocked server in the log ?
)

requests = pytest.importorskip("requests")
port = 9898  # Select the port to run the local server on
mocked_server_address = "http://127.0.0.1:%i" % port

start_with = (
    lambda f, x: f[0 : len(x)] == x if len(x) <= len(f) else False
)  # noqa: E731

"""
Load test data and create a dictionary mapping of URL requests as keys, and expected responses as values

This will be filling the mocked http server content.
The uri requested will be made relative, because the server name will be replaced by the mocked server address.
"""
MOCKED_REQUESTS = {}
TESTDATA_FOLDER = (
    Path(importlib.util.find_spec("argopy.tests").submodule_search_locations[0])
    .resolve()
    .joinpath("test_data")
)
if not TESTDATA_FOLDER.exists():
    raise RuntimeError(
        "Can't find tests data folder at: %s\n Note that test data are not included in the pypi distribution. You should fork the repo to get test data."
        % TESTDATA_FOLDER
    )

DB_FILE = TESTDATA_FOLDER.joinpath("httpmocked_uri_index.json")
URI = []

if DB_FILE.exists():
    with open(DB_FILE, "r") as f:
        URI = json.load(f)
    for resource in URI:
        test_data_file = TESTDATA_FOLDER.joinpath(
            "%s.%s" % (resource["sha"], resource["ext"])
        )
        with open(test_data_file, mode="rb") as file:
            data = file.read()

        # Remove all specific api/server names from absolute URIs
        # Because these are arguments passed to methods that will use mocked_server_address instead
        # (See for instance the argument 'server' in `argopy.data_fetchers.erddap_data.ErddapArgoDataFetcher`)
        patterns = [
            "https://github.com/euroargodev/argopy-data/raw/master",
            "https://erddap.ifremer.fr/erddap",
            "https://data-argo.ifremer.fr",
            "https://api.ifremer.fr",
            "https://coastwatch.pfeg.noaa.gov/erddap",
            "https://www.ocean-ops.org/api/1",
            "https://dataselection.euro-argo.eu/api",
            "https://vocab.nerc.ac.uk/collection",
            "https://argovis-api.colorado.edu",
            # "https://argovisbeta02.colorado.edu",
            "https://dx.doi.org",
            "https://archimer.ifremer.fr",
        ]
        for pattern in patterns:
            if start_with(resource["uri"], pattern):
                MOCKED_REQUESTS[resource["uri"].replace(pattern, "")] = data

else:
    raise RuntimeError(
        "Can't find test data index file at: %s.\n Note that test data are not included in the pypi distribution. You should fork the repo to get test data."
        % DB_FILE
    )


def get_html_landing_page():
    """Return a webpage with a listing of all available files with a href links"""
    html = ["<html><head></head><body>\n"]
    html.append(
        "<h1>Mocked HTTP server is up and running, serving %i files</h1>" % len(URI)
    )
    html.append("<ul>")
    for key, value in MOCKED_REQUESTS.items():
        html.append("<li><a href='%s'>%s</a></li>" % (key, key))
    html.append("</ul>")
    html.append("</body></html>")
    return str.encode("\n".join(html))


class HTTPTestHandler(BaseHTTPRequestHandler):
    static_files = {
        "": get_html_landing_page(),
    }
    dynamic_files = {}

    files = ChainMap(dynamic_files, static_files, MOCKED_REQUESTS)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # log.debug("HTTPTestHandler instance created with %i files at %s" % (len(self.files), self.client_address))

    def _respond(self, code=200, headers=None, data=b""):
        headers = headers or {}
        headers.update({"User-Agent": "Mocked http server for unit tests"})
        self.send_response(code)
        for k, v in headers.items():
            self.send_header(k, str(v))
        self.end_headers()
        if data:
            try:
                self.wfile.write(data)
            except socket.error as e:
                # socket error [Errno 32] Broken pipe
                # This might be happening when a client program doesn't wait till all the data from the server is
                # received and simply closes a socket
                if "32" not in str(e):
                    log.debug("socket error %s" % str(e))
                pass

    def log_message(self, format, *args):
        # Quiet logging !
        return

    def do_GET(self):
        file_path = unquote(self.path.rstrip("/"))
        # log.debug("Requesting: '%s'" % file_path)
        # log.debug("Found: %s" % (file_path in self.files.keys()))
        # log.debug("Found B: %s" % (file_path in MOCKED_REQUESTS.keys()))

        # [log.debug("\t└─ '%s'" % k) for k in self.files.keys()]

        file_data = self.files.get(file_path)
        if "give_path" in self.headers:
            return self._respond(200, data=json.dumps({"path": self.path}).encode())
        if file_data is None:
            file_data = self.files.get(self.path.rstrip("/"))  # try without unquoting
            if file_data is None:
                return self._respond(404)

        n = len(file_data)
        status = 200
        content_range = "bytes 0-%i/%i" % (n - 1, n)
        if ("Range" in self.headers) and ("ignore_range" not in self.headers):
            ran = self.headers["Range"]
            b, ran = ran.split("=")
            start, end = ran.split("-")
            if start:
                content_range = f"bytes {start}-{end}/{n}"
                file_data = file_data[int(start) : (int(end) + 1) if end else None]
            else:
                # suffix only
                content_range = f"bytes {n - int(end)}-{n - 1}/{n}"
                file_data = file_data[-int(end) :]
            if "use_206" in self.headers:
                status = 206
        if "give_length" in self.headers:
            response_headers = {"Content-Length": n}
            self._respond(status, response_headers, file_data)
        elif "give_range" in self.headers:
            self._respond(status, {"Content-Range": content_range}, file_data)
        else:
            self._respond(status, data=file_data)

    def do_POST(self):
        length = self.headers.get("Content-Length")
        file_path = unquote(self.path.rstrip("/"))
        if length is None:
            assert self.headers.get("Transfer-Encoding") == "chunked"
            self.files[file_path] = b"".join(self.read_chunks())
        else:
            self.files[file_path] = self.rfile.read(length)
        self._respond(200)

    do_PUT = do_POST

    def read_chunks(self):
        length = -1
        while length != 0:
            line = self.rfile.readline().strip()
            if len(line) == 0:
                length = 0
            else:
                length = int(line, 16)
            yield self.rfile.read(length)
            self.rfile.readline()

    def do_HEAD(self):
        self.headers.add_header("head_ok", "")
        self.headers.add_header("give_length", "")

        if "head_not_auth" in self.headers:
            return self._respond(
                403, {"Content-Length": 123}, b"not authorized for HEAD request"
            )
        elif "head_ok" not in self.headers:
            return self._respond(405)

        file_path = unquote(self.path.rstrip("/"))
        file_data = self.files.get(file_path)
        if file_data is None:
            return self._respond(404)
        else:
            n = len(file_data)

        if ("give_length" in self.headers) or ("head_give_length" in self.headers):
            response_headers = {"Content-Length": n}
            if "zero_length" in self.headers:
                response_headers["Content-Length"] = 0

            self._respond(200, response_headers)
        elif "give_range" in self.headers:
            self._respond(
                200, {"Content-Range": "0-%i/%i" % (n - 1, n)}
            )
        elif "give_etag" in self.headers:
            self._respond(200, {"ETag": "xxx"})
        else:
            self._respond(200)  # OK response, but no useful info


@contextlib.contextmanager
def serve_mocked_httpserver():
    server_address = ("", port)
    httpd = HTTPServer(server_address, HTTPTestHandler)
    th = threading.Thread(target=httpd.serve_forever)
    th.daemon = True
    th.start()
    try:
        log.info(
            "Mocked HTTP server up and ready at %s, serving %i URI. (id=%s)"
            % (mocked_server_address, len(HTTPTestHandler.files), id(httpd))
        )
        if LOG_SERVER_CONTENT:
            # Use these lines to log test data name and content
            for f in HTTPTestHandler.files.keys():
                log.info(f)
                log.info("└─ %s" % HTTPTestHandler.files[f][0:10])
        yield mocked_server_address
    finally:
        httpd.socket.close()
        httpd.shutdown()
        th.join()
        log.info("Teardown mocked HTTP server with id=%s" % id(httpd))


@pytest.fixture(scope="module")
def mocked_httpserver():
    with serve_mocked_httpserver() as s:
        yield s

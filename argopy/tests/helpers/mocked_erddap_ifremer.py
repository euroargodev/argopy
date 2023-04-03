"""
Mock the Ifremer erddap server

For this to work, we need to catch all the possible erddap URI sent from unit tests and return some json or netcdf data
we downloaded once before tests.

This requires to download data using a fully functional erddap server: this is done by this script when executed from
the command line.

The HTTPTestHandler class below is taken from the fsspec tests suite at:
https://github.com/fsspec/filesystem_spec/blob/55c5d71e657445cbfbdba15049d660a5c9639ff0/fsspec/tests/conftest.py

"""
import numpy as np
import contextlib
import json
import os
import sys
import threading
from collections import ChainMap
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest
import logging
from urllib.parse import unquote
import socket
import pickle
import xarray as xr
import hashlib
from urllib.parse import urlparse, parse_qs

log = logging.getLogger("argopy.tests.mocked_erddap")

requests = pytest.importorskip("requests")
port = 9898  # Select the port to run the local server on
mocked_server_address = "http://127.0.0.1:%i" % port

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from argopy.data_fetchers.erddap_data import api_server


# Create the dictionary mapping of URL requests as keys, and expected responses as values:
# (this will be filling the mocked http server content)
ERDDAP_FILES = {}
DATA_FOLDER = os.path.dirname(os.path.realpath(__file__)).replace("helpers", "test_data")
DB_FILE = os.path.join(DATA_FOLDER, "erddap_file_index.pkl")
if os.path.exists(DB_FILE):
    with open(os.path.join(DATA_FOLDER, "erddap_file_index.pkl"), "rb") as f:
        URI = pickle.load(f)
    for ressource in URI:
        test_data_file = os.path.join(DATA_FOLDER, "%s.%s" % (ressource['sha'], ressource['ext']))
        with open(test_data_file, mode='rb') as file:
            data = file.read()
        ERDDAP_FILES[ressource['uri'].replace(api_server, '')] = data
else:
    log.debug("Loading this sub-module with DB_FILE %s" % DB_FILE)

class HTTPTestHandler(BaseHTTPRequestHandler):
    static_files = {
        # "/index/realfile": data,
        # "/index/otherfile": data,
        # "/index": index,
        # "/data/20020401": listing,
    }
    dynamic_files = {}

    files = ChainMap(dynamic_files, static_files, ERDDAP_FILES)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _respond(self, code=200, headers=None, data=b""):
        headers = headers or {}
        headers.update({"User-Agent": "test"})
        self.send_response(code)
        for k, v in headers.items():
            self.send_header(k, str(v))
        self.end_headers()
        if data:
            try:
                self.wfile.write(data)
            except socket.error as e:
                log.debug("socket error %s" % str(e))
                pass

    def log_message(self, format, *args):
        # Quiet logging !
        return

    def do_GET(self):
        file_path = unquote(self.path.rstrip("/"))
        # log.debug("Requesting: %s" % file_path)
        # [log.debug("└─ %s" % k) for k in self.files.keys()]
        file_data = self.files.get(file_path)
        if "give_path" in self.headers:
            return self._respond(200, data=json.dumps({"path": self.path}).encode())
        if file_data is None:
            # log.debug("file data empty, returning 404")
            return self._respond(404)

        status = 200
        content_range = "bytes 0-%i/%i" % (len(file_data) - 1, len(file_data))
        if ("Range" in self.headers) and ("ignore_range" not in self.headers):
            ran = self.headers["Range"]
            b, ran = ran.split("=")
            start, end = ran.split("-")
            if start:
                content_range = f"bytes {start}-{end}/{len(file_data)}"
                file_data = file_data[int(start): (int(end) + 1) if end else None]
            else:
                # suffix only
                l = len(file_data)
                content_range = f"bytes {l-int(end)}-{l-1}/{l}"
                file_data = file_data[-int(end):]
            if "use_206" in self.headers:
                status = 206
        if "give_length" in self.headers:
            response_headers = {"Content-Length": len(file_data)}
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

        if ("give_length" in self.headers) or ("head_give_length" in self.headers):
            response_headers = {"Content-Length": len(file_data)}
            if "zero_length" in self.headers:
                response_headers["Content-Length"] = 0

            self._respond(200, response_headers)
        elif "give_range" in self.headers:
            self._respond(
                200, {"Content-Range": "0-%i/%i" % (len(file_data) - 1, len(file_data))}
            )
        elif "give_etag" in self.headers:
            self._respond(200, {"ETag": "xxx"})
        else:
            self._respond(200)  # OK response, but no useful info


@contextlib.contextmanager
def serve():
    server_address = ("", port)
    httpd = HTTPServer(server_address, HTTPTestHandler)
    th = threading.Thread(target=httpd.serve_forever)
    th.daemon = True
    th.start()
    try:
        log.info("Mocked ERDDAP HTTP server up and ready at %s, serving %i URI. (id=%s)" %
                 (mocked_server_address, len(HTTPTestHandler.files), id(httpd)))
        # for f in HTTPTestHandler.files.keys():
        #     log.info(f)
        #     log.info("└─ %s" % HTTPTestHandler.files[f][0:3])
        yield mocked_server_address
    finally:
        httpd.socket.close()
        httpd.shutdown()
        th.join()
        log.info("Teardown mocked ERDDAP HTTP server with id=%s" % id(httpd))


@pytest.fixture(scope="module")
def mocked_erddapserver():
    with serve() as s:
        yield s


def can_be_xr_opened(src, file):
    try:
        xr.open_dataset(file)
        return src
    except:
        # print("This source can't be opened with xarray: %s" % src)
        return src


def log_file_desc(file, data, src):
    size = float(os.stat(file).st_size / (1024 * 1024))
    prt_size = lambda x: "< 1Mb" if x < 1 else "%0.2dMb" % x
    msg = []
    # msg.append("- %s, %s" % (file.replace(DATA_FOLDER, '<DATA_FOLDER>'), data[0:3]))
    msg.append("\n🤖 %s" % parse_qs(src['uri']))
    msg.append("%s, %s, %s" % (file.replace(DATA_FOLDER, '<DATA_FOLDER>'), data[0:3], prt_size(size)))
    return "\n".join(msg)


if __name__ == '__main__':
    import aiohttp
    import asyncio
    import aiofiles

    DATA_FOLDER = os.path.dirname(os.path.realpath(__file__)).replace("helpers", "test_data")
    print("Data will be saved in: %s" % DATA_FOLDER)

    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from argopy import DataFetcher

    async def fetch_download_links(session: aiohttp.ClientSession):
        # Get the list of URI to download.
        # This should correspond to all the possible erddap requests made during CI tests.
        # And because of the erddap fetcher N_POINT attribute, we also need to fetch ".ncHeader" on top of ".nc" files
        requests_phy = {
            "float": [[1901393], [1901393, 6902746]],
            "profile": [[6902746, 34], [6902746, np.arange(12, 13)], [6902746, [1, 12]]],
            "region": [
                [-20, -16., 0, 1, 0, 100.],
                # [-20, -16., 0, 1, 0, 100., "1997-07-01", "1997-09-01"]
                [-20, -16., 0, 1, 0, 100., "2004-01-01", "2004-01-31"]
            ],
        }
        requests_bgc = {
            "float": [[5903248], [7900596, 2902264]],
            "profile": [[5903248, 34], [5903248, np.arange(12, 14)], [5903248, [1, 12]]],
            "region": [
                [-70, -65, 35.0, 40.0, 0, 10.0],
                [-70, -65, 35.0, 40.0, 0, 10.0, "2012-01-1", "2012-12-31"],
            ],
        }
        requests_ref = {
            "region": [
                [-70, -65, 35.0, 40.0, 0, 10.0],
                [-70, -65, 35.0, 40.0, 0, 10.0, "2012-01-01", "2012-12-31"],
            ]
        }
        requests_par = {
            # "float": [[6902766, 6902772, 6902914]],  # >128Mb file !
            "float": [[1900468, 1900117, 1900386]],
            "region": [[-60, -55, 40.0, 45.0, 0.0, 20.0],
                       [-60, -55, 40.0, 45.0, 0.0, 20.0, "2007-08-01", "2007-09-01"]]
        }

        requests = {'phy': requests_phy, 'bgc': requests_bgc, 'ref': requests_ref}
        # requests = {'phy': requests_phy}

        def nc_file(url, sha, iter):
            return {
                'uri': url,
                'ext': "nc",
                'sha': "%s_%03.d" % (sha, iter),
                'type': 'application/x-netcdf',
            }

        def nc_normfile(url, sha, iter):
            return {
                'uri': url,
                'ext': "nc",
                'sha': "%s_%03.d_norm" % (sha, iter),
                'type': 'application/x-netcdf',
            }

        def ncHeader_file(url, sha, iter):
            return {
                'uri': url,
                'ext': "ncHeader",
                'sha': "%s_%03.d" % (sha, iter),
                'type': 'text/plain'
            }

        def ncHeader_normfile(url, sha, iter):
            return {
                'uri': url,
                'ext': "ncHeader",
                'sha': "%s_%03.d_norm" % (sha, iter),
                'type': 'text/plain'
            }

        def add_to_URI(URI, this_fetcher):
            uri = this_fetcher.uri
            for ii, url in enumerate(uri):
                URI.append(nc_file(url, this_fetcher.fetcher.sha, ii))
                # URI.append(nc_normfile(normalize_url(url), f.fetcher.sha, ii))

            url = this_fetcher.fetcher.get_url().replace("." + this_fetcher.fetcher.erddap.response, ".ncHeader")
            URI.append(ncHeader_file(url, this_fetcher.fetcher.sha, ii))
            # URI.append(ncHeader_normfile(normalize_url(url), f.fetcher.sha, ii))
            return URI

        URI = []
        for ds in requests:
            fetcher = DataFetcher(src='erddap', ds=ds)
            for access_point in requests[ds]:
                if access_point == 'profile':
                    for cfg in requests[ds][access_point]:
                        URI = add_to_URI(URI, fetcher.profile(*cfg))
                if access_point == 'float':
                    for cfg in requests[ds][access_point]:
                        URI = add_to_URI(URI, fetcher.float(cfg))
                if access_point == 'region':
                    for cfg in requests[ds][access_point]:
                        URI = add_to_URI(URI, fetcher.region(cfg))


        fetcher = DataFetcher(src='erddap', ds='phy', parallel=True)
        for access_point in requests_par:
            if access_point == 'profile':
                for cfg in requests_par[access_point]:
                    URI = add_to_URI(URI, fetcher.profile(*cfg))
            if access_point == 'float':
                for cfg in requests_par[access_point]:
                    URI = add_to_URI(URI, fetcher.float(cfg))
            if access_point == 'region':
                for cfg in requests_par[access_point]:
                    URI = add_to_URI(URI, fetcher.region(cfg))


        # Add more URI:
        URI.append({'uri': 'https://erddap.ifremer.fr/erddap/info/ArgoFloats/index.json',
                    'ext': 'json',
                    'sha': hashlib.sha256('https://erddap.ifremer.fr/erddap/info/ArgoFloats/index.json'.encode()).hexdigest(),
                    'type': 'application/json',
                    })

        #
        return URI

    async def place_file(session: aiohttp.ClientSession, source: dict) -> None:
        async with session.get(source['uri'], ssl=False) as r:
            if r.content_type not in ['application/x-netcdf', 'text/plain', 'application/json']:
                print("Unexpected content type (%s) with this GET request: %s (%s extension)" %
                      (r.content_type, parse_qs(source['uri']), os.path.splitext(urlparse(source['uri']).path)[1]))

            test_data_file = os.path.join(DATA_FOLDER, "%s.%s" % (source['sha'], source['ext']))
            async with aiofiles.open(test_data_file, 'wb') as f:
                data = await r.content.read(n=-1)  # load all read bytes !
                await f.write(data)
                print(log_file_desc(f.name, data, source))
                return can_be_xr_opened(source, test_data_file)

    async def main():
        async with aiohttp.ClientSession() as session:
            urls = await fetch_download_links(session)
            return await asyncio.gather(*[place_file(session, url) for url in urls])

    # async def place_fileHEAD(session: aiohttp.ClientSession, source: dict) -> None:
    #     async with session.head(source['uri'], ssl=False) as r:
    #         if r.content_type not in ['application/x-netcdf', 'text/plain', 'application/json']:
    #             print("Unexpected content type (%s) with this HEAD request: %s (%s extension)" %
    #                   (r.content_type, parse_qs(source['uri']), os.path.splitext(urlparse(source['uri']).path)[1]))
    #         # else:
    #         #     print("\nLoading: %s" % source['uri'])
    #
    #         test_data_file = os.path.join(DATA_FOLDER, "%s.%s" % (source['sha'], source['ext']))
    #         async with aiofiles.open(test_data_file, 'wb') as f:
    #             data = await r.content.read(n=-1)  # load all read bytes !
    #             print(f.name, data[0:3])
    #             await f.write(data)
    #             return can_be_xr_opened(source, test_data_file)
    #
    # async def mainHEAD():
    #     async with aiohttp.ClientSession() as session:
    #         urls = await fetch_download_links(session)
    #         return await asyncio.gather(*[place_fileHEAD(session, url) for url in urls])

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    URLS = loop.run_until_complete(main())
    # [URLS.append(uri) for uri in loop.run_until_complete(mainHEAD())]

    # URLS = [url for url in URLS if url is not None]
    print("\nSaved %i urls" % len(URLS))

    # Save the URI list in a pickle to be loaded and used by the erddap mocker:
    with open(os.path.join(DATA_FOLDER, "erddap_file_index.pkl"), 'wb') as f:
        pickle.dump(URLS, f)

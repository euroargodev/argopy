import os
import warnings

import pytest
import tempfile

import numpy as np
import xarray as xr
import pandas as pd
import fsspec
from fsspec.registry import known_implementations
import aiohttp
import importlib
import shutil
import logging
from urllib.parse import urlparse
import ftplib

import argopy
from argopy.stores import (
    filestore,
    httpstore,
    memorystore,
    ftpstore,
    indexfilter_wmo,
    indexfilter_box,
    indexstore,
)
from argopy.stores.filesystems import new_fs
from argopy.options import OPTIONS
from argopy.errors import (
    FileSystemHasNoCache,
    CacheFileNotFound,
    FtpPathError, 
    InvalidMethod,
    DataNotFound,
    OptionValueError,
)
from argopy.utilities import (
    is_list_of_datasets,
    is_list_of_dicts,
    modified_environ,
    is_list_of_strings,
)
from argopy.stores.argo_index_pd import indexstore_pandas
from utils import requires_connection, requires_connected_argovis
from mocked_http import mocked_httpserver, mocked_server_address


log = logging.getLogger("argopy.tests.stores")

has_pyarrow = importlib.util.find_spec('pyarrow') is not None
skip_pyarrow = pytest.mark.skipif(not has_pyarrow, reason="Requires pyarrow")

skip_this = pytest.mark.skipif(0, reason="Skipped temporarily")
skip_for_debug = pytest.mark.skipif(False, reason="Taking too long !")

id_implementation = lambda x: [k for k, v in known_implementations.items()  # noqa: E731
                                      if x.__class__.__name__ == v['class'].split('.')[-1]]
is_initialised = lambda x: ((x is None) or (x == []))  # noqa: E731


@skip_this
class Test_new_fs:

    def test_default(self):
        fs, cache_registry = new_fs()
        assert id_implementation(fs) is not None
        assert is_initialised(cache_registry)

    def test_cache_type(self):
        fs, cache_registry = new_fs(cache=True)
        assert id_implementation(fs) == ['filecache']


@skip_this
@requires_connection
class Test_FileStore:
    ftproot = argopy.tutorial.open_dataset("gdac")[0]
    csvfile = os.path.sep.join([ftproot, "ar_index_global_prof.txt"])
    fs = filestore()  # Default filestore instance

    def test_implementation(self):
        fs = filestore(cache=False)
        assert isinstance(fs.fs, fsspec.implementations.local.LocalFileSystem)

    def test_nocache(self):
        fs = filestore(cache=False)
        with pytest.raises(FileSystemHasNoCache):
            fs.cachepath("dummy_uri")

    def test_cache(self):
        fs = filestore(cache=True)
        assert isinstance(fs.fs, fsspec.implementations.cached.WholeFileCacheFileSystem)

    def test_nocachefile(self):
        fs = filestore(cache=True)
        with pytest.raises(CacheFileNotFound):
            fs.cachepath("dummy_uri")

    def test_glob(self):
        assert isinstance(self.fs.glob(os.path.sep.join([self.ftproot, "dac/*"])), list)

    def test_open_dataset(self):
        ncfile = os.path.sep.join([self.ftproot, "dac/aoml/5900446/5900446_prof.nc"])
        assert isinstance(self.fs.open_dataset(ncfile), xr.Dataset)

    params = [(m, p, c) for m in ["seq", "thread", "process"] for p in [True, False] for c in [True, False]]
    ids_params = ["method=%s, progress=%s, concat=%s" % (p[0], p[1], p[2]) for p in params]
    @pytest.mark.parametrize("params", params, indirect=False, ids=ids_params)
    def test_open_mfdataset(self, params):
        uri = self.fs.glob(
            os.path.sep.join([self.ftproot, "dac/aoml/5900446/profiles/*_1*.nc"])
        )[0:2]
        method, progress, concat = params

        if method == "process":
            pytest.skip("concurrent.futures.ProcessPoolExecutor is too long on GA !")

        ds = self.fs.open_mfdataset(uri, method=method, progress='disable' if progress else False, concat=concat)
        if concat:
            assert isinstance(ds, xr.Dataset)
        else:
            assert is_list_of_datasets(ds)

    params = [(m) for m in ["seq", "thread", "invalid"]]
    ids_params = ["method=%s" % (p) for p in params]
    @pytest.mark.parametrize("params", params, indirect=False, ids=ids_params)
    def test_open_mfdataset_error(self, params):
        uri = self.fs.glob(
            os.path.sep.join([self.ftproot, "dac/aoml/5900446/profiles/*_1*.nc"])
        )[0:2]

        def preprocess(ds):
            """Fake preprocessor raising an error"""
            raise ValueError

        method = params
        if 'invalid' in method:
            err = InvalidMethod
        else:
            err = ValueError
        with pytest.raises(err):
            self.fs.open_mfdataset(uri, method=method, preprocess=preprocess, errors='raise')

        with pytest.raises(ValueError):
            self.fs.open_mfdataset(uri, method=method, preprocess=preprocess, errors="ignore")


    def test_open_mfdataset_DataNotFound(self):
        uri = self.fs.glob(
            os.path.sep.join([self.ftproot, "dac/aoml/5900446/profiles/*_1*.nc"])
        )[0:2]

        def preprocess(ds):
            """Fake preprocessor returning nothing"""
            return None

        with pytest.raises(DataNotFound):
            self.fs.open_mfdataset(uri, preprocess=preprocess)

    def test_read_csv(self):
        assert isinstance(
            self.fs.read_csv(self.csvfile, skiprows=8, header=0), pd.core.frame.DataFrame
        )

    def test_cachefile(self):
        with tempfile.TemporaryDirectory() as cachedir:
            fs = filestore(cache=True, cachedir=cachedir)
            fs.read_csv(self.csvfile, skiprows=8, header=0)
            assert isinstance(fs.cachepath(self.csvfile), str)

    def test_clear_cache(self):
        with tempfile.TemporaryDirectory() as cachedir:
            # Create dummy data to read and cache:
            uri = os.path.abspath("dummy_fileA.txt")
            with open(uri, "w") as fp:
                fp.write("Hello world!")
            # Create store:
            fs = filestore(cache=True, cachedir=cachedir)
            # Then we read some dummy data from the dummy file to trigger caching
            with fs.open(uri, "r") as fp:
                fp.read()
            assert isinstance(fs.cachepath(uri), str)
            assert os.path.isfile(fs.cachepath(uri))
            # Now, we can clear the cache:
            fs.clear_cache()
            # And verify it does not exist anymore:
            with pytest.raises(CacheFileNotFound):
                fs.cachepath(uri)
            os.remove(uri)  # Delete dummy file


@skip_this
class Test_HttpStore:
    repo = "https://github.com/euroargodev/argopy-data/raw/master"
    fs = httpstore(timeout=OPTIONS["api_timeout"])  # Default http file store

    # Parameters for multiple files opening
    mf_params_nc = [
        (m, p, c)
        for m in ["seq", "thread", "process"]
        for p in [True, False]
        for c in [True, False]
    ]
    mf_params_nc_ids = [
        "method=%s, progress=%s, concat=%s" % (p[0], p[1], p[2]) for p in mf_params_nc
    ]
    mf_nc = [
        repo + "ftp/dac/csiro/5900865/profiles/D5900865_001.nc",
        repo + "ftp/dac/csiro/5900865/profiles/D5900865_002.nc",
    ]

    mf_params_js = [(m, p) for m in ["seq", "thread", "process"] for p in [True, False]]
    mf_params_js_ids = ["method=%s, progress=%s" % (p[0], p[1]) for p in mf_params_js]
    mf_js = [
        "https://api.ifremer.fr/argopy/data/ARGO-FULL.json",
        "https://api.ifremer.fr/argopy/data/ARGO-BGC.json",
    ]

    #########
    # UTILS #
    #########
    def setup_class(self):
        """setup any state specific to the execution of the given class"""
        # Create the cache folder here, so that it's not the same for the pandas and pyarrow tests
        self.cachedir = tempfile.mkdtemp()

    def teardown_class(self):
        """Cleanup once we are finished."""

        def remove_test_dir():
            shutil.rmtree(self.cachedir)

        remove_test_dir()

    def _mockeduri(self, uri):
        """Replace real server by the mocked local server

        Simply return uri unchanged if you want to run tests with real servers.
        Otherwise, let this method take the tests offline with the mocked server.
        """
        for server_to_mock in [self.repo, "https://api.ifremer.fr/"]:
            if server_to_mock in uri:
                uri = uri.replace(server_to_mock, mocked_server_address + "/")
        return uri

    def test_load_mocked_server(self, mocked_httpserver):
        """This will easily ensure that the module scope fixture is available to all methods !"""
        assert True

    #########
    # TESTS #
    #########
    def test_implementation(self):
        fs = httpstore(cache=False)
        assert isinstance(fs.fs, fsspec.implementations.http.HTTPFileSystem)

    def test_trust_env(self):
        with modified_environ(HTTP_PROXY="http://dummy_proxy"):
            with argopy.set_options(trust_env=True):
                fs = httpstore(cache=False)
                with pytest.raises(aiohttp.client_exceptions.ClientConnectorError):
                    uri = self._mockeduri(
                        self.repo + "ftp/dac/csiro/5900865/5900865_prof.nc"
                    )
                    fs.open_dataset(uri)

    def test_nocache(self):
        fs = httpstore(cache=False)
        with pytest.raises(FileSystemHasNoCache):
            fs.cachepath("dummy_uri")

    def test_cacheable(self):
        with argopy.set_options(cachedir=self.cachedir):
            fs = httpstore(cache=True)
            assert isinstance(fs.fs, fsspec.implementations.cached.WholeFileCacheFileSystem)

    def test_nocachefile(self):
        with argopy.set_options(cachedir=self.cachedir):
            fs = httpstore(cache=True)
            with pytest.raises(CacheFileNotFound):
                fs.cachepath("dummy_uri")

    def test_cache_a_file(self):
        uri = self._mockeduri(self.repo + "ftp/ar_index_global_prof.txt")
        fs = httpstore(
            cache=True, cachedir=self.cachedir, timeout=OPTIONS["api_timeout"]
        )
        fs.read_csv(uri, skiprows=8, header=0)
        assert isinstance(fs.cachepath(uri), str)

    def test_clear_cache(self):
        uri = self._mockeduri(self.repo + "ftp/ar_index_global_prof.txt")
        fs = httpstore(
            cache=True, cachedir=self.cachedir, timeout=OPTIONS["api_timeout"]
        )
        fs.read_csv(uri, skiprows=8, header=0)
        assert isinstance(fs.cachepath(uri), str)
        assert os.path.isfile(fs.cachepath(uri))
        fs.clear_cache()
        with pytest.raises(CacheFileNotFound):
            fs.cachepath(uri)

    def test_open_dataset(self):
        uri = self._mockeduri(self.repo + "ftp/dac/csiro/5900865/5900865_prof.nc")
        assert isinstance(self.fs.open_dataset(uri), xr.Dataset)

    @pytest.mark.parametrize(
        "params", mf_params_nc, indirect=False, ids=mf_params_nc_ids
    )
    def test_open_mfdataset(self, params):
        uri = [self._mockeduri(u) for u in self.mf_nc]

        method, progress, concat = params

        if method == "process":
            pytest.skip("concurrent.futures.ProcessPoolExecutor is too long on GA !")

        ds = self.fs.open_mfdataset(
            uri, method=method, progress="disable" if progress else False, concat=concat
        )
        if concat:
            assert isinstance(ds, xr.Dataset)
        else:
            assert is_list_of_datasets(ds)

    params = [(m) for m in ["seq", "thread", "invalid"]]
    ids_params = ["method=%s" % (p) for p in params]

    @pytest.mark.parametrize("params", params, indirect=False, ids=ids_params)
    def test_open_mfdataset_error(self, params):
        uri = [self._mockeduri(u) for u in self.mf_nc]

        def preprocess(ds):
            """Fake preprocessor raising an error"""
            raise ValueError

        method = params
        if "invalid" in method:
            err = InvalidMethod
        else:
            err = ValueError
        with pytest.raises(err):
            self.fs.open_mfdataset(
                uri, method=method, preprocess=preprocess, errors="raise"
            )

        for errors in ["ignore", "silent"]:
            with pytest.raises(ValueError):
                self.fs.open_mfdataset(
                    uri, method=method, preprocess=preprocess, errors=errors
                )

    def test_open_mfdataset_DataNotFound(self):
        uri = [self._mockeduri(u) for u in self.mf_nc]

        def preprocess(ds):
            """Fake preprocessor returning nothing"""
            return None

        with pytest.raises(DataNotFound):
            self.fs.open_mfdataset(uri, preprocess=preprocess)

    def test_open_json(self):
        uri = self._mockeduri("https://api.ifremer.fr/argopy/data/ARGO-FULL.json")
        assert isinstance(self.fs.open_json(uri), dict)

    @pytest.mark.parametrize(
        "params", mf_params_js, indirect=False, ids=mf_params_js_ids
    )
    def test_open_mfjson(self, params):
        uri = [self._mockeduri(u) for u in self.mf_js]
        method, progress = params

        if method == "process":
            pytest.skip("concurrent.futures.ProcessPoolExecutor is too long on GA !")

        lst = self.fs.open_mfjson(
            uri, method=method, progress="disable" if progress else False
        )
        assert is_list_of_dicts(lst)

    params = [(m) for m in ["seq", "thread", "invalid"]]
    ids_params = ["method=%s" % (p) for p in params]

    @pytest.mark.parametrize("params", params, indirect=False, ids=ids_params)
    def test_open_mfjson_error(self, params):
        uri = [self._mockeduri(u) for u in self.mf_js]

        def preprocess(ds):
            """Fake preprocessor raising an error"""
            raise ValueError

        method = params
        if "invalid" in method:
            err = InvalidMethod
        else:
            err = ValueError
        with pytest.raises(err):
            self.fs.open_mfjson(
                uri, method=method, preprocess=preprocess, errors="raise"
            )

        for errors in ["ignore", "silent"]:
            with pytest.raises(ValueError):
                self.fs.open_mfjson(
                    uri, method=method, preprocess=preprocess, errors=errors
                )

    def test_open_mfjson_DataNotFound(self):
        uri = [self._mockeduri(u) for u in self.mf_js]

        def preprocess(ds):
            """Fake preprocessor returning nothing"""
            return None

        with pytest.raises(DataNotFound):
            self.fs.open_mfjson(uri, preprocess=preprocess)

    def test_read_csv(self):
        uri = self._mockeduri(self.repo + "ftp/ar_index_global_prof.txt")
        assert isinstance(
            self.fs.read_csv(uri, skiprows=8, header=0), pd.core.frame.DataFrame
        )


@skip_this
class Test_MemoryStore:
    def test_implementation(self):
        fs = memorystore(cache=False)
        assert isinstance(fs.fs, fsspec.implementations.memory.MemoryFileSystem)

    def test_nocache(self):
        fs = memorystore(cache=False)
        with pytest.raises(FileSystemHasNoCache):
            fs.cachepath("dummy_uri")

    def test_cacheable(self):
        fs = memorystore(cache=True)
        assert isinstance(fs.fs, fsspec.implementations.cached.WholeFileCacheFileSystem)

    def test_nocachefile(self):
        fs = memorystore(cache=True)
        with pytest.raises(CacheFileNotFound):
            fs.cachepath("dummy_uri")

    def test_exists(self):
        fs = memorystore(cache=False)
        assert not fs.exists('dummy.txt')
        fs = memorystore(cache=True)
        assert not fs.exists('dummy.txt')


@skip_this
class Test_FtpStore:
    fs = None
    mf_nc = ["dac/csiro/5900865/profiles/D5900865_00%i.nc" % i for i in [1, 2]]

    #########
    # UTILS #
    #########
    @property
    def host(self):
        # h = 'ftp.ifremer.fr'
        h = urlparse(pytest.MOCKFTP).hostname
        log.debug("Using FTP host: %s" % h)
        return h

    @property
    def port(self):
        # p = 0
        p = int(urlparse(pytest.MOCKFTP).port)
        log.debug("Using FTP port: %i" % p)
        return p

    @pytest.fixture
    def store(self, request):
        if self.fs is None:
            self.fs = ftpstore(host=self.host, port=self.port, cache=False)
        return self.fs

    #########
    # TESTS #
    #########
    def test_implementation(self, store):
        assert isinstance(store.fs, fsspec.implementations.ftp.FTPFileSystem)

    def test_open_dataset(self, store):
        uri = "dac/csiro/5900865/5900865_prof.nc"
        assert isinstance(store.open_dataset(uri), xr.Dataset)

    def test_open_dataset_error(self, store):
        uri = "dac/csiro/5900865/5900865_prof_error.nc"
        with pytest.raises(ftplib.error_perm):
            assert isinstance(store.open_dataset(uri), xr.Dataset)

    params = [
        (m, p, c)
        for m in ["seq", "process"]
        for p in [True, False]
        for c in [True, False]
    ]
    ids_params = [
        "method=%s, progress=%s, concat=%s" % (p[0], p[1], p[2]) for p in params
    ]

    @pytest.mark.parametrize("params", params, indirect=False, ids=ids_params)
    def test_open_mfdataset(self, store, params):
        uri = self.mf_nc

        def test(this_params):
            method, progress, concat = this_params

            if method == "process":
                pytest.skip(
                    "concurrent.futures.ProcessPoolExecutor is too long on GA !"
                )

            ds = store.open_mfdataset(
                uri,
                method=method,
                progress="disable" if progress else False,
                concat=concat,
                errors="raise",
            )
            if concat:
                assert isinstance(ds, xr.Dataset)
            else:
                assert is_list_of_datasets(ds)

        test(params)

    params = [(m) for m in ["seq", "process", "invalid"]]
    ids_params = ["method=%s" % (p) for p in params]

    @pytest.mark.parametrize("params", params, indirect=False, ids=ids_params)
    def test_open_mfdataset_error(self, store, params):
        method = params
        uri = self.mf_nc

        def preprocess(ds):
            """Fake preprocessor raising an error"""
            raise ValueError

        if method == "process":
            pytest.skip("concurrent.futures.ProcessPoolExecutor is too long on GA !")

        if "invalid" in method:
            err = InvalidMethod
        else:
            err = ValueError
        with pytest.raises(err):
            store.open_mfdataset(
                uri, method=method, preprocess=preprocess, errors="raise"
            )

        for errors in ["ignore", "silent"]:
            with pytest.raises(ValueError):
                store.open_mfdataset(
                    uri, method=method, preprocess=preprocess, errors=errors
                )

    def test_open_mfdataset_DataNotFound(self, store):
        uri = self.mf_nc

        def preprocess(ds):
            """Fake preprocessor returning nothing"""
            return None

        with pytest.raises(DataNotFound):
            store.open_mfdataset(uri, preprocess=preprocess)


@skip_this
class Test_IndexFilter_WMO:
    kwargs = [
        {"WMO": 6901929},
        {"WMO": [6901929, 2901623]},
        {"CYC": 1},
        {"CYC": [1, 6]},
        {"WMO": 6901929, "CYC": 36},
        {"WMO": 6901929, "CYC": [5, 45]},
        {"WMO": [6901929, 2901623], "CYC": 2},
        {"WMO": [6901929, 2901623], "CYC": [2, 23]},
        {},
    ]

    def test_implementation(self):
        for kw in self.kwargs:
            filt = indexfilter_wmo(**kw)
            assert isinstance(filt, argopy.stores.argo_index.indexfilter_wmo)

    def test_filters_uri(self):
        for kw in self.kwargs:
            filt = indexfilter_wmo(**kw)
            assert isinstance(filt.uri, str)

    def test_filters_sha(self):
        for kw in self.kwargs:
            filt = indexfilter_wmo(**kw)
            assert isinstance(filt.sha, str) and len(filt.sha) == 64

    @requires_connection
    def test_filters_run(self):
        ftproot, flist = argopy.tutorial.open_dataset("gdac")
        index_file = os.path.sep.join([ftproot, "ar_index_global_prof.txt"])
        for kw in self.kwargs:
            filt = indexfilter_wmo(**kw)
            with open(index_file, "r") as f:
                results = filt.run(f)
                if results:
                    assert isinstance(results, str)
                else:
                    assert results is None


@skip_this
@requires_connection
class Test_Legacy_IndexStore:
    ftproot, flist = argopy.tutorial.open_dataset("gdac")
    index_file = os.path.sep.join([ftproot, "ar_index_global_prof.txt"])

    kwargs_wmo = [
        {"WMO": 6901929},
        {"WMO": [6901929, 2901623]},
        {"CYC": 1},
        {"CYC": [1, 6]},
        {"WMO": 6901929, "CYC": 36},
        {"WMO": 6901929, "CYC": [5, 45]},
        {"WMO": [6901929, 2901623], "CYC": 2},
        {"WMO": [6901929, 2901623], "CYC": [2, 23]},
        {},
    ]

    kwargs_box = [
        {"BOX": [-60, -40, 40.0, 60.0]},
        {"BOX": [-60, -40, 40.0, 60.0, "2007-08-01", "2007-09-01"]},
    ]

    def test_implementation(self):
        assert isinstance(indexstore(), argopy.stores.argo_index.indexstore)
        assert isinstance(indexstore(cache=True), argopy.stores.argo_index.indexstore)
        assert isinstance(
            indexstore(cache=True, cachedir="."), argopy.stores.argo_index.indexstore
        )
        assert isinstance(
            indexstore(index_file="toto.txt"), argopy.stores.argo_index.indexstore
        )

    def test_search_wmo(self):
        for kw in self.kwargs_wmo:
            df = indexstore(cache=False, index_file=self.index_file).read_csv(
                indexfilter_wmo(**kw)
            )
            assert isinstance(df, pd.core.frame.DataFrame)

    def test_search_box(self):
        for kw in self.kwargs_box:
            df = indexstore(cache=False, index_file=self.index_file).read_csv(
                indexfilter_box(**kw)
            )
            assert isinstance(df, pd.core.frame.DataFrame)


"""
List ftp hosts to be tested. 
Since the fetcher is compatible with host from local, http or ftp protocols, we
try to test them all:
"""
VALID_HOSTS = [argopy.tutorial.open_dataset("gdac")[0],
             #'https://data-argo.ifremer.fr',
               mocked_server_address,
             # 'ftp://ftp.ifremer.fr/ifremer/argo',
             # 'ftp://usgodae.org/pub/outgoing/argo',  # ok, but takes too long to respond, slow down CI
             'MOCKFTP',  # keyword to use a fake/mocked ftp server (running on localhost)
             ]

"""
List index searches to be tested.
"""
VALID_SEARCHES = [
    {"wmo": [13857]},
    # {"wmo": [6901929, 2901623]},
    {"cyc": [5 ,45]},
    {"wmo_cyc": [13857, 2]},
    {"tim": [-60, -40, 40.0, 60.0, "2007-08-01", "2007-09-01"]},
    {"lat_lon": [-60, -40, 40.0, 60.0, "2007-08-01", "2007-09-01"]},
    {"lat_lon_tim": [-60, -40, 40.0, 60.0, "2007-08-01", "2007-09-01"]},
    ]

def run_a_search(idx_maker, fetcher_args, search_point, xfail=False):
    """ Create and run a search on a given index store

        Use xfail=True when a test with this is expected to fail
    """
    def core(fargs, apts):
        try:
            idx = idx_maker(**fargs)
            if "wmo" in apts:
                idx.search_wmo(apts['wmo'])
            if "cyc" in apts:
                idx.search_cyc(apts['cyc'])
            if "wmo_cyc" in apts:
                idx.search_wmo_cyc(apts['wmo_cyc'][0], apts['wmo_cyc'][1])
            if "tim" in apts:
                idx.search_tim(apts['tim'])
            if "lat_lon" in apts:
                idx.search_lat_lon(apts['lat_lon'])
            if "lat_lon_tim" in apts:
                idx.search_lat_lon_tim(apts['lat_lon_tim'])
        except Exception:
            raise
        return idx
    return core(fetcher_args, search_point)


def ftp_shortname(ftp):
    """Get a short name for scenarios IDs, given a FTP host"""
    if ftp == 'MOCKFTP':
        return 'ftp_mocked'
    elif 'localhost' in ftp or '127.0.0.1' in ftp:
        return 'http_mocked'
    else:
        return (lambda x: 'file' if x == "" else x)(urlparse(ftp).scheme)


class IndexStore_test_proto:
    host, flist = argopy.tutorial.open_dataset("gdac")
    index_file = "ar_index_global_prof.txt"

    search_scenarios = [(h, ap) for h in VALID_HOSTS for ap in VALID_SEARCHES]
    search_scenarios_ids = [
        "%s, %s" % (ftp_shortname(fix[0]), list(fix[1].keys())[0]) for fix
        in
        search_scenarios]

    #############
    # UTILITIES #
    #############

    def setup_class(self):
        """setup any state specific to the execution of the given class"""
        # Create the cache folder here, so that it's not the same for the pandas and pyarrow tests
        self.cachedir = tempfile.mkdtemp()

    def teardown_class(self):
        """Cleanup once we are finished."""
        def remove_test_dir():
            shutil.rmtree(self.cachedir)
        remove_test_dir()

    def _patch_ftp(self, ftp):
        """Patch Mocked FTP server keyword"""
        if ftp == 'MOCKFTP':
            return pytest.MOCKFTP  # this was set in conftest.py
        else:
            return ftp

    def create_store(self, store_args, xfail=False):
        def core(fargs):
            try:
                idx = self.indexstore(**fargs)
            except Exception:
                raise
            return idx
        return core(store_args)

    def _setup_store(self, this_request, cached=False):
        """Helper method to set up options for an index store creation"""
        index_file = self.index_file
        if hasattr(this_request, 'param'):
            if isinstance(this_request.param, tuple):
                host = this_request.param[0]
            else:
                host = this_request.param
        else:
            host = this_request['param']['host']
            index_file = this_request['param']['index_file']
        N_RECORDS = None if 'tutorial' in host or 'MOCK' in host else 100  # Make sure we're not going to load the full index
        fetcher_args = {"host": self._patch_ftp(host), "index_file": index_file, "cache": False}
        if cached:
            fetcher_args = {**fetcher_args, **{"cache": True, "cachedir": self.cachedir}}
        return fetcher_args, N_RECORDS

    def new_idx(self, cache=False, cachedir=None, **kwargs):
        host = kwargs['host'] if 'host' in kwargs else self.host
        index_file = kwargs['index_file'] if 'index_file' in kwargs else self.index_file
        fetcher_args, N_RECORDS = self._setup_store({'param': {'host': host, 'index_file': index_file}}, cached=cache)
        idx = self.create_store(fetcher_args).load(nrows=N_RECORDS)
        return idx

    @pytest.fixture
    def a_store(self, request):
        """Fixture to create an index store for a given host."""
        fetcher_args, N_RECORDS = self._setup_store(request)
        # yield self.indexstore(**fetcher_args).load(nrows=N_RECORDS)
        yield self.create_store(fetcher_args).load(nrows=N_RECORDS)

    @pytest.fixture
    def a_search(self, request):
        """ Fixture to create a FTP fetcher for a given host and access point """
        host = request.param[0]
        srch = request.param[1]
        yield run_a_search(self.new_idx, {'host': host, 'cache': True}, srch)

    def assert_index(self, this_idx, cacheable=False):
        assert hasattr(this_idx, 'index')
        assert this_idx.shape[0] == this_idx.index.shape[0]
        assert this_idx.N_RECORDS == this_idx.index.shape[0]
        assert is_list_of_strings(this_idx.uri_full_index) and len(this_idx.uri_full_index) == this_idx.N_RECORDS
        if cacheable:
            assert is_list_of_strings(this_idx.cachepath('index'))

    def assert_search(self, this_idx, cacheable=False):
        assert hasattr(this_idx, 'search')
        assert this_idx.N_MATCH == this_idx.search.shape[0]
        assert this_idx.N_FILES == this_idx.N_MATCH
        assert is_list_of_strings(this_idx.uri) and len(this_idx.uri) == this_idx.N_MATCH
        if cacheable:
            assert is_list_of_strings(this_idx.cachepath('search'))

    #########
    # TESTS #
    #########

    @pytest.mark.parametrize("a_store", VALID_HOSTS,
                             indirect=True,
                             ids=["%s" % ftp_shortname(ftp) for ftp in VALID_HOSTS])
    def test_hosts(self, mocked_httpserver, a_store):
        self.assert_index(a_store) # assert (this_store.N_RECORDS >= 1)  # Make sure we loaded the index file content

    @pytest.mark.parametrize("ftp_host", ['invalid', 'https://invalid_ftp', 'ftp://invalid_ftp'], indirect=False)
    def test_hosts_invalid(self, ftp_host):
        # Invalid servers:
        with pytest.raises(FtpPathError):
            self.indexstore(host=ftp_host)

    def test_index(self):
        def new_idx():
            return self.indexstore(host=self.host, index_file=self.index_file, cache=False)
        self.assert_index(new_idx().load())
        self.assert_index(new_idx().load(force=True))

        N = np.random.randint(1, 100+1)
        idx = new_idx().load(nrows=N)
        self.assert_index(idx)
        assert idx.index.shape[0] == N
        # Since no search was triggered:
        assert idx.N_FILES == idx.N_RECORDS

        with pytest.raises(OptionValueError):
            idx = self.indexstore(host=self.host, index_file="ar_greylist.txt", cache=False)
            idx.load()

    @pytest.mark.parametrize("a_search", search_scenarios, indirect=True, ids=search_scenarios_ids)
    def test_search(self, mocked_httpserver, a_search):
        self.assert_search(a_search, cacheable=False)

    def test_to_dataframe_index(self):
        idx = self.new_idx()
        assert isinstance(idx.to_dataframe(), pd.core.frame.DataFrame)

        df = idx.to_dataframe(index=True)
        assert df.shape[0] == idx.N_RECORDS

        df = idx.to_dataframe()
        assert df.shape[0] == idx.N_RECORDS

        N = np.random.randint(1, 20+1)
        df = idx.to_dataframe(index=True, nrows=N)
        assert df.shape[0] == N

    def test_to_dataframe_search(self):
        idx = self.new_idx()
        wmo = [s['wmo'] for s in VALID_SEARCHES if 'wmo' in s.keys()][0]
        idx = idx.search_wmo(wmo)

        df = idx.to_dataframe()
        assert isinstance(df, pd.core.frame.DataFrame)
        assert df.shape[0] == idx.N_MATCH

        N = np.random.randint(1,10+1)
        df = idx.to_dataframe(nrows=N)
        assert df.shape[0] == N

    def test_caching_index(self):
        idx = self.new_idx(cache=True)
        self.assert_index(idx, cacheable=True)

    def test_caching_search(self):
        idx = self.new_idx(cache=True)
        wmo = [s['wmo'] for s in VALID_SEARCHES if 'wmo' in s.keys()][0]
        idx.search_wmo(wmo)
        self.assert_search(idx, cacheable=True)

    def test_read_wmo(self):
        wmo = [s['wmo'] for s in VALID_SEARCHES if 'wmo' in s.keys()][0]
        idx = self.new_idx().search_wmo(wmo)
        assert len(idx.read_wmo()) == len(wmo)

    def test_records_per_wmo(self):
        wmo = [s['wmo'] for s in VALID_SEARCHES if 'wmo' in s.keys()][0]
        idx = self.new_idx().search_wmo(wmo)
        C = idx.records_per_wmo()
        for w in C:
            assert str(C[w]).isdigit()

    def test_to_indexfile(self):
        # Create a store and make a simple float search:
        idx = self.new_idx()
        wmo = [s['wmo'] for s in VALID_SEARCHES if 'wmo' in s.keys()][0]
        idx = idx.search_wmo(wmo)

        # Then save this search as a new Argo index file:
        tf = tempfile.NamedTemporaryFile(delete=False)
        new_indexfile = idx.to_indexfile(tf.name)

        # Finally try to load the new index file, like it was an official one:
        idx = self.new_idx(host=os.path.dirname(new_indexfile), index_file=os.path.basename(new_indexfile))
        self.assert_index(idx.load())

        # Cleanup
        tf.close()


@skip_this
class Test_IndexStore_pandas(IndexStore_test_proto):
    indexstore = indexstore_pandas


@skip_this
@skip_pyarrow
class Test_IndexStore_pyarrow(IndexStore_test_proto):
    from argopy.stores.argo_index_pa import indexstore_pyarrow
    indexstore = indexstore_pyarrow

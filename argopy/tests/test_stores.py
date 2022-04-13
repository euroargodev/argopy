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
from fsspec.core import split_protocol

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
from argopy.options import OPTIONS, check_gdac_path
from argopy.errors import FileSystemHasNoCache, CacheFileNotFound, InvalidDatasetStructure, FtpPathError
from utils import requires_connection, requires_connected_argovis, safe_to_server_errors, fct_safe_to_server_errors
from argopy.utilities import (
    is_list_of_datasets,
    is_list_of_dicts,
    modified_environ,
    is_list_of_strings,
    check_wmo, check_cyc,
    lscache, isconnected
)

from argopy.stores.argo_index_pd import indexstore_pandas


log = logging.getLogger("argopy.tests.stores")

has_pyarrow = importlib.util.find_spec('pyarrow') is not None
skip_pyarrow = pytest.mark.skipif(not has_pyarrow, reason="Requires pyarrow")

skip_this = pytest.mark.skipif(False, reason="Skipped temporarily")
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
    ftproot = argopy.tutorial.open_dataset("localftp")[0]
    csvfile = os.path.sep.join([ftproot, "ar_index_global_prof.txt"])

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
        fs = filestore()
        assert isinstance(fs.glob(os.path.sep.join([self.ftproot, "dac/*"])), list)

    def test_open_dataset(self):
        ncfile = os.path.sep.join([self.ftproot, "dac/aoml/5900446/5900446_prof.nc"])
        fs = filestore()
        assert isinstance(fs.open_dataset(ncfile), xr.Dataset)

    def test_open_mfdataset(self):
        fs = filestore()
        ncfiles = fs.glob(
            os.path.sep.join([self.ftproot, "dac/aoml/5900446/profiles/*_1*.nc"])
        )[0:2]
        for method in ["seq", "thread", "process"]:
            for progress in [True, False]:
                assert isinstance(
                    fs.open_mfdataset(ncfiles, method=method, progress=progress),
                    xr.Dataset,
                )
                assert is_list_of_datasets(
                    fs.open_mfdataset(
                        ncfiles, method=method, progress=progress, concat=False
                    )
                )

    def test_read_csv(self):
        fs = filestore()
        assert isinstance(
            fs.read_csv(self.csvfile, skiprows=8, header=0), pd.core.frame.DataFrame
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
@requires_connection
class Test_HttpStore:
    def test_implementation(self):
        fs = httpstore(cache=False)
        assert isinstance(fs.fs, fsspec.implementations.http.HTTPFileSystem)

    def test_trust_env(self):
        with modified_environ(HTTP_PROXY='http://dummy_proxy'):
            with argopy.set_options(trust_env=True):
                fs = httpstore(cache=False)
                with pytest.raises(aiohttp.client_exceptions.ClientConnectorError):
                    uri = "http://github.com/euroargodev/argopy-data/raw/master/ftp/dac/csiro/5900865/5900865_prof.nc"
                    fs.open_dataset(uri)

    def test_nocache(self):
        fs = httpstore(cache=False)
        with pytest.raises(FileSystemHasNoCache):
            fs.cachepath("dummy_uri")

    def test_cacheable(self):
        fs = httpstore(cache=True)
        assert isinstance(fs.fs, fsspec.implementations.cached.WholeFileCacheFileSystem)

    def test_nocachefile(self):
        fs = httpstore(cache=True)
        with pytest.raises(CacheFileNotFound):
            fs.cachepath("dummy_uri")

    @safe_to_server_errors
    def test_cache_a_file(self):
        uri = "https://github.com/euroargodev/argopy-data/raw/master/ftp/ar_index_global_prof.txt"
        with tempfile.TemporaryDirectory() as cachedir:
            fs = httpstore(cache=True, cachedir=cachedir, timeout=OPTIONS['api_timeout'])
            fs.read_csv(uri, skiprows=8, header=0)
            assert isinstance(fs.cachepath(uri), str)

    @safe_to_server_errors
    def test_clear_cache(self):
        uri = "https://github.com/euroargodev/argopy-data/raw/master/ftp/ar_index_global_prof.txt"
        with tempfile.TemporaryDirectory() as cachedir:
            fs = httpstore(cache=True, cachedir=cachedir, timeout=OPTIONS['api_timeout'])
            fs.read_csv(uri, skiprows=8, header=0)
            assert isinstance(fs.cachepath(uri), str)
            assert os.path.isfile(fs.cachepath(uri))
            fs.clear_cache()
            with pytest.raises(CacheFileNotFound):
                fs.cachepath(uri)

    @safe_to_server_errors
    def test_open_dataset(self):
        uri = "https://github.com/euroargodev/argopy-data/raw/master/ftp/dac/csiro/5900865/5900865_prof.nc"
        fs = httpstore(timeout=OPTIONS['api_timeout'])
        assert isinstance(fs.open_dataset(uri), xr.Dataset)

    @safe_to_server_errors
    def test_open_mfdataset(self):
        fs = httpstore(timeout=OPTIONS['api_timeout'])
        uri = [
            "https://github.com/euroargodev/argopy-data/raw/master/ftp/dac/csiro/5900865/profiles/D5900865_00%i.nc"
            % i
            for i in [1, 2]
        ]
        for method in ["seq", "thread"]:
            for progress in [True, False]:
                assert isinstance(
                    fs.open_mfdataset(uri, method=method, progress=progress), xr.Dataset
                )
                assert is_list_of_datasets(
                    fs.open_mfdataset(
                        uri, method=method, progress=progress, concat=False
                    )
                )

    @requires_connected_argovis
    @safe_to_server_errors
    def test_open_json(self):
        uri = "https://argovis.colorado.edu/catalog/mprofiles/?ids=['6902746_12']"
        fs = httpstore(timeout=OPTIONS['api_timeout'])
        assert is_list_of_dicts(fs.open_json(uri))

    @requires_connected_argovis
    @safe_to_server_errors
    def test_open_mfjson(self):
        fs = httpstore(timeout=OPTIONS['api_timeout'])
        uri = [
            "https://argovis.colorado.edu/catalog/mprofiles/?ids=['6902746_%i']" % i
            for i in [12, 13]
        ]
        for method in ["seq", "thread"]:
            for progress in [True, False]:
                lst = fs.open_mfjson(uri, method=method, progress=progress)
                assert all(is_list_of_dicts(x) for x in lst)

    @safe_to_server_errors
    def test_read_csv(self):
        uri = "https://github.com/euroargodev/argopy-data/raw/master/ftp/ar_index_global_prof.txt"
        fs = httpstore(timeout=OPTIONS['api_timeout'])
        assert isinstance(
            fs.read_csv(uri, skiprows=8, header=0), pd.core.frame.DataFrame
        )


@skip_this
@requires_connection
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
@requires_connection
class Test_FtpStore:
    host = 'ftp.ifremer.fr'

    @safe_to_server_errors
    def test_implementation(self):
        fs = ftpstore(host=self.host, cache=False)
        assert isinstance(fs.fs, fsspec.implementations.ftp.FTPFileSystem)

    @safe_to_server_errors
    def test_open_dataset(self):
        uri = "ifremer/argo/dac/csiro/5900865/5900865_prof.nc"
        fs = ftpstore(host=self.host, cache=False)
        assert isinstance(fs.open_dataset(uri), xr.Dataset)

    @safe_to_server_errors
    def test_open_mfdataset(self):
        fs = ftpstore(host=self.host, cache=False)
        uri = [
            "ifremer/argo/dac/csiro/5900865/profiles/D5900865_00%i.nc"
            % i
            for i in [1, 2]
        ]
        for method in ["seq", "thread"]:
            for progress in [True, False]:
                assert isinstance(
                    fs.open_mfdataset(uri, method=method, progress=progress), xr.Dataset
                )
                assert is_list_of_datasets(
                    fs.open_mfdataset(
                        uri, method=method, progress=progress, concat=False
                    )
                )


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
        ftproot, flist = argopy.tutorial.open_dataset("localftp")
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
    ftproot, flist = argopy.tutorial.open_dataset("localftp")
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
host_list = [argopy.tutorial.open_dataset("localftp")[0],
             'https://data-argo.ifremer.fr',
             'ftp://ftp.ifremer.fr/ifremer/argo',
             # 'ftp://usgodae.org/pub/outgoing/argo',  # ok, but takes too long to respond, slow down CI
             ]
# Make sure hosts are valid and available at test time:
valid_hosts = [h for h in host_list if isconnected(h) and check_gdac_path(h, errors='ignore')]

"""
List index searches to be tested.
"""
valid_searches = [
    # {"wmo": [6901929]},
    {"wmo": [6901929, 2901623]},
    # {"cyc": [36]},
    {"cyc": [5 ,45]},
    {"wmo_cyc": [6901929, 36]},
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
    return fct_safe_to_server_errors(core)(fetcher_args, search_point, xfail=xfail)


class IndexStore_test_proto:
    host, flist = argopy.tutorial.open_dataset("localftp")
    index_file = "ar_index_global_prof.txt"

    search_scenarios = [(h, ap) for h in valid_hosts for ap in valid_searches]
    search_scenarios_ids = [
        "%s, %s" % ((lambda x: 'file' if x is None else x)(split_protocol(fix[0])[0]), list(fix[1].keys())[0]) for fix
        in
        search_scenarios]

    def setup_class(self):
        """setup any state specific to the execution of the given class"""
        # Create the cache folder here, so that it's not the same for the pandas and pyarrow tests
        self.cachedir = tempfile.mkdtemp()

    def create_store(self, store_args, xfail=False):
        def core(fargs):
            try:
                idx = self.indexstore(**fargs)
            except Exception:
                raise
            return idx
        return fct_safe_to_server_errors(core)(store_args, xfail=xfail)

    def _setup_store(self, this_request, cached=False):
        """Helper method to set-up options for an index store creation"""
        if hasattr(this_request, 'param'):
            if isinstance(this_request.param, tuple):
                host = this_request.param[0]
            else:
                host = this_request.param
        else:
            host = this_request['param']
        N_RECORDS = None if 'tutorial' in host else 100  # Make sure we're not going to load the full index
        fetcher_args = {"host": host, "index_file": self.index_file, "cache": False}
        if cached:
            fetcher_args = {**fetcher_args, **{"cache": True, "cachedir": self.cachedir}}
        return fetcher_args, N_RECORDS

    def new_idx(self, cache=False, cachedir=None, **kwargs):
        host = kwargs['host'] if 'host' in kwargs else self.host
        fetcher_args, N_RECORDS = self._setup_store({'param': host}, cached=cache)
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

    # @skip_this
    @pytest.mark.parametrize("a_store", valid_hosts, indirect=True)
    def test_hosts(self, a_store):
        @safe_to_server_errors
        def test(this_store):
            self.assert_index(this_store) # assert (this_store.N_RECORDS >= 1)  # Make sure we loaded the index file content
        test(a_store)

    # @skip_this
    @pytest.mark.parametrize("ftp_host", ['invalid', 'https://invalid_ftp', 'ftp://invalid_ftp'], indirect=False)
    def test_hosts_invalid(self, ftp_host):
        # Invalid servers:
        with pytest.raises(FtpPathError):
            self.indexstore(host=ftp_host)

    # @skip_this
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

        with pytest.raises(InvalidDatasetStructure):
            idx = self.indexstore(host=self.host, index_file="ar_greylist.txt", cache=False)
            idx.load()

    @pytest.mark.parametrize("a_search", search_scenarios, indirect=True, ids=search_scenarios_ids)
    def test_search(self, a_search):
        @safe_to_server_errors
        def test(this_searched_store):
            self.assert_search(this_searched_store, cacheable=False)
        test(a_search)

    # @skip_this
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

    # @skip_this
    def test_to_dataframe_search(self):
        idx = self.new_idx()
        wmo = [s['wmo'] for s in valid_searches if 'wmo' in s.keys()][0]
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

    # @skip_this
    def test_caching_search(self):
        idx = self.new_idx(cache=True)
        wmo = [s['wmo'] for s in valid_searches if 'wmo' in s.keys()][0]
        idx.search_wmo(wmo)
        self.assert_search(idx, cacheable=True)

    # @skip_this
    def test_read_wmo(self):
        wmo = [s['wmo'] for s in valid_searches if 'wmo' in s.keys()][0]
        idx = self.new_idx().search_wmo(wmo)
        assert len(idx.read_wmo()) == len(wmo)

    # @skip_this
    def test_records_per_wmo(self):
        wmo = [s['wmo'] for s in valid_searches if 'wmo' in s.keys()][0]
        idx = self.new_idx().search_wmo(wmo)
        C = idx.records_per_wmo()
        for w in C:
            assert str(C[w]).isdigit()

    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self, request):
        """Cleanup once we are finished."""
        def remove_test_dir():
            # warnings.warn("\n%s" % argopy.lscache(self.cachedir))
            shutil.rmtree(self.cachedir)
        request.addfinalizer(remove_test_dir)


# @skip_this
@skip_for_debug
class Test_IndexStore_pandas(IndexStore_test_proto):
    indexstore = indexstore_pandas

# @skip_for_debug
@skip_pyarrow
class Test_IndexStore_pyarrow(IndexStore_test_proto):
    from argopy.stores.argo_index_pa import indexstore_pyarrow
    indexstore = indexstore_pyarrow

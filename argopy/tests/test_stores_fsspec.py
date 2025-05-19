import os

import pytest
import tempfile

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
import netCDF4

import argopy
from argopy.stores import (
    filestore,
    httpstore,
    memorystore,
    ftpstore,
)
from argopy.stores.filesystems import new_fs
from argopy.options import OPTIONS
from argopy.errors import (
    FileSystemHasNoCache,
    CacheFileNotFound,
    InvalidMethod,
    DataNotFound,
)
from argopy.utils.locals import modified_environ
from argopy.utils.checkers import (
    is_list_of_datasets,
    is_list_of_dicts,
)
from utils import requires_connection, requires_connected_argovis, create_temp_folder
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
        fs, cache_registry, fsspec_kwargs = new_fs()
        assert id_implementation(fs) is not None
        assert is_initialised(cache_registry)

    def test_cache_type(self):
        fs, cache_registry, fsspec_kwargs = new_fs(cache=True)
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

    params = [False, True]
    ids_params = ["netCDF4=%s" % p for p in params]
    @pytest.mark.parametrize("CDF4_options", params, indirect=False, ids=ids_params)
    def test_open_dataset(self, CDF4_options):
        ncfile = os.path.sep.join([self.ftproot, "dac/aoml/5900446/5900446_prof.nc"])
        ds = self.fs.open_dataset(ncfile, netCDF4=CDF4_options)
        instance = {False: xr.Dataset, True: netCDF4.Dataset}
        assert isinstance(ds, instance[CDF4_options])

    params = [(m, p, c) for m in ["sequential", "thread", "process"] for p in [True, False] for c in [True, False]]
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

    params = [(m) for m in ["sequential", "thread", "invalid"]]
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
        for m in ["sequential", "thread", "process"]
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

    mf_params_js = [(m, p) for m in ["sequential", "thread", "process"] for p in [True, False]]
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
        self.cachedir = create_temp_folder().folder

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

    params = [False, True]
    ids_params = ["netCDF4=%s" % p for p in params]
    @pytest.mark.parametrize("CDF4_options", params, indirect=False, ids=ids_params)
    def test_open_dataset(self, CDF4_options):
        uri = self._mockeduri(self.repo + "ftp/dac/csiro/5900865/5900865_prof.nc")
        ds = self.fs.open_dataset(uri, netCDF4=CDF4_options)
        instance = {False: xr.Dataset, True: netCDF4.Dataset}
        assert isinstance(ds, instance[CDF4_options])

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

    params = [(m) for m in ["sequential", "thread", "invalid"]]
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

    params = [(m) for m in ["sequential", "thread", "invalid"]]
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

    # def test_open_dataset(self, store):
    #     uri = "dac/csiro/5900865/5900865_prof.nc"
    #     assert isinstance(store.open_dataset(uri), xr.Dataset)

    params = [False, True]
    ids_params = ["netCDF4=%s" % p for p in params]
    @pytest.mark.parametrize("CDF4_options", params, indirect=False, ids=ids_params)
    def test_open_dataset(self, store, CDF4_options):
        uri = "dac/csiro/5900865/5900865_prof.nc"
        ds = store.open_dataset(uri, netCDF4=CDF4_options)
        instance = {False: xr.Dataset, True: netCDF4.Dataset}
        assert isinstance(ds, instance[CDF4_options])

    def test_open_dataset_error(self, store):
        uri = "dac/csiro/5900865/5900865_prof_error.nc"
        with pytest.raises((FileNotFoundError, ftplib.error_perm)):
            assert isinstance(store.open_dataset(uri), xr.Dataset)

    params = [
        (m, p, c)
        for m in ["sequential", "process"]
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

    params = [(m) for m in ["sequential", "process", "invalid"]]
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

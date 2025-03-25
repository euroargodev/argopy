import pytest
import xarray as xr
import fsspec
import importlib
import logging
from urllib.parse import urlparse
import ftplib

from argopy.stores import ftpstore
from argopy.errors import (
    InvalidMethod,
    DataNotFound,
)
from argopy.utils.checkers import (
    is_list_of_datasets,
)


log = logging.getLogger("argopy.tests.stores")


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

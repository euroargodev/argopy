import numpy as np
import xarray as xr

import pytest
import tempfile

import argopy
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import (
    CacheFileNotFound,
    FileSystemHasNoCache,
    FtpPathError
)
from argopy.utilities import is_list_of_strings
from . import (
    requires_connected_gdac,
    safe_to_server_errors
)


@requires_connected_gdac
class Test_Backend:
    """ Test GDAC FTP data fetching backend """

    src = "ftp"
    requests = {
        "float": [[1901393], [6902746]],
        "profile": [[6902746, 34], [6902746, np.arange(12, 14)], [6902746, [1, 12]]],
        "region": [
            [-65.1, -65, 35.1, 36., 0, 10.0],
            [-65.1, -65, 35.1, 36., 0, 10.0, "2013-01", "2013-03"]
        ],
    }
    local_ftp = argopy.tutorial.open_dataset("localftp")[0]

    def test_cachepath_notfound(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                fetcher = (
                    ArgoDataFetcher(src=self.src, cache=True)
                    .profile(*self.requests["profile"][0])
                    .fetcher
                )
                with pytest.raises(CacheFileNotFound):
                    fetcher.cachepath

    @safe_to_server_errors
    def test_nocache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                fetcher = (
                    ArgoDataFetcher(src=self.src, cache=False)
                    .profile(*self.requests["profile"][0])
                    .fetcher
                )
                with pytest.raises(FileSystemHasNoCache):
                    fetcher.cachepath

    @safe_to_server_errors
    def test_clearcache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                fetcher = (
                    ArgoDataFetcher(src=self.src, cache=True)
                    .profile(*self.requests["profile"][0])
                    .fetcher
                )
                fetcher.to_xarray()
                fetcher.clear_cache()
                with pytest.raises(CacheFileNotFound):
                    fetcher.cachepath

    @safe_to_server_errors
    def test_cached_float(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                fetcher = (
                    ArgoDataFetcher(src=self.src, cache=True)
                    .float(self.requests["float"][0])
                    .fetcher
                )
                ds = fetcher.to_xarray()
                assert isinstance(ds, xr.Dataset)
                assert is_list_of_strings(fetcher.uri)
                assert is_list_of_strings(fetcher.cachepath)

    @safe_to_server_errors
    def test_cached_profile(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                fetcher = (
                    ArgoDataFetcher(src=self.src, cache=True)
                    .profile(*self.requests["profile"][0])
                    .fetcher
                )
                ds = fetcher.to_xarray()
                assert isinstance(ds, xr.Dataset)
                assert is_list_of_strings(fetcher.uri)
                assert is_list_of_strings(fetcher.cachepath)

    @safe_to_server_errors
    def test_cached_region(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                fetcher = (
                    ArgoDataFetcher(src=self.src, cache=True)
                    .region(self.requests["region"][1])
                    .fetcher
                )
                ds = fetcher.to_xarray()
                assert isinstance(ds, xr.Dataset)
                assert is_list_of_strings(fetcher.uri)
                assert is_list_of_strings(fetcher.cachepath)

    def test_ftp_server(self):
        # Invalid servers:
        with pytest.raises(FtpPathError):
            ArgoDataFetcher(src='ftp', ftp='invalid').profile(1900857, np.arange(10,20))
        with pytest.raises(FtpPathError):
            ArgoDataFetcher(src=self.src, ftp='https://invalid_ftp').profile(1900857, np.arange(10,20))
        with pytest.raises(FtpPathError):
            ArgoDataFetcher(src=self.src, ftp='ftp://invalid_ftp').profile(1900857, np.arange(10,20))

        # Valid list of servers, using all possible protocols:
        for this_ftp in ['https://data-argo.ifremer.fr',
                    'ftp://ftp.ifremer.fr/ifremer/argo',
                    # 'ftp://usgodae.org/pub/outgoing/argo',  # ok, but takes too long to respond, slow down CI
                    self.local_ftp]:
            fetcher = ArgoDataFetcher(src=self.src, ftp=this_ftp).profile(*self.requests["profile"][0]).fetcher
            assert(fetcher.N_RECORDS >= 1)  # Make sure we loaded the index file content

    def __testthis_profile(self, dataset):
        fetcher_args = {"src": self.src, "ds": dataset}
        for arg in self.args["profile"]:
            f = ArgoDataFetcher(**fetcher_args).profile(*arg).fetcher
            assert isinstance(f.to_xarray(), xr.Dataset)
            assert is_list_of_strings(f.uri)

    def __testthis_float(self, dataset):
        fetcher_args = {"src": self.src, "ds": dataset}
        for arg in self.args["float"]:
            f = ArgoDataFetcher(**fetcher_args).float(arg).fetcher
            assert isinstance(f.to_xarray(), xr.Dataset)
            assert is_list_of_strings(f.uri)

    def __testthis_region(self, dataset):
        fetcher_args = {"src": self.src, "ds": dataset}
        for arg in self.args["region"]:
            f = ArgoDataFetcher(**fetcher_args).region(arg).fetcher
            assert isinstance(f.to_xarray(), xr.Dataset)
            assert is_list_of_strings(f.uri)

    def __testthis(self, dataset):
        for access_point in self.args:
            if access_point == "profile":
                self.__testthis_profile(dataset)
            elif access_point == "float":
                self.__testthis_float(dataset)
            elif access_point == "region":
                self.__testthis_region(dataset)

    @safe_to_server_errors
    def test_phy_float(self):
        self.args = {"float": self.requests["float"]}
        self.__testthis("phy")

    @safe_to_server_errors
    def test_phy_profile(self):
        self.args = {"profile": self.requests["profile"]}
        self.__testthis("phy")

    @safe_to_server_errors
    def test_phy_region(self):
        self.args = {"region": self.requests["region"]}
        self.__testthis("phy")


@requires_connected_gdac
class Test_BackendParallel:
    """ This test backend for parallel requests """

    src = "ftp"
    requests = {
        "region": [
            [-65.1, -65, 35.1, 36., 0, 10.0],
            [-65.4, -65, 35.1, 36., 0, 10.0, "2013-01-01", "2013-09-01"],
        ],
        "wmo": [[6902766, 6902772, 6902914]],
    }

    def test_methods(self):
        args_list = [
            {"src": self.src, "parallel": "thread"},
            {"src": self.src, "parallel": True, "parallel_method": "thread"},
        ]
        for fetcher_args in args_list:
            loader = ArgoDataFetcher(**fetcher_args).float(self.requests["wmo"][0])
            assert isinstance(loader, argopy.fetchers.ArgoDataFetcher)

        args_list = [
            {"src": self.src, "parallel": True, "parallel_method": "toto"},
            # {"src": self.src, "parallel": "process"},
            # {"src": self.src, "parallel": True, "parallel_method": "process"},
        ]
        for fetcher_args in args_list:
            with pytest.raises(ValueError):
                ArgoDataFetcher(**fetcher_args).float(self.requests["wmo"][0])

    @safe_to_server_errors
    def test_parallel(self):
        for access_arg in self.requests["region"]:
            fetcher_args = {
                "src": self.src,
                "parallel": True,
            }
            f = ArgoDataFetcher(**fetcher_args).region(access_arg).fetcher
            assert isinstance(f.to_xarray(), xr.Dataset)
            assert is_list_of_strings(f.uri)

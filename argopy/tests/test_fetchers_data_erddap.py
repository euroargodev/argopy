import numpy as np
import xarray as xr

import pytest
import tempfile

import argopy
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import (
    CacheFileNotFound,
    FileSystemHasNoCache
)
from argopy.utilities import is_list_of_strings
from utils import (
    requires_connected_erddap,
    requires_connected_erddap_phy,
    requires_connected_erddap_bgc,
    requires_connected_erddap_ref,
    safe_to_server_errors
)


@requires_connected_erddap
class Test_Backend:
    """ Test ERDDAP data fetching backend """

    src = "erddap"
    requests = {
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

    @requires_connected_erddap_phy
    def test_cachepath_notfound(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                fetcher = ArgoDataFetcher(src=self.src, cache=True).profile(
                    *self.requests["profile"][0]
                ).fetcher
                with pytest.raises(CacheFileNotFound):
                    fetcher.cachepath

    @requires_connected_erddap_phy
    @safe_to_server_errors
    def test_nocache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                fetcher = ArgoDataFetcher(src=self.src, cache=False).profile(
                    *self.requests["profile"][0]
                ).fetcher
                with pytest.raises(FileSystemHasNoCache):
                    fetcher.cachepath

    @requires_connected_erddap_phy
    @safe_to_server_errors
    def test_clearcache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                fetcher = ArgoDataFetcher(src=self.src, cache=True).float(
                    self.requests["float"][0]
                ).fetcher
                fetcher.to_xarray()
                fetcher.clear_cache()
                with pytest.raises(CacheFileNotFound):
                    fetcher.cachepath

    @requires_connected_erddap_phy
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

    @requires_connected_erddap_phy
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

    @requires_connected_erddap_phy
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

    @requires_connected_erddap_phy
    @safe_to_server_errors
    def test_N_POINTS(self):
        n = (
            ArgoDataFetcher(src=self.src)
            .region(self.requests["region"][1])
            .fetcher.N_POINTS
        )
        assert isinstance(n, int)

    def test_minimal_vlist(self):
        f = ArgoDataFetcher(src=self.src).region(self.requests["region"][1]).fetcher
        assert is_list_of_strings(f._minimal_vlist)

    def test_dtype(self):
        f = ArgoDataFetcher(src=self.src).region(self.requests["region"][1]).fetcher
        assert isinstance(f._dtype, dict)

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

    @requires_connected_erddap_phy
    @safe_to_server_errors
    def test_phy_float(self):
        self.args = {"float": self.requests["float"]}
        self.__testthis("phy")

    @requires_connected_erddap_phy
    @safe_to_server_errors
    def test_phy_profile(self):
        self.args = {"profile": self.requests["profile"]}
        self.__testthis("phy")

    @requires_connected_erddap_phy
    @safe_to_server_errors
    def test_phy_region(self):
        self.args = {"region": self.requests["region"]}
        self.__testthis("phy")

    @requires_connected_erddap_bgc
    @safe_to_server_errors
    def test_bgc_float(self):
        self.args = {"float": self.requests_bgc["float"]}
        self.__testthis("bgc")

    @requires_connected_erddap_bgc
    @safe_to_server_errors
    def test_bgc_profile(self):
        self.args = {"profile": self.requests_bgc["profile"]}
        self.__testthis("bgc")

    @requires_connected_erddap_bgc
    @safe_to_server_errors
    def test_bgc_region(self):
        self.args = {"region": self.requests_bgc["region"]}
        self.__testthis("bgc")

    @requires_connected_erddap_ref
    @safe_to_server_errors
    def test_ref_region(self):
        self.args = {"region": self.requests_ref["region"]}
        self.__testthis("ref")


@requires_connected_erddap_phy
class Test_BackendParallel:
    """ This test backend for parallel requests """

    src = "erddap"
    requests = {
        "region": [
            [-60, -55, 40.0, 45.0, 0.0, 20.0],
            [-60, -55, 40.0, 45.0, 0.0, 20.0, "2007-08-01", "2007-09-01"],
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
            {"src": self.src, "parallel": "process"},
            {"src": self.src, "parallel": True, "parallel_method": "process"},
        ]
        for fetcher_args in args_list:
            with pytest.raises(ValueError):
                ArgoDataFetcher(**fetcher_args).float(self.requests["wmo"][0])

    @safe_to_server_errors
    def test_chunks_region(self):
        for access_arg in self.requests["region"]:
            fetcher_args = {
                "src": self.src,
                "parallel": True,
                "chunks": {"lon": 1, "lat": 2, "dpt": 2, "time": 1},
            }
            f = ArgoDataFetcher(**fetcher_args).region(access_arg).fetcher
            assert isinstance(f.to_xarray(), xr.Dataset)
            assert is_list_of_strings(f.uri)
            assert len(f.uri) == np.prod(
                [v for k, v in fetcher_args["chunks"].items()]
            )

    @safe_to_server_errors
    def test_chunks_wmo(self):
        for access_arg in self.requests["wmo"]:
            fetcher_args = {
                "src": self.src,
                "parallel": True,
                "chunks_maxsize": {"wmo": 1},
            }
            f = ArgoDataFetcher(**fetcher_args).profile(access_arg, 12).fetcher
            assert isinstance(f.to_xarray(), xr.Dataset)
            assert is_list_of_strings(f.uri)
            assert len(f.uri) == len(access_arg)

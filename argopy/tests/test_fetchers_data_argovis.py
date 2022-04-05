import warnings

import numpy as np
import xarray as xr
import pandas as pd

import pytest
import tempfile

import argopy
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import (
    CacheFileNotFound,
    FileSystemHasNoCache,
)
from argopy.utilities import is_list_of_strings
from utils import requires_connected_argovis, safe_to_server_errors


skip_this_for_debug = pytest.mark.skipif(False, reason="Skipped temporarily for debug")


@requires_connected_argovis
class Test_Backend:
    """ Test main API facade for all available dataset and access points of the ARGOVIS data fetching backend """

    src = "argovis"
    requests = {
        "float": [[1901393], [1901393, 6902746]],
        # "profile": [[6902746, 12], [6902746, np.arange(12, 13)], [6902746, [1, 12]]],
        "profile": [[6902746, 12]],
        "region": [
            [-70, -65, 35.0, 40.0, 0, 10.0, "2012-01", "2012-03"],
            [-70, -65, 35.0, 40.0, 0, 10.0, "2012-01", "2012-06"],
        ],
    }

    @skip_this_for_debug
    def test_cachepath_notfound(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                fetcher = ArgoDataFetcher(src=self.src, cache=True).profile(*self.requests['profile'][0]).fetcher
                with pytest.raises(CacheFileNotFound):
                    fetcher.cachepath

    @skip_this_for_debug
    @safe_to_server_errors
    def test_nocache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                fetcher = ArgoDataFetcher(src=self.src, cache=False).profile(*self.requests['profile'][0]).fetcher
                with pytest.raises(FileSystemHasNoCache):
                    fetcher.cachepath

    @skip_this_for_debug
    @safe_to_server_errors
    def test_clearcache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                fetcher = ArgoDataFetcher(src=self.src, cache=True).float(self.requests['float'][0]).fetcher
                fetcher.to_xarray()
                fetcher.clear_cache()
                with pytest.raises(CacheFileNotFound):
                    fetcher.cachepath

    @skip_this_for_debug
    @safe_to_server_errors
    def test_caching_float(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                fetcher = (
                    ArgoDataFetcher(src=self.src, cache=True).float(self.requests['float'][0]).fetcher
                )
                ds = fetcher.to_xarray()
                assert isinstance(ds, xr.Dataset)
                assert is_list_of_strings(fetcher.uri)
                assert is_list_of_strings(fetcher.cachepath)

    @skip_this_for_debug
    @safe_to_server_errors
    def test_caching_profile(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                fetcher = ArgoDataFetcher(src=self.src, cache=True).profile(*self.requests['profile'][0]).fetcher
                ds = fetcher.to_xarray()
                assert isinstance(ds, xr.Dataset)
                assert is_list_of_strings(fetcher.uri)
                assert is_list_of_strings(fetcher.cachepath)

    @skip_this_for_debug
    @safe_to_server_errors
    def test_caching_region(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                fetcher = (
                    ArgoDataFetcher(src=self.src, cache=True)
                    .region(self.requests['region'][1])
                    .fetcher
                )
                ds = fetcher.to_xarray()
                assert isinstance(ds, xr.Dataset)
                assert is_list_of_strings(fetcher.uri)
                assert is_list_of_strings(fetcher.cachepath)

    def __testthis_profile(self, dataset):
        fetcher_args = {"src": self.src, "ds": dataset}
        for arg in self.args["profile"]:
            f = ArgoDataFetcher(**fetcher_args).profile(*arg).fetcher
            assert isinstance(f.to_xarray(), xr.Dataset)
            # assert isinstance(f.to_dataframe(), pd.core.frame.DataFrame)
            # ds = xr.Dataset.from_dataframe(f.to_dataframe())
            # ds = ds.sortby(
            #     ["TIME", "PRES"]
            # )  # should already be sorted by date in descending order
            # ds["N_POINTS"] = np.arange(
            #     0, len(ds["N_POINTS"])
            # )  # Re-index to avoid duplicate values
            #
            # # Set coordinates:
            # # ds = ds.set_coords('N_POINTS')
            # coords = ("LATITUDE", "LONGITUDE", "TIME", "N_POINTS")
            # ds = ds.reset_coords()
            # ds["N_POINTS"] = ds["N_POINTS"]
            # # Convert all coordinate variable names to upper case
            # for v in ds.data_vars:
            #     ds = ds.rename({v: v.upper()})
            # ds = ds.set_coords(coords)
            #
            # # Cast data types and add variable attributes (not available in the csv download):
            # warnings.warn(type(ds['TIME'].data))
            # warnings.warn(ds['TIME'].data[0])
            # ds['TIME'] = ds['TIME'].astype(np.datetime64)
            # assert isinstance(ds, xr.Dataset)
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

    @skip_this_for_debug
    @safe_to_server_errors
    def test_phy_float(self):
        self.args = {"float": self.requests['float']}
        self.__testthis("phy")

    @safe_to_server_errors
    def test_phy_profile(self):
        self.args = {"profile": self.requests['profile']}
        self.__testthis("phy")

    @skip_this_for_debug
    @safe_to_server_errors
    def test_phy_region(self):
        self.args = {"region": self.requests['region']}
        self.__testthis("phy")


@skip_this_for_debug
@requires_connected_argovis
class Test_BackendParallel:
    """ This test backend for parallel requests """

    src = "argovis"
    requests = {
        "region": [
            [-60, -55, 40.0, 45.0, 0.0, 10.0],
            [-60, -55, 40.0, 45.0, 0.0, 10.0, "2007-08-01", "2007-09-01"],
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
                "chunks": {"lon": 1, "lat": 2, "dpt": 1, "time": 2},
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

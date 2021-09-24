import pandas as pd
import xarray as xr

import pytest
import tempfile

import argopy
from argopy import IndexFetcher as ArgoIndexFetcher
from argopy.errors import (
    FileSystemHasNoCache,
    CacheFileNotFound,
)
from . import requires_connected_erddap_index, safe_to_server_errors, safe_to_fsspec_version


@requires_connected_erddap_index
class Test_Backend_WMO:
    """ Test ERDDAP index fetching backend for WMO access point"""

    src = "erddap"
    requests = {
        "float": [[2901623], [2901623, 6901929]]
    }

    @safe_to_fsspec_version
    def test_cachepath_notfound(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                loader = ArgoIndexFetcher(src=self.src, cache=True).float(self.requests['float'][0])
                with pytest.raises(CacheFileNotFound):
                    loader.fetcher.cachepath

    def test_nocache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                loader = ArgoIndexFetcher(src=self.src, cache=False).float(self.requests['float'][0])
                loader.to_dataframe()
                with pytest.raises(FileSystemHasNoCache):
                    loader.fetcher.cachepath

    @safe_to_fsspec_version
    @safe_to_server_errors
    def test_clearcache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                loader = ArgoIndexFetcher(src=self.src, cache=True).float(self.requests['float'][0])
                loader.fetcher.to_dataframe()  # 1st call to load from source and save in memory
                loader.fetcher.to_dataframe()  # 2nd call to load from memory and save in cache
                loader.fetcher.clear_cache()
                with pytest.raises(CacheFileNotFound):
                    loader.fetcher.cachepath

    @safe_to_fsspec_version
    @safe_to_server_errors
    def test_caching(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                loader = ArgoIndexFetcher(src=self.src, cache=True).float(self.requests['float'][0])
                loader.to_dataframe()  # 1st call to load from source and save in memory
                loader.to_dataframe()  # 2nd call to load from memory and save in cache
                assert isinstance(loader.index, pd.core.frame.DataFrame)
                assert isinstance(loader.fetcher.cachepath, str)

    def test_url(self):
        for arg in self.requests["float"]:
            loader = ArgoIndexFetcher(src=self.src).float(arg)
            assert isinstance(loader.fetcher.url, str)

    def __testthis(self):
        for arg in self.requests["float"]:
            fetcher = ArgoIndexFetcher(src=self.src).float(arg).fetcher
            df = fetcher.to_dataframe()
            assert isinstance(df, pd.core.frame.DataFrame)

    # @safe_to_server_errors
    # def test_phy_float(self):
    #     self.args = {"float": self.requests["float"]}
    #     self.__testthis()


@requires_connected_erddap_index
class Test_Backend_BOX:
    """ Test ERDDAP index fetching backend for the BOX access point """

    src = "erddap"
    requests = {
        "region": [
            [-60, -50, 40.0, 50.0],
            [-60, -55, 40.0, 45.0, "2007-08-01", "2007-09-01"],
        ],
    }

    def test_cachepath_notfound(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                loader = ArgoIndexFetcher(src=self.src, cache=True).region(self.requests['region'][-1])
                with pytest.raises(CacheFileNotFound):
                    loader.fetcher.cachepath

    def test_nocache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                loader = ArgoIndexFetcher(src=self.src, cache=False).region(self.requests['region'][-1])
                loader.fetcher.to_dataframe()
                with pytest.raises(FileSystemHasNoCache):
                    loader.fetcher.cachepath

    @safe_to_server_errors
    def test_clearcache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                loader = ArgoIndexFetcher(src=self.src, cache=True).region(self.requests['region'][-1])
                loader.fetcher.to_dataframe()  # 1st call to load from source and save in memory
                loader.fetcher.to_dataframe()  # 2nd call to load from memory and save in cache
                loader.fetcher.clear_cache()
                with pytest.raises(CacheFileNotFound):
                    loader.fetcher.cachepath

    @safe_to_server_errors
    def test_caching(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                loader = ArgoIndexFetcher(src=self.src, cache=True).region(self.requests['region'][-1])
                loader.fetcher.to_dataframe()  # 1st call to load from source and save in memory
                df = loader.fetcher.to_dataframe()  # 2nd call to load from memory and save in cache
                assert isinstance(df, pd.core.frame.DataFrame)
                assert isinstance(loader.fetcher.cachepath, str)

    def test_url(self):
        for arg in self.requests["region"]:
            loader = ArgoIndexFetcher(src=self.src).region(arg)
            assert isinstance(loader.fetcher.url, str)

    def __testthis(self):
        for arg in self.requests["region"]:
            fetcher = ArgoIndexFetcher(src=self.src).region(arg).fetcher
            df = fetcher.to_dataframe()
            assert isinstance(df, pd.core.frame.DataFrame)

    # @safe_to_server_errors
    # def test_phy_float(self):
    #     self.args = {"float": self.requests["float"]}
    #     self.__testthis()
    #
    # # @pytest.mark.skip(reason="Waiting for https://github.com/euroargodev/argopy/issues/16")
    # # def test_phy_profile(self):
    # #     self.args = {'profile': [[6901929, 36],
    # #                              [6901929, [5, 45]]]}
    # #     self.__testthis()
    #
    # # @pytest.mark.skip(reason="Waiting for https://github.com/euroargodev/argopy/issues/16")
    # @safe_to_server_errors
    # def test_phy_region(self):
    #     self.args = {"region": self.requests["region"]}
    #     self.__testthis()

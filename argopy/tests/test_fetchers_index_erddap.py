import pandas as pd

import pytest
import tempfile

import argopy
from argopy import IndexFetcher as ArgoIndexFetcher
from argopy.errors import (
    FileSystemHasNoCache,
    CacheFileNotFound
)
from utils import requires_connected_erddap_index, safe_to_server_errors, ci_erddap_index

ERDDAP_TIMEOUT = 3 * 60
safe_to_no_cache = pytest.mark.skipif(True, reason="Cache disabled for erddap index fetcher")


@ci_erddap_index
@requires_connected_erddap_index
class Test_Backend_WMO:
    """ Test ERDDAP index fetching backend for WMO access point"""
    src = "erddap"
    requests = {
        "float": [[2901623], [2901623, 6901929]]
    }

    def test_nocache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                fetcher = ArgoIndexFetcher(src=self.src, cache=False).float(self.requests['float'][0]).fetcher
                with pytest.raises(FileSystemHasNoCache):
                    fetcher.cachepath

    @safe_to_no_cache
    def test_cachepath_notfound(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                fetcher = ArgoIndexFetcher(src=self.src, cache=True).float(self.requests['float'][0]).fetcher
                with pytest.raises(CacheFileNotFound):
                    fetcher.cachepath

    @safe_to_no_cache
    @safe_to_server_errors
    def test_cached(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir, api_timeout=ERDDAP_TIMEOUT):
                fetcher = ArgoIndexFetcher(src=self.src, cache=True).float(self.requests['float'][0]).fetcher
                df = fetcher.to_dataframe()
                assert isinstance(df, pd.core.frame.DataFrame)
                assert isinstance(fetcher.cachepath, str)

    @safe_to_no_cache
    @safe_to_server_errors
    def test_clearcache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir, api_timeout=ERDDAP_TIMEOUT):
                fetcher = ArgoIndexFetcher(src=self.src, cache=True).float(self.requests['float'][0]).fetcher
                fetcher.to_dataframe()
                fetcher.clear_cache()
                with pytest.raises(CacheFileNotFound):
                    fetcher.cachepath

    def test_url(self):
        for arg in self.requests["float"]:
            fetcher = ArgoIndexFetcher(src=self.src).float(arg).fetcher
            assert isinstance(fetcher.url, str)

    @safe_to_server_errors
    def test_phy_float(self):
        for arg in self.requests["float"]:
            with argopy.set_options(api_timeout=ERDDAP_TIMEOUT):
                fetcher = ArgoIndexFetcher(src=self.src).float(arg).fetcher
                df = fetcher.to_dataframe()
                assert isinstance(df, pd.core.frame.DataFrame)


@ci_erddap_index
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

    def test_nocache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                fetcher = ArgoIndexFetcher(src=self.src, cache=False).region(self.requests['region'][-1]).fetcher
                with pytest.raises(FileSystemHasNoCache):
                    fetcher.cachepath

    @safe_to_no_cache
    def test_cachepath_notfound(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                fetcher = ArgoIndexFetcher(src=self.src, cache=True).region(self.requests['region'][-1]).fetcher
                with pytest.raises(CacheFileNotFound):
                    fetcher.cachepath

    @safe_to_no_cache
    @safe_to_server_errors
    def test_cached(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir, api_timeout=ERDDAP_TIMEOUT):
                fetcher = ArgoIndexFetcher(src=self.src, cache=True).region(self.requests['region'][-1]).fetcher
                df = fetcher.to_dataframe()
                assert isinstance(df, pd.core.frame.DataFrame)
                assert isinstance(fetcher.cachepath, str)

    @safe_to_no_cache
    @safe_to_server_errors
    def test_clearcache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir, api_timeout=ERDDAP_TIMEOUT):
                fetcher = ArgoIndexFetcher(src=self.src, cache=True).region(self.requests['region'][-1]).fetcher
                fetcher.to_dataframe()
                fetcher.clear_cache()
                with pytest.raises(CacheFileNotFound):
                    fetcher.cachepath

    def test_url(self):
        for arg in self.requests["region"]:
            fetcher = ArgoIndexFetcher(src=self.src).region(arg).fetcher
            assert isinstance(fetcher.url, str)

    @safe_to_server_errors
    def test_phy_region(self):
        for arg in self.requests["region"]:
            with argopy.set_options(api_timeout=ERDDAP_TIMEOUT):
                fetcher = ArgoIndexFetcher(src=self.src).region(arg).fetcher
                df = fetcher.to_dataframe()
                assert isinstance(df, pd.core.frame.DataFrame)

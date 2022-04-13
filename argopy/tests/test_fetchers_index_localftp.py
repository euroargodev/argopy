import pandas as pd
import tempfile

import pytest

import argopy
from argopy import IndexFetcher as ArgoIndexFetcher
from argopy.errors import FileSystemHasNoCache, CacheFileNotFound, DataNotFound
from utils import requires_localftp_index, requires_connection, safe_to_server_errors


@requires_localftp_index
@requires_connection
class Test_Backend:
    """ Test localftp index fetcher """

    src = "localftp"
    local_ftp = argopy.tutorial.open_dataset("localftp")[0]
    requests = {
        "float": [[2901623], [2901623, 6901929]],
        "profile": [[2901623, 12], [6901929, [5, 45]]],
        "region": [
            [-60, -40, 40.0, 60.0],
            [-60, -40, 40.0, 60.0, "2007-08-01", "2007-09-01"],
        ],
    }

    def test_cachepath_notfound(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir, local_ftp=self.local_ftp):
                fetcher = ArgoIndexFetcher(src=self.src, cache=True).profile(*self.requests['profile'][0]).fetcher
                with pytest.raises(CacheFileNotFound):
                    fetcher.cachepath

    def test_nocache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir, local_ftp=self.local_ftp):
                fetcher = ArgoIndexFetcher(src=self.src, cache=False).profile(*self.requests['profile'][0]).fetcher
                with pytest.raises(FileSystemHasNoCache):
                    fetcher.cachepath

    @safe_to_server_errors
    def test_clearcache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir, local_ftp=self.local_ftp):
                fetcher = ArgoIndexFetcher(src=self.src, cache=True).profile(*self.requests['profile'][0]).fetcher
                fetcher.to_dataframe()
                fetcher.clear_cache()
                with pytest.raises(CacheFileNotFound):
                    fetcher.cachepath

    @safe_to_server_errors
    def test_cached(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir, local_ftp=self.local_ftp):
                fetcher = ArgoIndexFetcher(src=self.src, cache=True).float(self.requests['float'][0]).fetcher
                df = fetcher.to_dataframe()
                assert isinstance(df, pd.core.frame.DataFrame)
                assert isinstance(fetcher.cachepath, str)

    def test_noresults(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            with pytest.raises(DataNotFound):
                ArgoIndexFetcher(src=self.src).region(
                    [-70, -65, 30.0, 35.0, "2030-01-01", "2030-06-30"]
                ).fetcher.to_dataframe()

    def __testthis(self):
        for access_point in self.args:

            if access_point == "profile":
                for arg in self.args["profile"]:
                    with argopy.set_options(local_ftp=self.local_ftp):
                        fetcher = ArgoIndexFetcher(src=self.src).profile(*arg).fetcher
                        df = fetcher.to_dataframe()
                        assert isinstance(df, pd.core.frame.DataFrame)

            if access_point == "float":
                for arg in self.args["float"]:
                    with argopy.set_options(local_ftp=self.local_ftp):
                        fetcher = ArgoIndexFetcher(src=self.src).float(arg).fetcher
                        df = fetcher.to_dataframe()
                        assert isinstance(df, pd.core.frame.DataFrame)

            if access_point == "region":
                for arg in self.args["region"]:
                    with argopy.set_options(local_ftp=self.local_ftp):
                        fetcher = ArgoIndexFetcher(src=self.src).region(arg).fetcher
                        df = fetcher.to_dataframe()
                        assert isinstance(df, pd.core.frame.DataFrame)

    @safe_to_server_errors
    def test_phy_float(self):
        self.args = {"float": self.requests["float"]}
        self.__testthis()

    @safe_to_server_errors
    def test_phy_profile(self):
        self.args = {"profile": self.requests["profile"]}
        self.__testthis()

    @safe_to_server_errors
    def test_phy_region(self):
        self.args = {"region": self.requests["region"]}
        self.__testthis()

import pandas as pd
import numpy as np
import tempfile

import pytest

import argopy
from argopy import IndexFetcher as ArgoIndexFetcher
from argopy.errors import FileSystemHasNoCache, CacheFileNotFound, DataNotFound, FtpPathError
from . import requires_localftp_index, requires_connection, safe_to_server_errors, requires_connected_gdac


@requires_connected_gdac
class Test_Backend:
    """ Test GDAC FTP index fetcher backend """

    src = "ftp"
    requests = {
        # "float": [[4902252], [2901746, 4902252]],
        # "profile": [[2901746, 90], [6901929, np.arange(12, 14)]],
        "float": [13857],
        "profile": [[13857, 90]],
        "region": [
            [-20, -16., 0, 1],
            [-60, -58, 40.0, 45.0, "2007-08-01", "2007-09-01"],
        ],
    }

    ftp = ['https://data-argo.ifremer.fr',
     'ftp://ftp.ifremer.fr/ifremer/argo',
     # 'ftp://usgodae.org/pub/outgoing/argo',  # ok, but takes too long to respond, slow down CI
     argopy.tutorial.open_dataset("localftp")[0]]

    def test_nocache(self):
        for this_ftp in self.ftp:
            with tempfile.TemporaryDirectory() as testcachedir:
                with argopy.set_options(cachedir=testcachedir, gdac_ftp=this_ftp):
                    fetcher = ArgoIndexFetcher(src=self.src, cache=False).profile(*self.requests['profile'][0]).fetcher
                    with pytest.raises(FileSystemHasNoCache):
                        fetcher.cachepath

    @safe_to_server_errors
    def test_clearcache(self):
        for this_ftp in self.ftp:
            with tempfile.TemporaryDirectory() as testcachedir:
                with argopy.set_options(cachedir=testcachedir, gdac_ftp=this_ftp):
                    fetcher = ArgoIndexFetcher(src=self.src, cache=True).profile(*self.requests['profile'][0]).fetcher
                    fetcher.to_dataframe()
                    fetcher.clear_cache()
                    with pytest.raises(CacheFileNotFound):
                        fetcher.cachepath

    @safe_to_server_errors
    def test_cached(self):
        for this_ftp in self.ftp:
            with tempfile.TemporaryDirectory() as testcachedir:
                with argopy.set_options(cachedir=testcachedir, gdac_ftp=this_ftp):
                    fetcher = ArgoIndexFetcher(src=self.src, cache=True).float(self.requests['float'][0]).fetcher
                    df = fetcher.to_dataframe()
                    assert isinstance(df, pd.core.frame.DataFrame)
                    assert isinstance(fetcher.cachepath, str)

    def test_noresults(self):
        for this_ftp in self.ftp:
            with argopy.set_options(gdac_ftp=this_ftp):
                with pytest.raises(DataNotFound):
                    ArgoIndexFetcher(src=self.src).region(
                        [-70, -65, 30.0, 35.0, "2030-01-01", "2030-06-30"]
                    ).fetcher.to_dataframe()

    def test_ftp_server(self):
        # Invalid servers:
        with pytest.raises(FtpPathError):
            ArgoIndexFetcher(src='ftp', ftp='invalid').profile(1900857, np.arange(10,20))
        with pytest.raises(FtpPathError):
            ArgoIndexFetcher(src=self.src, ftp='https://invalid_ftp').profile(1900857, np.arange(10,20))
        with pytest.raises(FtpPathError):
            ArgoIndexFetcher(src=self.src, ftp='ftp://invalid_ftp').profile(1900857, np.arange(10,20))

        # Valid list of servers, test all possible host protocols:
        for this_ftp in self.ftp:
            fetcher = ArgoIndexFetcher(src=self.src, ftp=this_ftp).profile(*self.requests["profile"][0]).fetcher
            assert(fetcher.N_RECORDS >= 1)  # Make sure we loaded the index file content

    def _assert_fetcher(self, this_fetcher):
        df = this_fetcher.to_dataframe()
        assert isinstance(df, pd.core.frame.DataFrame)
        if this_fetcher.indexfs.cache:
            assert isinstance(this_fetcher.cachepath, str)

    def __testthis_profile(self, dataset):
        for this_ftp in self.ftp:
            if 'tutorial' not in this_ftp:
                N_RECORDS = 100
            else:
                N_RECORDS = None
            fetcher_args = {"src": self.src, "ds": dataset, "ftp": this_ftp, "N_RECORDS": N_RECORDS}
            for arg in self.args["profile"]:
                fetcher = ArgoIndexFetcher(**fetcher_args).profile(*arg).fetcher
                self._assert_fetcher(fetcher)

    def __testthis_float(self, dataset):
        for this_ftp in self.ftp:
            if 'tutorial' not in this_ftp:
                N_RECORDS = 100
            else:
                N_RECORDS = None
            fetcher_args = {"src": self.src, "ds": dataset, "ftp": this_ftp, "N_RECORDS": N_RECORDS}
            for arg in self.args["float"]:
                fetcher = ArgoIndexFetcher(**fetcher_args).float(arg).fetcher
                self._assert_fetcher(fetcher)

    def __testthis_region(self, dataset):
        for this_ftp in self.ftp:
            if 'tutorial' not in this_ftp:
                N_RECORDS = 100
            else:
                N_RECORDS = None
            fetcher_args = {"src": self.src, "ds": dataset, "ftp": this_ftp, "N_RECORDS": N_RECORDS}
            for arg in self.args["region"]:
                fetcher = ArgoIndexFetcher(**fetcher_args).region(arg).fetcher
                self._assert_fetcher(fetcher)

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

import os
import numpy as np
import xarray as xr
import shutil

import pytest
import unittest

import argopy
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import ArgovisServerError, CacheFileNotFound, FileSystemHasNoCache
from argopy.utilities import (
    list_available_data_src,
    isconnected,
    isAPIconnected,
    is_list_of_strings,
)


argopy.set_options(api_timeout=3 * 60)  # From Github actions, requests can take a while
AVAILABLE_SOURCES = list_available_data_src()
CONNECTED = isconnected()
CONNECTEDAPI = isAPIconnected(src="argovis", data=True)


@unittest.skipUnless("argovis" in AVAILABLE_SOURCES, "requires argovis data fetcher")
@unittest.skipUnless(CONNECTED, "argovis requires an internet connection")
@unittest.skipUnless(CONNECTEDAPI, "argovis API is not alive")
class Backend(unittest.TestCase):
    src = "argovis"
    testcachedir = os.path.expanduser(os.path.join("~", ".argopytest_tmp"))

    def test_cachepath_notfound(self):
        with argopy.set_options(cachedir=self.testcachedir):
            loader = ArgoDataFetcher(src=self.src, cache=True).profile(6902746, 34)
            with pytest.raises(CacheFileNotFound):
                loader.fetcher.cachepath
        shutil.rmtree(self.testcachedir)  # Make sure the cache is left empty

    def test_nocache(self):
        with argopy.set_options(cachedir="dummy"):
            loader = ArgoDataFetcher(src=self.src, cache=False).profile(6902746, 34)
            loader.to_xarray()
            with pytest.raises(FileSystemHasNoCache):
                loader.fetcher.cachepath

    def test_caching_float(self):
        with argopy.set_options(cachedir=self.testcachedir):
            try:
                loader = ArgoDataFetcher(src=self.src, cache=True).float(1901393)
                # 1st call to load and save to cachedir:
                ds = loader.to_xarray()
                # 2nd call to load from cached file:
                ds = loader.to_xarray()
                print(loader.fetcher.cachepath)
                assert isinstance(ds, xr.Dataset)
                assert isinstance(loader.fetcher.cachepath, str)
                shutil.rmtree(self.testcachedir)
            except ArgovisServerError:  # Test is passed when something goes wrong because of the argovis server, not our fault !
                shutil.rmtree(self.testcachedir)
                pass
            except Exception:
                shutil.rmtree(self.testcachedir)
                raise

    def test_caching_profile(self):
        with argopy.set_options(cachedir=self.testcachedir):
            loader = ArgoDataFetcher(src=self.src, cache=True).profile(6902746, 34)
            try:
                # 1st call to load and save to cachedir:
                ds = loader.to_xarray()
                # 2nd call to load from cached file
                ds = loader.to_xarray()
                assert isinstance(ds, xr.Dataset)
                assert isinstance(loader.fetcher.cachepath, str)
                shutil.rmtree(self.testcachedir)
            except ArgovisServerError:  # Test is passed when something goes wrong because of the argovis server, not our fault !
                shutil.rmtree(self.testcachedir)
                pass
            except Exception:
                shutil.rmtree(self.testcachedir)
                raise

    def test_caching_region(self):
        with argopy.set_options(cachedir=self.testcachedir):
            loader = ArgoDataFetcher(src=self.src, cache=True).region(
                [-40, -30, 30, 40, 0, 100, "2011", "2012"]
            )
            try:
                # 1st call to load and save to cachedir:
                ds = loader.to_xarray()
                # 2nd call to load from cached file
                ds = loader.to_xarray()
                assert isinstance(ds, xr.Dataset)
                assert isinstance(loader.fetcher.cachepath, str)
                shutil.rmtree(self.testcachedir)
            except ArgovisServerError:  # Test is passed when something goes wrong because of the argovis server, not our fault !
                shutil.rmtree(self.testcachedir)
                pass
            except Exception:
                shutil.rmtree(self.testcachedir)
                raise

    def __testthis_profile(self, dataset):
        fetcher_args = {"src": self.src, "ds": dataset}
        for arg in self.args["profile"]:
            try:
                f = ArgoDataFetcher(**fetcher_args).profile(*arg)
                assert isinstance(f.fetcher.uri, str)
                assert isinstance(f.to_xarray(), xr.Dataset)
            except ArgovisServerError:
                # Test is passed when something goes wrong because of the argovis server, not our fault !
                pass
            except Exception:
                print("ARGOVIS request:\n", f.fetcher.uri)
                pass

    def __testthis_float(self, dataset):
        fetcher_args = {"src": self.src, "ds": dataset}
        for arg in self.args["float"]:
            try:
                f = ArgoDataFetcher(**fetcher_args).float(arg)
                assert isinstance(f.fetcher.uri, str)
                assert isinstance(f.to_xarray(), xr.Dataset)
            except ArgovisServerError:
                # Test is passed when something goes wrong because of the argovis server, not our fault !
                pass
            except Exception:
                print("ARGOVIS request:\n", f.fetcher.uri)
                pass

    def __testthis_region(self, dataset):
        fetcher_args = {"src": self.src, "ds": dataset}
        for arg in self.args["region"]:
            try:
                f = ArgoDataFetcher(**fetcher_args).region(arg)
                assert isinstance(f.fetcher.uri, str)
                assert isinstance(f.to_xarray(), xr.Dataset)
            except ArgovisServerError:
                # Test is passed when something goes wrong because of the argovis server, not our fault !
                pass
            except Exception:
                print("ARGOVIS request:\n", f.fetcher.uri)
                pass

    def __testthis(self, dataset):
        for access_point in self.args:
            if access_point == "profile":
                self.__testthis_profile(dataset)
            elif access_point == "float":
                self.__testthis_float(dataset)
            elif access_point == "region":
                self.__testthis_region(dataset)

    def test_phy_float(self):
        self.args = {}
        self.args["float"] = [[1901393], [1901393, 6902746]]
        self.__testthis("phy")

    def test_phy_profile(self):
        self.args = {}
        self.args["profile"] = [
            [6902746, 34],
            [6902746, np.arange(12, 13)],
            [6902746, [1, 12]],
        ]
        self.__testthis("phy")

    def test_phy_region(self):
        self.args = {}
        self.args["region"] = [
            [-70, -65, 35.0, 40.0, 0, 10.0],
            [-70, -65, 35.0, 40.0, 0, 10.0, "2012-01", "2013-12"],
        ]
        self.__testthis("phy")


@unittest.skipUnless("argovis" in AVAILABLE_SOURCES, "requires argovis data fetcher")
@unittest.skipUnless(CONNECTED, "argovis requires an internet connection")
@unittest.skipUnless(CONNECTEDAPI, "argovis API is not alive")
class BackendParallel(unittest.TestCase):
    """ This test backend for parallel requests """

    src = "argovis"
    requests = {}
    requests["region"] = [
        [-40, -30, 30, 40, 0, 100],
        [-40, -30, 30, 40, 0, 100, "2011", "2012"],
    ]
    requests["wmo"] = [
        5900446,
        5906072,
        6901929,
        1900857,
        3902131,
        2902696,
    ]

    def test_chunks_region(self):
        for access_arg in self.requests["region"]:
            fetcher_args = {"src": self.src, "parallel": True}
            try:
                f = ArgoDataFetcher(**fetcher_args).region(access_arg)
                assert is_list_of_strings(f.fetcher.uri)
                assert isinstance(f.to_xarray(), xr.Dataset)
            except ArgovisServerError:
                # Test is passed when something goes wrong because of the argovis server, not our fault !
                pass
            except Exception:
                print("ARGOVIS request:\n", f.fetcher.uri)
                pass

    def test_chunks_wmo(self):
        for access_arg in self.requests["wmo"]:
            fetcher_args = {"src": self.src, "parallel": True}
            try:
                f = ArgoDataFetcher(**fetcher_args).float(access_arg)
                assert is_list_of_strings(f.fetcher.uri)
                assert isinstance(f.to_xarray(), xr.Dataset)
            except ArgovisServerError:
                # Test is passed when something goes wrong because of the argovis server, not our fault !
                pass
            except Exception:
                print("ARGOVIS request:\n", f.fetcher.uri)
                pass


if __name__ == "__main__":
    unittest.main()

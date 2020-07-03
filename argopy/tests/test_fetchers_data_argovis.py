#!/bin/env python
# -*coding: UTF-8 -*-
#
# Test data fetchers
#
# This is not designed as it should
# We need to have:
# - one class to test the facade API
# - one class to test specific methods of each backends
#
# At this point, we are testing real data fetching both through facade and through direct call to backends.

import os
import numpy as np
import xarray as xr
import shutil

import pytest
import unittest
from unittest import TestCase

import argopy
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import InvalidFetcher, ArgovisServerError, CacheFileNotFound, FileSystemHasNoCache
from argopy.utilities import list_available_data_src, isconnected, isAPIconnected


argopy.set_options(api_timeout=3*60)  # From Github actions, requests can take a while
AVAILABLE_SOURCES = list_available_data_src()
CONNECTED = isconnected()
CONNECTEDAPI = isAPIconnected(src='argovis', data=True)


@unittest.skipUnless('argovis' in AVAILABLE_SOURCES, "requires argovis data fetcher")
@unittest.skipUnless(CONNECTED, "argovis requires an internet connection")
@unittest.skipUnless(CONNECTEDAPI, "argovis API is not alive")
class Backend(TestCase):
    """ Test main API facade for all available dataset and access points of the ERDDAP fetching backend """
    src = 'argovis'
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
                # 1st call to load from argovis and init cachedir:
                ds = loader.to_xarray()
                # 2nd call to load from cached file:
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

    def test_caching_profile(self):
        with argopy.set_options(cachedir=self.testcachedir):
            loader = ArgoDataFetcher(src=self.src, cache=True).profile(6902746, 34)
            try:
                # 1st call to load from argovis and save to cachedir:
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
        for arg in self.args['profile']:
            try:
                ds = ArgoDataFetcher(src=self.src, ds=dataset).profile(*arg).to_xarray()
                assert isinstance(ds, xr.Dataset)
            except ArgovisServerError:
                # Test is passed when something goes wrong because of the argovis server, not our fault !
                pass
            except Exception:
                print("ARGOVIS request:\n",
                      ArgoDataFetcher(src=self.src, ds=dataset).profile(*arg).fetcher.url)
                pass

    def __testthis_float(self, dataset):
        for arg in self.args['float']:
            try:
                ds = ArgoDataFetcher(src=self.src, ds=dataset).float(arg).to_xarray()
                assert isinstance(ds, xr.Dataset)
            except ArgovisServerError:
                # Test is passed when something goes wrong because of the argovis server, not our fault !
                pass
            except Exception:
                print("ARGOVIS request:\n",
                      ArgoDataFetcher(src=self.src, ds=dataset).float(arg).fetcher.url)
                pass

    def __testthis_region(self, dataset):
        for arg in self.args['region']:
            try:
                ds = ArgoDataFetcher(src=self.src, ds=dataset).region(arg).to_xarray()
                assert isinstance(ds, xr.Dataset)
            except ArgovisServerError:
                # Test is passed when something goes wrong because of the argovis server, not our fault !
                pass
            except Exception:
                print("ARGOVIS request:\n",
                      ArgoDataFetcher(src=self.src, ds=dataset).region(arg).fetcher.url)
                pass

    def __testthis(self, dataset):
        for access_point in self.args:
            if access_point == 'profile':
                self.__testthis_profile(dataset)
            elif access_point == 'float':
                self.__testthis_float(dataset)
            elif access_point == 'region':
                self.__testthis_region(dataset)

    def test_phy_float(self):
        self.args = {}
        self.args['float'] = [[1901393],
                              [1901393, 6902746]]
        self.__testthis('phy')

    def test_phy_profile(self):
        self.args = {}
        self.args['profile'] = [[6902746, 34],
                                [6902746, np.arange(12, 13)], [6902746, [1, 12]]]
        self.__testthis('phy')

    def test_phy_region(self):
        self.args = {}
        self.args['region'] = [[-70, -65, 35., 40., 0, 10.],
                               [-70, -65, 35., 40., 0, 10., '2012-01', '2013-12']]
        self.__testthis('phy')

if __name__ == '__main__':
    unittest.main()

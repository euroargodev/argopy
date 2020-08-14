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
from argopy.errors import InvalidFetcher, ErddapServerError, CacheFileNotFound, FileSystemHasNoCache, FtpPathError
from argopy.utilities import list_available_data_src, isconnected, isAPIconnected, erddap_ds_exists

AVAILABLE_SOURCES = list_available_data_src()


@unittest.skipUnless('localftp' in AVAILABLE_SOURCES, "requires localftp data fetcher")
class Backend(TestCase):
    """ Test main API facade for all available dataset and access points of the localftp fetching backend """
    src = 'localftp'
    testcachedir = os.path.expanduser(os.path.join("~", ".argopytest_tmp"))
    local_ftp = argopy.tutorial.open_dataset('localftp')[0]

    def test_cachepath_notfound(self):
        with argopy.set_options(cachedir=self.testcachedir, local_ftp=self.local_ftp):
            loader = ArgoDataFetcher(src=self.src, cache=True).profile(2901623, 12)
            with pytest.raises(CacheFileNotFound):
                loader.fetcher.cachepath
        shutil.rmtree(self.testcachedir)  # Make sure the cache is left empty

    def test_nocache(self):
        with argopy.set_options(cachedir="dummy", local_ftp=self.local_ftp):
            loader = ArgoDataFetcher(src=self.src, cache=False).profile(2901623, 12)
            loader.to_xarray()
            with pytest.raises(FileSystemHasNoCache):
                loader.fetcher.cachepath

    def test_caching_float(self):
        with argopy.set_options(cachedir=self.testcachedir, local_ftp=self.local_ftp):
            try:
                loader = ArgoDataFetcher(src=self.src, cache=True).float(2901623)
                # 1st call to load data and init cachedir:
                ds = loader.to_xarray()
                # 2nd call to load from cached file:
                ds = loader.to_xarray()
                assert isinstance(ds, xr.Dataset)
                assert np.all([isinstance(path, str) for path in loader.fetcher.cachepath])
                shutil.rmtree(self.testcachedir)
            except Exception:
                shutil.rmtree(self.testcachedir)
                raise

    def test_caching_profile(self):
        with argopy.set_options(cachedir=self.testcachedir, local_ftp=self.local_ftp):
            loader = ArgoDataFetcher(src=self.src, cache=True).profile(2901623, 1)
            try:
                # 1st call to load from argovis and save to cachedir:
                ds = loader.to_xarray()
                # 2nd call to load from cached file
                ds = loader.to_xarray()
                assert isinstance(ds, xr.Dataset)
                assert np.all([isinstance(path, str) for path in loader.fetcher.cachepath])
                shutil.rmtree(self.testcachedir)
            except Exception:
                shutil.rmtree(self.testcachedir)
                raise

    def test_invalidFTPpath(self):
        with pytest.raises(ValueError):
            with argopy.set_options(local_ftp="dummy"):
                ArgoDataFetcher(src=self.src).profile(2901623, 12)

        with pytest.raises(FtpPathError):
            with argopy.set_options(local_ftp=os.path.sep.join([self.local_ftp, "dac"])):
                ArgoDataFetcher(src=self.src).profile(2901623, 12)

    def __testthis_profile(self, dataset):
        with argopy.set_options(local_ftp=self.local_ftp):
            for arg in self.args['profile']:
                try:
                    ds = ArgoDataFetcher(src=self.src, ds=dataset).profile(*arg).to_xarray()
                    assert isinstance(ds, xr.Dataset)
                except ErddapServerError:
                    # Test is passed when something goes wrong because of the erddap server, not our fault !
                    pass
                except Exception:
                    print("ERROR LOCALFTP request:\n",
                          ArgoDataFetcher(src=self.src, ds=dataset).profile(*arg).fetcher.uri)
                    pass

    def __testthis_float(self, dataset):
        with argopy.set_options(local_ftp=self.local_ftp):
            for arg in self.args['float']:
                try:
                    ds = ArgoDataFetcher(src=self.src, ds=dataset).float(arg).to_xarray()
                    assert isinstance(ds, xr.Dataset)
                except ErddapServerError:
                    # Test is passed when something goes wrong because of the erddap server, not our fault !
                    pass
                except Exception:
                    print("ERROR LOCALFTP request:\n",
                          ArgoDataFetcher(src=self.src, ds=dataset).float(arg).fetcher.uri)
                    pass

    def __testthis_region(self, dataset):
        with argopy.set_options(local_ftp=self.local_ftp):
            for arg in self.args['region']:
                try:
                    ds = ArgoDataFetcher(src=self.src, ds=dataset).region(arg).to_xarray()
                    assert isinstance(ds, xr.Dataset)
                except ErddapServerError:
                    # Test is passed when something goes wrong because of the erddap server, not our fault !
                    pass
                except Exception:
                    print("ERROR LOCALFTP request:\n",
                          ArgoDataFetcher(src=self.src, ds=dataset).region(arg).fetcher.uri)
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
        self.args['float'] = [[2901623],
                              [6901929, 2901623]]
        self.__testthis('phy')

    def test_phy_profile(self):
        self.args = {}
        self.args['profile'] = [[2901623, 12],
                                [2901623, np.arange(12, 14)],
                                [2901623, [1, 6]]]
        self.__testthis('phy')

    def test_phy_region(self):
        self.args = {}
        self.args['region'] = [[-60, -40, 40., 60., 0., 100.],
                               [-60, -40, 40., 60., 0., 100., '2007-08-01', '2007-09-01']]
        self.__testthis('phy')


if __name__ == '__main__':
    unittest.main()

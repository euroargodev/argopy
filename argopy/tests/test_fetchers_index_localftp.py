#!/bin/env python
# -*coding: UTF-8 -*-
#
# Test data fetchers
#

import os
import xarray as xr
import shutil

import pytest
import unittest
from unittest import TestCase

import argopy
from argopy import IndexFetcher as ArgoIndexFetcher
from argopy.errors import InvalidFetcher, FileSystemHasNoCache, CacheFileNotFound, DataNotFound
from argopy.utilities import list_available_index_src, isconnected, isAPIconnected, erddap_ds_exists


argopy.set_options(api_timeout=3*60)  # From Github actions, requests can take a while
AVAILABLE_INDEX_SOURCES = list_available_index_src()
CONNECTED = isconnected()
CONNECTEDAPI = {src: False for src in AVAILABLE_INDEX_SOURCES.keys()}
if CONNECTED:
    DSEXISTS = erddap_ds_exists(ds="ArgoFloats-index")
    for src in AVAILABLE_INDEX_SOURCES.keys():
        try:
            CONNECTEDAPI[src] = isAPIconnected(src=src, data=False)
        except InvalidFetcher:
            pass
else:
    DSEXISTS = False


@unittest.skipUnless('localftp' in AVAILABLE_INDEX_SOURCES, "requires localftp index fetcher")
class Backend(TestCase):
    """ Test localftp index fetcher """
    src = 'localftp'
    ftproot, flist = argopy.tutorial.open_dataset('localftp')
    local_ftp = ftproot

    def test_cachepath_notfound(self):
        testcachedir = os.path.expanduser(os.path.join("~", ".argopytest_tmp"))
        with argopy.set_options(cachedir=testcachedir, local_ftp=self.local_ftp):
            loader = ArgoIndexFetcher(src=self.src, cache=True).profile(2901623, 2)
            with pytest.raises(CacheFileNotFound):
                loader.fetcher.cachepath
        shutil.rmtree(testcachedir)  # Make sure the cache folder is cleaned

    def test_nocache(self):
        with argopy.set_options(cachedir="dummy", local_ftp=self.local_ftp):
            loader = ArgoIndexFetcher(src=self.src, cache=False).profile(2901623, 2)
            loader.to_xarray()
            with pytest.raises(FileSystemHasNoCache):
                loader.fetcher.cachepath

    def test_caching_float(self):
        testcachedir = os.path.expanduser(os.path.join("~", ".argopytest_tmp"))
        with argopy.set_options(cachedir=testcachedir, local_ftp=self.local_ftp):
            try:
                loader = ArgoIndexFetcher(src=self.src, cache=True).float(6901929)
                # 1st call to load from erddap and save to cachedir:
                ds = loader.to_xarray()
                # 2nd call to load from cached file:
                ds = loader.to_xarray()
                assert isinstance(ds, xr.Dataset)
                assert isinstance(loader.fetcher.cachepath, str)
                shutil.rmtree(testcachedir)
            except Exception:
                shutil.rmtree(testcachedir)
                raise

    def test_noresults(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            with pytest.raises(DataNotFound):
                ArgoIndexFetcher(src=self.src).region([-70, -65, 30., 35., '2030-01-01', '2030-06-30']).to_dataframe()

    def __testthis(self, dataset):
        for access_point in self.args:

            if access_point == 'profile':
                for arg in self.args['profile']:
                    with argopy.set_options(local_ftp=self.local_ftp):
                        try:
                            ds = ArgoIndexFetcher(src=self.src).profile(*arg).to_xarray()
                            assert isinstance(ds, xr.Dataset)
                        except Exception:
                            print("ERROR LOCALFTP request:\n",
                                  ArgoIndexFetcher(src=self.src).profile(*arg).fetcher.cname())
                            pass

            if access_point == 'float':
                for arg in self.args['float']:
                    with argopy.set_options(local_ftp=self.local_ftp):
                        try:
                            ds = ArgoIndexFetcher(src=self.src).float(arg).to_xarray()
                            assert isinstance(ds, xr.Dataset)
                        except Exception:
                            print("ERROR LOCALFTP request:\n",
                                  ArgoIndexFetcher(src=self.src).float(arg).fetcher.cname())
                            pass

            if access_point == 'region':
                for arg in self.args['region']:
                    with argopy.set_options(local_ftp=self.local_ftp):
                        try:
                            ds = ArgoIndexFetcher(src=self.src).region(arg).to_xarray()
                            assert isinstance(ds, xr.Dataset)
                        except Exception:
                            print("ERROR LOCALFTP request:\n",
                                  ArgoIndexFetcher(src=self.src).region(arg).fetcher.cname())
                            pass

    def test_phy_float(self):
        self.args = {}
        self.args['float'] = [[2901623],
                              [2901623, 6901929]]
        self.__testthis('phy')

    def test_phy_profile(self):
        self.args = {}
        self.args['profile'] = [[6901929, 36],
                                [6901929, [5, 45]]]
        self.__testthis('phy')

    def test_phy_region(self):
        self.args = {}
        self.args['region'] = [[-60, -40, 40., 60.],
                               [-60, -40, 40., 60., '2007-08-01', '2007-09-01']]
        self.__testthis('phy')


if __name__ == '__main__':
    unittest.main()

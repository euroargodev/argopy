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
from argopy.errors import InvalidFetcher, FileSystemHasNoCache, CacheFileNotFound, ErddapServerError
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


@unittest.skipUnless('erddap' in AVAILABLE_INDEX_SOURCES, "requires erddap index fetcher")
@unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
@unittest.skipUnless(DSEXISTS, "erddap requires a valid core index Argo dataset from Ifremer server")
# @unittest.skipUnless(False, "Waiting for https://github.com/euroargodev/argopy/issues/16")
@unittest.skipUnless(CONNECTEDAPI['erddap'], "erddap API is not alive")
class Backend(TestCase):
    """ Test main API facade for all available dataset of the ERDDAP index fetching backend """
    testcachedir = os.path.expanduser(os.path.join("~", ".argopytest_tmp"))

    def test_cachepath_notfound(self):
        with argopy.set_options(cachedir=self.testcachedir):
            loader = ArgoIndexFetcher(src='erddap', cache=True).float(6902746)
            with pytest.raises(CacheFileNotFound):
                loader.fetcher.cachepath
        shutil.rmtree(self.testcachedir)  # Make sure the cache is empty

    @unittest.skipUnless(False, "Waiting for https://github.com/euroargodev/argopy/issues/16")
    def test_nocache(self):
        with argopy.set_options(cachedir=self.testcachedir):
            loader = ArgoIndexFetcher(src='erddap', cache=False).float(6902746)
            loader.to_xarray()
            with pytest.raises(FileSystemHasNoCache):
                loader.fetcher.cachepath
        shutil.rmtree(self.testcachedir)  # Make sure the cache is empty

    @unittest.skipUnless(False, "Waiting for https://github.com/euroargodev/argopy/issues/16")
    def test_caching_index(self):
        with argopy.set_options(cachedir=self.testcachedir):
            try:
                loader = ArgoIndexFetcher(src='erddap', cache=True).float(6902746)
                # 1st call to load from erddap and save to cachedir:
                ds = loader.to_xarray()
                # 2nd call to load from cached file:
                ds = loader.to_xarray()
                assert isinstance(ds, xr.Dataset)
                assert isinstance(loader.fetcher.cachepath, str)
                shutil.rmtree(self.testcachedir)
            except ErddapServerError:  # Test is passed when something goes wrong because of the erddap server, not our fault !
                shutil.rmtree(self.testcachedir)
                pass
            except Exception:
                shutil.rmtree(self.testcachedir)
                raise

    def test_url(self):
        loader = ArgoIndexFetcher(src='erddap', cache=True).float(2901623)
        assert isinstance(loader.fetcher.url, str)
        # loader = ArgoIndexFetcher(src='erddap', cache=True).profile(2901623, 12)
        # assert isinstance(loader.fetcher.url, str)
        loader = ArgoIndexFetcher(src='erddap', cache=True).region([-60, -40, 40., 60., '2007-08-01', '2007-09-01'])
        assert isinstance(loader.fetcher.url, str)


if __name__ == '__main__':
    unittest.main()

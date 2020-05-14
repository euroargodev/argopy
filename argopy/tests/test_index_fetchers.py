#!/bin/env python
# -*coding: UTF-8 -*-
#
# Test data fetchers
#

import os
import sys
import numpy as np
import xarray as xr
import shutil

import pytest
import unittest
from unittest import TestCase

import argopy
from argopy import IndexFetcher as ArgoIndexFetcher
from argopy.errors import InvalidFetcherAccessPoint, InvalidFetcher

from argopy.utilities import list_available_data_src, isconnected, erddap_ds_exists
AVAILABLE_SOURCES = list_available_data_src()
CONNECTED = isconnected()
if CONNECTED:
    DSEXISTS = erddap_ds_exists(ds="ArgoFloats-index")
else:
    DSEXISTS = False

def test_invalid_accesspoint():
    with pytest.raises(InvalidFetcherAccessPoint):
        ArgoIndexFetcher().invalid_accesspoint.to_xarray()

def test_invalid_fetcher():
    with pytest.raises(InvalidFetcher):
        ArgoIndexFetcher().to_xarray() # Can't get data if access point not defined first

@unittest.skipUnless('localftp' in AVAILABLE_SOURCES, "requires localftp data fetcher")
def test_unavailable_accesspoint():
    with pytest.raises(InvalidFetcherAccessPoint):
        ArgoIndexFetcher(src='localftp').region([-85,-45,10.,20.,0,100.]).to_xarray()

class EntryPoints_AllBackends(TestCase):
    """ Test main API facade for all available index fetching backends """

    def setUp(self):
        # todo Determine the list of output format to test
        # what else beyond .to_xarray() ?

        self.fetcher_opts = {}

        # Define API entry point options to tests:
        self.args = {}
        self.args['float'] = [[1900033],
                              [6901929, 3902131]]
        self.args['region'] = [[-70, -65, 30., 35.],
                               [-70, -65, 30., 35., '2012-01-01', '2012-06-30']]

    def __test_float(self, bk, **ftc_opts):
        """ Test float index fetching for a given backend """
        for arg in self.args['float']:
            options = {**self.fetcher_opts, **ftc_opts}
            ds = ArgoIndexFetcher(src=bk, **options).float(arg).to_xarray()
            assert isinstance(ds, xr.Dataset) == True

    def __test_region(self, bk):
        """ Test float index fetching for a given backend """
        for arg in self.args['region']:
            ds = ArgoIndexFetcher(src=bk).region(arg).to_xarray()
            assert isinstance(ds, xr.Dataset) == True

    @unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
    @unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core index Argo dataset from Ifremer server")
    @unittest.skipUnless(False, "Waiting for https://github.com/euroargodev/argopy/issues/16")
    def test_float_index_erddap(self):
        self.__test_float('erddap')

    @unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
    @unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core index Argo dataset from Ifremer server")
    @unittest.skipUnless(False, "Waiting for https://github.com/euroargodev/argopy/issues/16")
    def test_region_index_erddap(self):
        self.__test_region('erddap')

    @unittest.skipUnless('localftp' in AVAILABLE_SOURCES, "requires localftp data fetcher")
    def test_float_index_localftp(self):
        ftproot, findex = argopy.tutorial.open_dataset('global_index_prof')
        with argopy.set_options(local_ftp=ftproot):
            self.__test_float('localftp', index_file='ar_index_global_prof.txt')

@unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
@unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
@unittest.skipUnless(DSEXISTS, "erddap requires a valid core index Argo dataset from Ifremer server")
@unittest.skipUnless(False, "Waiting for https://github.com/euroargodev/argopy/issues/16")
class Erddap_backend(TestCase):
    """ Test main API facade for all available dataset of the ERDDAP index fetching backend """

    def test_cachepath_index(self):
        assert isinstance(ArgoIndexFetcher(src='erddap').float(6902746).fetcher.cachepath, str) == True

    def test_caching_index(self):
        cachedir = os.path.expanduser(os.path.join("~", ".argopytest_tmp"))
        try:
            # 1st call to load index from erddap and save to cachedir:
            ds = ArgoIndexFetcher(src='erddap', cache=True, cachedir=cachedir).float(6902746).to_xarray()
            # 2nd call to load from cached file
            ds = ArgoIndexFetcher(src='erddap', cache=True, cachedir=cachedir).float(6902746).to_xarray()
            assert isinstance(ds, xr.Dataset) == True
            shutil.rmtree(cachedir)
        except:
            shutil.rmtree(cachedir)
            raise

if __name__ == '__main__':
    unittest.main()
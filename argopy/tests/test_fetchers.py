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
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import InvalidFetcherAccessPoint

from argopy.utilities import list_available_data_backends
AVAILABLE_BACKENDS = list_available_data_backends()

import urllib.request
def connected(host='http://www.ifremer.fr'):
    try:
        urllib.request.urlopen(host) #Python 3.x
        return True
    except:
        return False
CONNECTED = connected()

# List tests:
def test_invalid_accesspoint():
    with pytest.raises(InvalidFetcherAccessPoint):
        ArgoDataFetcher().invalid_accesspoint.to_xarray()

@unittest.skipUnless('localftp' in AVAILABLE_BACKENDS, "requires localftp data fetcher")
def test_unavailable_accesspoint():
    with pytest.raises(InvalidFetcherAccessPoint):
        ArgoDataFetcher(backend='localftp').region([-85,-45,10.,20.,0,100.]).to_xarray()

class EntryPoints_AllBackends(TestCase):
    """ Test main API facade for all available fetching backends and default dataset """

    def setUp(self):
        #todo Determine the list of output format to test
        # what else beyond .to_xarray() ?

        self.fetcher_opts = {}

        # Define API entry point options to tests:
        self.args = {}
        self.args['float'] = [[5900446],
                              [6901929, 3902131]]
        self.args['profile'] = [[2902696, 12],
                                [2902269, np.arange(12, 14)], [2901746, [1, 6]]]
        self.args['region'] = [[-70, -65, 30., 35., 0, 10.],
                               [-70, -65, 30., 35., 0, 10., '2012-01-01', '2012-06-30']]

    def __test_float(self, bk, **ftc_opts):
        """ Test float for a given backend """
        for arg in self.args['float']:
            options = {**self.fetcher_opts, **ftc_opts}
            ds = ArgoDataFetcher(backend=bk, **options).float(arg).to_xarray()
            assert isinstance(ds, xr.Dataset) == True

    def __test_profile(self, bk):
        """ Test float for a given backend """
        for arg in self.args['profile']:
            ds = ArgoDataFetcher(backend=bk).profile(*arg).to_xarray()
            assert isinstance(ds, xr.Dataset) == True

    def __test_region(self, bk):
        """ Test float for a given backend """
        for arg in self.args['region']:
            ds = ArgoDataFetcher(backend=bk).region(arg).to_xarray()
            assert isinstance(ds, xr.Dataset) == True

    @unittest.skipUnless('erddap' in AVAILABLE_BACKENDS, "requires erddap data fetcher")
    @unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
    def test_float_erddap(self):
        self.__test_float('erddap')

    @unittest.skipUnless('erddap' in AVAILABLE_BACKENDS, "requires erddap data fetcher")
    @unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
    def test_profile_erddap(self):
        self.__test_profile('erddap')

    @unittest.skipUnless('erddap' in AVAILABLE_BACKENDS, "requires erddap data fetcher")
    @unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
    def test_region_erddap(self):
        self.__test_region('erddap')

    @unittest.skipUnless('localftp' in AVAILABLE_BACKENDS, "requires localftp data fetcher")
    def test_float_localftp(self):
        ftproot, flist = argopy.tutorial.open_dataset('localftp')
        with argopy.set_options(local_ftp=os.path.join(ftproot,'dac')):
            self.__test_float('localftp', )
    
    @unittest.skipUnless('localftp' in AVAILABLE_BACKENDS, "requires localftp data fetcher")
    def test_profile_localftp(self):
        ftproot, flist = argopy.tutorial.open_dataset('localftp')
        with argopy.set_options(local_ftp=os.path.join(ftproot,'dac')):
            self.__test_profile('localftp')

    @unittest.skipUnless('argovis' in AVAILABLE_BACKENDS, "requires argovis data fetcher")
    @unittest.skipUnless(CONNECTED, "argovis requires an internet connection")
    def test_float_argovis(self):
        self.__test_float('argovis')

@unittest.skipUnless('erddap' in AVAILABLE_BACKENDS, "requires erddap data fetcher")
@unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
class Erddap_backend(TestCase):
    """ Test main API facade for all available dataset of the ERDDAP fetching backend """

    def test_cachepath(self):
        assert isinstance(ArgoDataFetcher(backend='erddap').profile(6902746, 34).fetcher.cachepath, str) == True

    def test_caching(self):
        cachedir = os.path.expanduser(os.path.join("~",".argopytest_tmp"))
        # 1st call to load from erddap and save to cachedir:
        ds = ArgoDataFetcher(backend='erddap', cache=True, cachedir=cachedir).profile(6902746, 34).to_xarray()
        # 2nd call to load from cached file
        ds = ArgoDataFetcher(backend='erddap', cache=True, cachedir=cachedir).profile(6902746, 34).to_xarray()
        assert isinstance(ds, xr.Dataset) == True
        shutil.rmtree(cachedir)

    def __testthis(self, dataset):
        for access_point in self.args:

            if access_point == 'profile':
                for arg in self.args['profile']:
                    try:
                        ds = ArgoDataFetcher(backend='erddap', ds=dataset).profile(*arg).to_xarray()
                        assert isinstance(ds, xr.Dataset) == True
                    except:
                        print("ERDDAP request:\n",
                              ArgoDataFetcher(backend='erddap', ds=dataset).profile(*arg).fetcher.url)
                        pass

            if access_point == 'float':
                for arg in self.args['float']:
                    try:
                        ds = ArgoDataFetcher(backend='erddap', ds=dataset).float(arg).to_xarray()
                        assert isinstance(ds, xr.Dataset) == True
                    except:
                        print("ERDDAP request:\n",
                              ArgoDataFetcher(backend='erddap', ds=dataset).float(arg).fetcher.url)
                        pass

            if access_point == 'region':
                for arg in self.args['region']:
                    try:
                        ds = ArgoDataFetcher(backend='erddap', ds=dataset).region(arg).to_xarray()
                        assert isinstance(ds, xr.Dataset) == True
                    except:
                        print("ERDDAP request:\n",
                              ArgoDataFetcher(backend='erddap', ds=dataset).region(arg).fetcher.url)
                        pass

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

    def test_bgc_float(self):
        self.args = {}
        self.args['float'] = [[5903248],
                              [7900596, 2902264]]
        self.__testthis('bgc')

    def test_bgc_profile(self):
        self.args = {}
        self.args['profile'] = [[5903248, 34],
                                [5903248, np.arange(12, 14)], [5903248, [1, 12]]]
        self.__testthis('bgc')

    def test_bgc_region(self):
        self.args = {}
        self.args['region'] = [[-70, -65, 35., 40., 0, 10.],
                               [-70, -65, 35., 40., 0, 10., '2012-01-1', '2012-12-31']]
        self.__testthis('bgc')

    def test_ref_region(self):
        self.args = {}
        self.args['region'] = [[-70, -65, 35., 40., 0, 10.],
                               [-70, -65, 35., 40., 0, 10., '2012-01-01', '2012-12-31']]
        self.__testthis('ref')

@unittest.skipUnless('localftp' in AVAILABLE_BACKENDS, "requires localftp data fetcher")
class LocalFTP_DataSets(TestCase):
    """ Test main API facade for all available dataset of the localftp fetching backend """

    def __testthis(self, dataset):
        ftproot, flist = argopy.tutorial.open_dataset('localftp')
        self.local_ftp = os.path.join(ftproot, 'dac')
        for access_point in self.args:

            if access_point == 'profile':
                for arg in self.args['profile']:
                    with argopy.set_options(local_ftp=self.local_ftp):
                        try:
                            ds = ArgoDataFetcher(backend='localftp', ds=dataset).profile(*arg).to_xarray()
                            assert isinstance(ds, xr.Dataset) == True
                        except:
                            print("LOCALFTP request:\n",
                                  ArgoDataFetcher(backend='localftp', ds=dataset).profile(*arg).fetcher.files)
                            pass

            if access_point == 'float':
                for arg in self.args['float']:
                    with argopy.set_options(local_ftp=self.local_ftp):
                        try:
                            ds = ArgoDataFetcher(backend='localftp', ds=dataset).float(arg).to_xarray()
                            assert isinstance(ds, xr.Dataset) == True
                        except:
                            print("LOCALFTP request:\n",
                                  ArgoDataFetcher(backend='localftp', ds=dataset).float(arg).fetcher.files)
                            pass

    def test_phy_float(self):
        self.args = {}
        self.args['float'] = [[5900446],
                              [6901929, 3902131]]
        self.__testthis('phy')

    def test_phy_profile(self):
        self.args = {}
        self.args['profile'] = [[2902696, 12],
                                [2902269, np.arange(12, 14)],
                                [2901746, [1, 6]]]
        self.__testthis('phy')


if __name__ == '__main__':
    unittest.main()
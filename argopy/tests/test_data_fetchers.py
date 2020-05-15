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
# At this point, we are testing real data fetching both through facade and through direct call to backends

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
from argopy.errors import InvalidFetcherAccessPoint, InvalidFetcher, ErddapServerError

from argopy.utilities import list_available_data_src, isconnected, erddap_ds_exists
AVAILABLE_SOURCES = list_available_data_src()
CONNECTED = isconnected()
if CONNECTED:
    DSEXISTS = erddap_ds_exists(ds="ArgoFloats")
    DSEXISTS_bgc = erddap_ds_exists(ds="ArgoFloats-bio")
    DSEXISTS_ref = erddap_ds_exists(ds="ArgoFloats-ref")
else:
    DSEXISTS = False
    DSEXISTS_bgc = False
    DSEXISTS_ref = False

# List tests:
def test_invalid_accesspoint():
    with pytest.raises(InvalidFetcherAccessPoint):
        ArgoDataFetcher().invalid_accesspoint.to_xarray()

def test_invalid_fetcher():
    with pytest.raises(InvalidFetcher):
        ArgoDataFetcher().to_xarray() # Can't get data if access point not defined first

@unittest.skipUnless('localftp' in AVAILABLE_SOURCES, "requires localftp data fetcher")
def test_unavailable_accesspoint():
    with pytest.raises(InvalidFetcherAccessPoint):
        ArgoDataFetcher(src='localftp').region([-85,-45,10.,20.,0,100.]).to_xarray()

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
            try:
                ds = ArgoDataFetcher(src=bk, **options).float(arg).to_xarray()
                assert isinstance(ds, xr.Dataset) == True
            except ErddapServerError: # Test is passed when something goes wrong because of the erddap server, not our fault !
                pass

    def __test_profile(self, bk):
        """ Test float for a given backend """
        for arg in self.args['profile']:
            try:
                ds = ArgoDataFetcher(src=bk).profile(*arg).to_xarray()
                assert isinstance(ds, xr.Dataset) == True
            except ErddapServerError: # Test is passed when something goes wrong because of the erddap server, not our fault !
                pass

    def __test_region(self, bk):
        """ Test float for a given backend """
        for arg in self.args['region']:
            try:
                ds = ArgoDataFetcher(src=bk).region(arg).to_xarray()
                assert isinstance(ds, xr.Dataset) == True
            except ErddapServerError: # Test is passed when something goes wrong because of the erddap server, not our fault !
                pass

    @unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
    @unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
    def test_float_erddap(self):
        self.__test_float('erddap')

    @unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
    @unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
    def test_profile_erddap(self):
        self.__test_profile('erddap')

    @unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
    @unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
    def test_region_erddap(self):
        self.__test_region('erddap')

    @unittest.skipUnless('localftp' in AVAILABLE_SOURCES, "requires localftp data fetcher")
    def test_float_localftp(self):
        ftproot, flist = argopy.tutorial.open_dataset('localftp')
        with argopy.set_options(local_ftp=os.path.join(ftproot,'dac')):
            self.__test_float('localftp', )

    @unittest.skipUnless('localftp' in AVAILABLE_SOURCES, "requires localftp data fetcher")
    def test_profile_localftp(self):
        ftproot, flist = argopy.tutorial.open_dataset('localftp')
        with argopy.set_options(local_ftp=os.path.join(ftproot,'dac')):
            self.__test_profile('localftp')

    @unittest.skipUnless('argovis' in AVAILABLE_SOURCES, "requires argovis data fetcher")
    @unittest.skipUnless(CONNECTED, "argovis requires an internet connection")
    def test_float_argovis(self):
        self.__test_float('argovis')

@unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
@unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
class Erddap_backend(TestCase):
    """ Test main API facade for all available dataset of the ERDDAP fetching backend """

    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
    def test_cachepath(self):
        assert isinstance(ArgoDataFetcher(src='erddap').profile(6902746, 34).fetcher.cachepath, str) == True

    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
    def test_caching_float(self):
        cachedir = os.path.expanduser(os.path.join("~",".argopytest_tmp"))
        try:
            # 1st call to load from erddap and save to cachedir:
            ds = ArgoDataFetcher(src='erddap', cache=True, cachedir=cachedir).float([1901393, 6902746]).to_xarray()
            # 2nd call to load from cached file
            ds = ArgoDataFetcher(src='erddap', cache=True, cachedir=cachedir).float([1901393, 6902746]).to_xarray()
            assert isinstance(ds, xr.Dataset) == True
            shutil.rmtree(cachedir)
        except ErddapServerError: # Test is passed when something goes wrong because of the erddap server, not our fault !
            shutil.rmtree(cachedir)
            pass
        except:
            shutil.rmtree(cachedir)
            raise

    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
    def test_caching_profile(self):
        cachedir = os.path.expanduser(os.path.join("~",".argopytest_tmp"))
        try:
            # 1st call to load from erddap and save to cachedir:
            ds = ArgoDataFetcher(src='erddap', cache=True, cachedir=cachedir).profile(6902746, 34).to_xarray()
            # 2nd call to load from cached file
            ds = ArgoDataFetcher(src='erddap', cache=True, cachedir=cachedir).profile(6902746, 34).to_xarray()
            assert isinstance(ds, xr.Dataset) == True
            shutil.rmtree(cachedir)
        except ErddapServerError: # Test is passed when something goes wrong because of the erddap server, not our fault !
            shutil.rmtree(cachedir)
            pass
        except:
            shutil.rmtree(cachedir)
            raise

    def test_N_POINTS(self):
        n = ArgoDataFetcher(src='erddap').region([-70, -65, 35., 40., 0, 10., '2012-01', '2013-12']).fetcher.N_POINTS
        assert isinstance(n, int) == True

    def __testthis(self, dataset):
        for access_point in self.args:

            if access_point == 'profile':
                for arg in self.args['profile']:
                    try:
                        ds = ArgoDataFetcher(src='erddap', ds=dataset).profile(*arg).to_xarray()
                        assert isinstance(ds, xr.Dataset) == True
                    except ErddapServerError: # Test is passed when something goes wrong because of the erddap server, not our fault !
                        pass
                    except:
                        print("ERDDAP request:\n",
                              ArgoDataFetcher(src='erddap', ds=dataset).profile(*arg).fetcher.url)
                        pass

            if access_point == 'float':
                for arg in self.args['float']:
                    try:
                        ds = ArgoDataFetcher(src='erddap', ds=dataset).float(arg).to_xarray()
                        assert isinstance(ds, xr.Dataset) == True
                    except ErddapServerError: # Test is passed when something goes wrong because of the erddap server, not our fault !
                        pass
                    except:
                        print("ERDDAP request:\n",
                              ArgoDataFetcher(src='erddap', ds=dataset).float(arg).fetcher.url)
                        pass

            if access_point == 'region':
                for arg in self.args['region']:
                    try:
                        ds = ArgoDataFetcher(src='erddap', ds=dataset).region(arg).to_xarray()
                        assert isinstance(ds, xr.Dataset) == True
                    except ErddapServerError: # Test is passed when something goes wrong because of the erddap server, not our fault !
                        pass
                    except:
                        print("ERDDAP request:\n",
                              ArgoDataFetcher(src='erddap', ds=dataset).region(arg).fetcher.url)
                        pass

    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
    def test_phy_float(self):
        self.args = {}
        self.args['float'] = [[1901393],
                              [1901393, 6902746]]
        self.__testthis('phy')

    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
    def test_phy_profile(self):
        self.args = {}
        self.args['profile'] = [[6902746, 34],
                                [6902746, np.arange(12, 13)], [6902746, [1, 12]]]
        self.__testthis('phy')

    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
    def test_phy_region(self):
        self.args = {}
        self.args['region'] = [[-70, -65, 35., 40., 0, 10.],
                               [-70, -65, 35., 40., 0, 10., '2012-01', '2013-12']]
        self.__testthis('phy')

    @unittest.skipUnless(DSEXISTS_bgc, "erddap requires a valid BGC Argo dataset from Ifremer server")
    def test_bgc_float(self):
        self.args = {}
        self.args['float'] = [[5903248],
                              [7900596, 2902264]]
        self.__testthis('bgc')

    @unittest.skipUnless(DSEXISTS_bgc, "erddap requires a valid BGC Argo dataset from Ifremer server")
    def test_bgc_profile(self):
        self.args = {}
        self.args['profile'] = [[5903248, 34],
                                [5903248, np.arange(12, 14)], [5903248, [1, 12]]]
        self.__testthis('bgc')

    @unittest.skipUnless(DSEXISTS_bgc, "erddap requires a valid BGC Argo dataset from Ifremer server")
    def test_bgc_region(self):
        self.args = {}
        self.args['region'] = [[-70, -65, 35., 40., 0, 10.],
                               [-70, -65, 35., 40., 0, 10., '2012-01-1', '2012-12-31']]
        self.__testthis('bgc')

    @unittest.skipUnless(DSEXISTS_ref, "erddap requires a valid Reference Argo dataset from Ifremer server")
    def test_ref_region(self):
        self.args = {}
        self.args['region'] = [[-70, -65, 35., 40., 0, 10.],
                               [-70, -65, 35., 40., 0, 10., '2012-01-01', '2012-12-31']]
        self.__testthis('ref')

@unittest.skipUnless('localftp' in AVAILABLE_SOURCES, "requires localftp data fetcher")
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
                            ds = ArgoDataFetcher(src='localftp', ds=dataset).profile(*arg).to_xarray()
                            assert isinstance(ds, xr.Dataset) == True
                        except:
                            print("LOCALFTP request:\n",
                                  ArgoDataFetcher(src='localftp', ds=dataset).profile(*arg).fetcher.files)
                            pass

            if access_point == 'float':
                for arg in self.args['float']:
                    with argopy.set_options(local_ftp=self.local_ftp):
                        try:
                            ds = ArgoDataFetcher(src='localftp', ds=dataset).float(arg).to_xarray()
                            assert isinstance(ds, xr.Dataset) == True
                        except:
                            print("LOCALFTP request:\n",
                                  ArgoDataFetcher(src='localftp', ds=dataset).float(arg).fetcher.files)
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
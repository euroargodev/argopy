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
import numpy as np
import xarray as xr
import shutil

import pytest
import unittest
from unittest import TestCase

import argopy
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import InvalidFetcherAccessPoint, InvalidFetcher, ErddapServerError, \
    CacheFileNotFound, FileSystemHasNoCache, FtpPathError

from argopy.utilities import list_available_data_src, isconnected, isAPIconnected, erddap_ds_exists
AVAILABLE_SOURCES = list_available_data_src()
CONNECTED = isconnected()
CONNECTEDAPI = {src: False for src in AVAILABLE_SOURCES.keys()}
if CONNECTED:
    DSEXISTS = erddap_ds_exists(ds="ArgoFloats")
    DSEXISTS_bgc = erddap_ds_exists(ds="ArgoFloats-bio")
    DSEXISTS_ref = erddap_ds_exists(ds="ArgoFloats-ref")

    for src in AVAILABLE_SOURCES.keys():
        try:
            CONNECTEDAPI[src] = isAPIconnected(src=src, data=True)
        except InvalidFetcher:
            pass

else:
    DSEXISTS = False
    DSEXISTS_bgc = False
    DSEXISTS_ref = False

# List tests:
def test_invalid_accesspoint():
    src = list(AVAILABLE_SOURCES.keys())[0]  # Use the first valid data source
    with pytest.raises(InvalidFetcherAccessPoint):
        ArgoDataFetcher(src=src).invalid_accesspoint.to_xarray()  # Can't get data if access point not defined first
    with pytest.raises(InvalidFetcherAccessPoint):
        ArgoDataFetcher(src=src).to_xarray()  # Can't get data if access point not defined first


def test_invalid_fetcher():
    with pytest.raises(InvalidFetcher):
        ArgoDataFetcher(src='invalid_fetcher').to_xarray()


# @unittest.skipUnless('localftp' in AVAILABLE_SOURCES, "requires localftp data fetcher")
# def test_unavailable_accesspoint():
#     with pytest.raises(InvalidFetcherAccessPoint):
#         ArgoDataFetcher(src='localftp').region([-85., -45., 10., 20., 0., 100.]).to_xarray()


class EntryPoints_AllBackends(TestCase):
    """ Test main API facade for all available fetching backends and default dataset """
    ftproot = argopy.tutorial.open_dataset('localftp')[0]

    # kwargs_wmo = [{'WMO': 6901929},
    #               {'WMO': [6901929, 2901623]},
    #               {'CYC': 1},
    #               {'CYC': [1, 6]},
    #               {'WMO': 6901929, 'CYC': 36},
    #               {'WMO': 6901929, 'CYC': [5, 45]},
    #               {'WMO': [6901929, 2901623], 'CYC': 2},
    #               {'WMO': [6901929, 2901623], 'CYC': [2, 23]},
    #               {}]
    #
    # kwargs_box = [{'BOX': [-60, -40, 40., 60.]},
    #               {'BOX': [-60, -40, 40., 60., '2007-08-01', '2007-09-01']}]


    def setUp(self):
        # todo Determine the list of output format to test
        # what else beyond .to_xarray() ?

        self.fetcher_opts = {}

        # Define API entry point options to tests:
        self.args = {}
        self.args['float'] = [[2901623],
                              [2901623, 6901929]]
        self.args['profile'] = [[2901623, 12],
                                [2901623, np.arange(12, 14)], [6901929, [1, 6]]]
        self.args['region'] = [[-60, -55, 40., 45., 0., 10.],
                               [-60, -55, 40., 45., 0., 10., '2007-08-01', '2007-09-01']]

    def __test_float(self, bk, **ftc_opts):
        """ Test float for a given backend """
        for arg in self.args['float']:
            options = {**self.fetcher_opts, **ftc_opts}
            try:
                ds = ArgoDataFetcher(src=bk, **options).float(arg).to_xarray()
                assert isinstance(ds, xr.Dataset)
            except ErddapServerError:  # Test is passed when something goes wrong because of the erddap server, not our fault !
                pass

    def __test_profile(self, bk):
        """ Test float for a given backend """
        for arg in self.args['profile']:
            try:
                ds = ArgoDataFetcher(src=bk).profile(*arg).to_xarray()
                assert isinstance(ds, xr.Dataset)
            except ErddapServerError:  # Test is passed when something goes wrong because of the erddap server, not our fault !
                pass

    def __test_region(self, bk):
        """ Test float for a given backend """
        for arg in self.args['region']:
            try:
                ds = ArgoDataFetcher(src=bk).region(arg).to_xarray()
                assert isinstance(ds, xr.Dataset)
            except ErddapServerError:  # Test is passed when something goes wrong because of the erddap server, not our fault !
                pass

    @unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
    @unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
    @unittest.skipUnless(CONNECTEDAPI['erddap'], "erddap API is not alive")
    def test_float_erddap(self):
        self.__test_float('erddap')

    @unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
    @unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
    @unittest.skipUnless(CONNECTEDAPI['erddap'], "erddap API is not alive")
    def test_profile_erddap(self):
        self.__test_profile('erddap')

    @unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
    @unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
    @unittest.skipUnless(CONNECTEDAPI['erddap'], "erddap API is not alive")
    def test_region_erddap(self):
        self.__test_region('erddap')

    @unittest.skipUnless('localftp' in AVAILABLE_SOURCES, "requires localftp data fetcher")
    def test_float_localftp(self):
        with argopy.set_options(local_ftp=self.ftproot):
            self.__test_float('localftp')

    @unittest.skipUnless('localftp' in AVAILABLE_SOURCES, "requires localftp data fetcher")
    def test_profile_localftp(self):
        with argopy.set_options(local_ftp=self.ftproot):
            self.__test_profile('localftp')

    @unittest.skipUnless('localftp' in AVAILABLE_SOURCES, "requires localftp data fetcher")
    def test_region_localftp(self):
        with argopy.set_options(local_ftp=self.ftproot):
            self.__test_region('localftp')

    @unittest.skipUnless('argovis' in AVAILABLE_SOURCES, "requires argovis data fetcher")
    @unittest.skipUnless(CONNECTEDAPI['argovis'], "argovis API is not alive")
    def test_float_argovis(self):
        self.__test_float('argovis')

    @unittest.skipUnless('argovis' in AVAILABLE_SOURCES, "requires argovis data fetcher")
    @unittest.skipUnless(CONNECTEDAPI['argovis'], "argovis API is not alive")
    def test_profile_argovis(self):
        self.__test_profile('argovis')

    @unittest.skipUnless('argovis' in AVAILABLE_SOURCES, "requires argovis data fetcher")
    @unittest.skipUnless(CONNECTEDAPI['argovis'], "argovis API is not alive")
    def test_region_argovis(self):
        self.__test_region('argovis')



@unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
@unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
@unittest.skipUnless(CONNECTEDAPI['erddap'], "erddap API is not alive")
class Erddap(TestCase):
    """ Test main API facade for all available dataset and access points of the ERDDAP fetching backend """
    src = 'erddap'
    testcachedir = os.path.expanduser(os.path.join("~", ".argopytest_tmp"))

    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
    def test_cachepath_notfound(self):
        with argopy.set_options(cachedir=self.testcachedir):
            loader = ArgoDataFetcher(src=self.src, cache=True).profile(6902746, 34)
            with pytest.raises(CacheFileNotFound):
                loader.fetcher.cachepath
        shutil.rmtree(self.testcachedir)  # Make sure the cache is left empty

    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
    def test_nocache(self):
        with argopy.set_options(cachedir="dummy"):
            loader = ArgoDataFetcher(src=self.src, cache=False).profile(6902746, 34)
            loader.to_xarray()
            with pytest.raises(FileSystemHasNoCache):
                loader.fetcher.cachepath

    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
    def test_caching_float(self):
        with argopy.set_options(cachedir=self.testcachedir):
            try:
                loader = ArgoDataFetcher(src=self.src, cache=True).float([1901393, 6902746])
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

    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
    def test_caching_profile(self):
        with argopy.set_options(cachedir=self.testcachedir):
            loader = ArgoDataFetcher(src=self.src, cache=True).profile(6902746, 34)
            try:
                # 1st call to load from erddap and save to cachedir:
                ds = loader.to_xarray()
                # 2nd call to load from cached file
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

    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
    def test_N_POINTS(self):
        n = ArgoDataFetcher(src=self.src).region([-70, -65, 35., 40., 0, 10., '2012-01', '2013-12']).fetcher.N_POINTS
        assert isinstance(n, int)

    def __testthis_profile(self, dataset):
        for arg in self.args['profile']:
            try:
                ds = ArgoDataFetcher(src=self.src, ds=dataset).profile(*arg).to_xarray()
                assert isinstance(ds, xr.Dataset)
            except ErddapServerError:
                # Test is passed when something goes wrong because of the erddap server, not our fault !
                pass
            except Exception:
                print("ERDDAP request:\n",
                      ArgoDataFetcher(src=self.src, ds=dataset).profile(*arg).fetcher.url)
                pass

    def __testthis_float(self, dataset):
        for arg in self.args['float']:
            try:
                ds = ArgoDataFetcher(src=self.src, ds=dataset).float(arg).to_xarray()
                assert isinstance(ds, xr.Dataset)
            except ErddapServerError:
                # Test is passed when something goes wrong because of the erddap server, not our fault !
                pass
            except Exception:
                print("ERDDAP request:\n",
                      ArgoDataFetcher(src=self.src, ds=dataset).float(arg).fetcher.url)
                pass

    def __testthis_region(self, dataset):
        for arg in self.args['region']:
            try:
                ds = ArgoDataFetcher(src=self.src, ds=dataset).region(arg).to_xarray()
                assert isinstance(ds, xr.Dataset)
            except ErddapServerError:
                # Test is passed when something goes wrong because of the erddap server, not our fault !
                pass
            except Exception:
                print("ERDDAP request:\n",
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
class LocalFTP(TestCase):
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

    def test_invalidFTPpath(self):
        with pytest.raises(ValueError):
            with argopy.set_options(local_ftp="dummy"):
                ArgoDataFetcher(src=self.src).profile(2901623, 12)

        with pytest.raises(FtpPathError):
            with argopy.set_options(local_ftp = os.path.sep.join([self.local_ftp, "dac"]) ):
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
                          ArgoDataFetcher(src=self.src, ds=dataset).profile(*arg).fetcher.files)
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
                          ArgoDataFetcher(src=self.src, ds=dataset).float(arg).fetcher.files)
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
                          ArgoDataFetcher(src=self.src, ds=dataset).region(arg).fetcher.files)
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

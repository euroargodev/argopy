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

import numpy as np
import xarray as xr

import pytest
import unittest
from unittest import TestCase

import argopy
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import InvalidFetcherAccessPoint, InvalidFetcher, ErddapServerError
from argopy.utilities import list_available_data_src, isconnected, isAPIconnected, erddap_ds_exists


argopy.set_options(api_timeout=3*60)  # From Github actions, requests can take a while
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


class AllBackends(TestCase):
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


if __name__ == '__main__':
    unittest.main()

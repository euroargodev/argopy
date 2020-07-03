#!/bin/env python
# -*coding: UTF-8 -*-
#
# Test data fetchers
#

import xarray as xr

import pytest
import unittest
from unittest import TestCase

import argopy
from argopy import IndexFetcher as ArgoIndexFetcher
from argopy.errors import InvalidFetcherAccessPoint, InvalidFetcher
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


def test_invalid_accesspoint():
    src = list(AVAILABLE_INDEX_SOURCES.keys())[0]  # Use the first valid data source
    with pytest.raises(InvalidFetcherAccessPoint):
        ArgoIndexFetcher(src=src).invalid_accesspoint.to_xarray()  # Can't get data if access point not defined first
    with pytest.raises(InvalidFetcherAccessPoint):
        ArgoIndexFetcher(src=src).to_xarray()  # Can't get data if access point not defined first


def test_invalid_fetcher():
    with pytest.raises(InvalidFetcher):
        ArgoIndexFetcher(src='invalid_fetcher').to_xarray()


# @unittest.skipUnless('localftp' in AVAILABLE_SOURCES, "requires localftp data fetcher")
# def test_unavailable_accesspoint():
#     with pytest.raises(InvalidFetcherAccessPoint):
#         ArgoIndexFetcher((src=self.src).region([-85., -45., 10., 20., 0., 100.]).to_xarray()


class AllBackends(TestCase):
    """ Test main API facade for all available index fetching backends """

    def setUp(self):
        # todo Determine the list of output format to test
        # what else beyond .to_xarray() ?

        self.fetcher_opts = {}

        # Define API entry point options to tests:
        # These should be available online and with the argopy-data dummy gdac ftp
        self.args = {}
        self.args['float'] = [[2901623],
                              [6901929, 2901623]]
        self.args['region'] = [[-60, -40, 40., 60.],
                               [-60, -40, 40., 60., '2007-08-01', '2007-09-01']]
        self.args['profile'] = [[2901623, 2],
                                [6901929, [5, 45]]]

    def __test_float(self, bk, **ftc_opts):
        """ Test float index fetching for a given backend """
        for arg in self.args['float']:
            options = {**self.fetcher_opts, **ftc_opts}
            ds = ArgoIndexFetcher(src=bk, **options).float(arg).to_xarray()
            assert isinstance(ds, xr.Dataset)

    def __test_profile(self, bk, **ftc_opts):
        """ Test profile index fetching for a given backend """
        for arg in self.args['profile']:
            options = {**self.fetcher_opts, **ftc_opts}
            ds = ArgoIndexFetcher(src=bk, **options).profile(*arg).to_xarray()
            assert isinstance(ds, xr.Dataset)

    def __test_region(self, bk, **ftc_opts):
        """ Test float index fetching for a given backend """
        for arg in self.args['region']:
            options = {**self.fetcher_opts, **ftc_opts}
            ds = ArgoIndexFetcher(src=bk, **options).region(arg).to_xarray()
            assert isinstance(ds, xr.Dataset)

    @unittest.skipUnless('erddap' in AVAILABLE_INDEX_SOURCES, "requires erddap index fetcher")
    @unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core index Argo dataset from Ifremer server")
    @unittest.skipUnless(CONNECTEDAPI['erddap'], "erddap API is not alive")
    @unittest.skipUnless(False, "Waiting for https://github.com/euroargodev/argopy/issues/16")
    def test_float_index_erddap(self):
        self.__test_float('erddap')

    @unittest.skipUnless('erddap' in AVAILABLE_INDEX_SOURCES, "requires erddap index fetcher")
    @unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
    @unittest.skipUnless(DSEXISTS, "erddap requires a valid core index Argo dataset from Ifremer server")
    @unittest.skipUnless(CONNECTEDAPI['erddap'], "erddap API is not alive")
    @unittest.skipUnless(False, "Waiting for https://github.com/euroargodev/argopy/issues/16")
    def test_region_index_erddap(self):
        self.__test_region('erddap')

    @unittest.skipUnless('localftp' in AVAILABLE_INDEX_SOURCES, "requires localftp index fetcher")
    def test_float_index_localftp(self):
        ftproot, findex = argopy.tutorial.open_dataset('global_index_prof')
        with argopy.set_options(local_ftp=ftproot):
            self.__test_float('localftp', index_file='ar_index_global_prof.txt')

    @unittest.skipUnless('localftp' in AVAILABLE_INDEX_SOURCES, "requires localftp index fetcher")
    def test_profile_index_localftp(self):
        ftproot, findex = argopy.tutorial.open_dataset('global_index_prof')
        with argopy.set_options(local_ftp=ftproot):
            self.__test_profile('localftp', index_file='ar_index_global_prof.txt')

    @unittest.skipUnless('localftp' in AVAILABLE_INDEX_SOURCES, "requires localftp index fetcher")
    def test_region_index_localftp(self):
        ftproot, findex = argopy.tutorial.open_dataset('global_index_prof')
        with argopy.set_options(local_ftp=ftproot):
            self.__test_region('localftp', index_file='ar_index_global_prof.txt')


if __name__ == '__main__':
    unittest.main()

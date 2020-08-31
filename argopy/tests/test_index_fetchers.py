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
from argopy.errors import InvalidFetcherAccessPoint, InvalidFetcher, \
    FileSystemHasNoCache, CacheFileNotFound, ErddapServerError, DataNotFound

from argopy.utilities import list_available_index_src, isconnected, isAPIconnected, erddap_ds_exists
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


class EntryPoints_AllBackends(TestCase):
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


@unittest.skipUnless('erddap' in AVAILABLE_INDEX_SOURCES, "requires erddap index fetcher")
@unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
@unittest.skipUnless(DSEXISTS, "erddap requires a valid core index Argo dataset from Ifremer server")
# @unittest.skipUnless(False, "Waiting for https://github.com/euroargodev/argopy/issues/16")
@unittest.skipUnless(CONNECTEDAPI['erddap'], "erddap API is not alive")
class Erddap(TestCase):
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

@unittest.skipUnless('localftp' in AVAILABLE_INDEX_SOURCES, "requires localftp index fetcher")
class LocalFTP(TestCase):
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

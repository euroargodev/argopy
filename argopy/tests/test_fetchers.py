#!/bin/env python
# -*coding: UTF-8 -*-
#
# Test data data_fetchers
#
# Created by gmaze on 09/03/2020

import os
import sys
import numpy as np
import xarray as xr
import unittest
from unittest import TestCase
from argopy import DataFetcher as ArgoDataFetcher

# List available backends:
backends = list()
try:
    from erddapy import ERDDAP
    from argopy.data_fetchers import erddap as Erddap_Fetcher
    backends.append('erddap')
except ModuleNotFoundError:
    pass
except ImportError:
    pass

try:
    from argopy.data_fetchers import localftp as LocalFTP_Fetcher
    backends.append('localftp')
except ModuleNotFoundError:
    pass
except ImportError:
    pass

class EntryPoints(TestCase):
    """ Test main API facade for all available fetching backends and default dataset """

    def setUp(self):
        #todo Determine the list of output format to test
        # what else beyond .to_xarray() ?

        # Define API entry point options to tests:
        self.args = {}
        self.args['float'] = [[1901393],
                              [1901393, 6902746]]
        self.args['profile'] = [[6902746, 34],
                                [6902746, np.arange(12, 16)], [6902746, [1, 12]]]
        self.args['region'] = [[-75, -65, 30., 35., 0, 10.],
                               [-75, -65, 30., 35., 0, 10., '2012-01-01', '2012-06-30']]

    def __test_float(self, bk):
        """ Test float for a given backend """
        for arg in self.args['float']:
            print("\nTesting:", arg)
            ds = ArgoDataFetcher(backend=bk).float(arg).to_xarray()
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

    @unittest.skipUnless('erddap' in backends, "requires erddap data fetcher")
    def test_float_erddap(self):
        self.__test_float('erddap')

    @unittest.skipUnless('erddap' in backends, "requires erddap data fetcher")
    def test_profile_erddap(self):
        self.__test_profile('erddap')

    @unittest.skipUnless('erddap' in backends, "requires erddap data fetcher")
    def test_region_erddap(self):
        self.__test_region('erddap')

    @unittest.skipUnless('localftp' in backends, "requires localftp data fetcher")
    def test_float_localftp(self):
        self.__test_float('localftp')

    @unittest.skipUnless('argovis' in backends, "requires argovis data fetcher")
    def test_float_argovis(self):
        self.__test_float('argovis')

@unittest.skipUnless('erddap' in backends, "requires erddap data fetcher")
class Erddap_DataSets(TestCase):
    """ Test main API facade for all available dataset of Erddap fetching backend """

    def __testthis(self, dataset, to_test):
        for access_point in to_test:

            if access_point == 'profile':
                for arg in to_test['profile']:
                    print("\nTesting:", arg)
                    try:
                        ds = ArgoDataFetcher(backend='erddap', ds=dataset).profile(*arg).to_xarray()
                        assert isinstance(ds, xr.Dataset) == True
                    except:
                        print("ERDDAP request:\n",
                              ArgoDataFetcher(backend='erddap', ds=dataset).profile(*arg).fetcher.url)
                        pass

            if access_point == 'float':
                for arg in to_test['float']:
                    print("\nTesting:", arg)
                    try:
                        ds = ArgoDataFetcher(backend='erddap', ds=dataset).float(arg).to_xarray()
                        assert isinstance(ds, xr.Dataset) == True
                    except:
                        print("ERDDAP request:\n",
                              ArgoDataFetcher(backend='erddap', ds=dataset).float(arg).fetcher.url)
                        pass

            if access_point == 'region':
                for arg in to_test['region']:
                    print("\nTesting:", arg)
                    try:
                        ds = ArgoDataFetcher(backend='erddap', ds=dataset).region(arg).to_xarray()
                        assert isinstance(ds, xr.Dataset) == True
                    except:
                        print("ERDDAP request:\n",
                              ArgoDataFetcher(backend='erddap', ds=dataset).region(arg).fetcher.url)
                        pass

    def test_phy_float(self):
        to_test = {}
        to_test['float'] = [[1901393],
                              [1901393, 6902746]]
        self.__testthis('phy', to_test)

    def test_phy_profile(self):
        to_test = {}
        to_test['profile'] = [[6902746, 34],
                                [6902746, np.arange(12, 16)], [6902746, [1, 12]]]
        self.__testthis('phy', to_test)

    def test_phy_region(self):
        to_test = {}
        to_test['region'] = [[-75, -65, 30., 40., 0, 10.],
                               [-75, -65, 30., 40., 0, 10., '2012-01', '2013-12']]
        self.__testthis('phy', to_test)

    def test_bgc_float(self):
        to_test = {}
        to_test['float'] = [[5903248],
                              [7900596, 2902264]]
        self.__testthis('bgc', to_test)

    def test_bgc_profile(self):
        to_test = {}
        to_test['profile'] = [[5903248, 34],
                                [5903248, np.arange(12, 16)], [5903248, [1, 12]]]
        self.__testthis('bgc', to_test)

    def test_bgc_region(self):
        to_test = {}
        to_test['region'] = [[-75, -65, 30., 35., 0, 10.],
                               [-75, -65, 30., 35., 0, 10., '2012-01-1', '2012-06-30']]
        self.__testthis('bgc', to_test)

    def test_ref_region(self):
        to_test = {}
        to_test['region'] = [[-75, -65, 30., 35., 0, 10.],
                               [-75, -65, 30., 35., 0, 10., '2012-01-01', '2012-06-30']]
        self.__testthis('ref', to_test)

if __name__ == '__main__':
    unittest.main()
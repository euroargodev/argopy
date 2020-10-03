import numpy as np
import xarray as xr

import pytest
# import unittest

import argopy
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import InvalidFetcherAccessPoint, InvalidFetcher, ErddapServerError, ArgovisServerError
from argopy.utilities import list_available_data_src, isconnected, isAPIconnected, erddap_ds_exists, is_list_of_strings
from . import AVAILABLE_SOURCES, requires_fetcher, requires_connected_erddap_phy, requires_localftp, requires_connected_argovis

# AVAILABLE_SOURCES = list_available_data_src()
# CONNECTED = isconnected()
# CONNECTEDAPI = {src: False for src in AVAILABLE_SOURCES.keys()}
# if CONNECTED:
#     DSEXISTS = erddap_ds_exists(ds="ArgoFloats")
#     DSEXISTS_bgc = erddap_ds_exists(ds="ArgoFloats-bio")
#     DSEXISTS_ref = erddap_ds_exists(ds="ArgoFloats-ref")
#     for src in AVAILABLE_SOURCES.keys():
#         try:
#             CONNECTEDAPI[src] = isAPIconnected(src=src, data=True)
#         except InvalidFetcher:
#             pass
# else:
#     DSEXISTS = False
#     DSEXISTS_bgc = False
#     DSEXISTS_ref = False
#

@requires_fetcher
def test_invalid_accesspoint():
    src = list(AVAILABLE_SOURCES.keys())[0]  # Use the first valid data source
    with pytest.raises(InvalidFetcherAccessPoint):
        ArgoDataFetcher(src=src).invalid_accesspoint.to_xarray()  # Can't get data if access point not defined first
    with pytest.raises(InvalidFetcherAccessPoint):
        ArgoDataFetcher(src=src).to_xarray()  # Can't get data if access point not defined first


def test_invalid_fetcher():
    with pytest.raises(InvalidFetcher):
        ArgoDataFetcher(src='invalid_fetcher').to_xarray()


@requires_fetcher
class Test_AllBackends():
    """ Test main API facade for all available fetching backends and default dataset """
    local_ftp = argopy.tutorial.open_dataset('localftp')[0]

    # todo Determine the list of output format to test
    # what else beyond .to_xarray() ?

    fetcher_opts = {}

    mode = ['standard', 'expert']

    # Define API entry point options to tests:
    args = {}
    args['float'] = [[2901623],
                      [2901623, 6901929]]
    args['profile'] = [[2901623, 12],
                        [2901623, np.arange(12, 14)], [6901929, [1, 6]]]
    args['region'] = [[-60, -55, 40., 45., 0., 10.],
                       [-60, -55, 40., 45., 0., 10., '2007-08-01', '2007-09-01']]

    def __test_float(self, bk, **ftc_opts):
        """ Test float for a given backend """
        for arg in self.args['float']:
            for mode in self.mode:
                options = {**self.fetcher_opts, **ftc_opts}
                try:
                    f = ArgoDataFetcher(src=bk, mode=mode, **options).float(arg)
                    assert isinstance(f.to_xarray(), xr.Dataset)
                    assert is_list_of_strings(f.uri)
                except ErddapServerError:
                    # Test is passed even if something goes wrong with the erddap server
                    pass
                except ArgovisServerError:
                    # Test is passed even if something goes wrong with the argovis server
                    pass
                except Exception:
                    raise

    def __test_profile(self, bk):
        """ Test float for a given backend """
        for arg in self.args['profile']:
            for mode in self.mode:
                try:
                    f = ArgoDataFetcher(src=bk, mode=mode).profile(*arg)
                    assert isinstance(f.to_xarray(), xr.Dataset)
                    assert is_list_of_strings(f.uri)
                except ErddapServerError:
                    # Test is passed even if something goes wrong with the erddap server
                    pass
                except ArgovisServerError:
                    # Test is passed even if something goes wrong with the argovis server
                    pass
                except Exception:
                    raise

    def __test_region(self, bk):
        """ Test float for a given backend """
        for arg in self.args['region']:
            for mode in self.mode:
                try:
                    f = ArgoDataFetcher(src=bk, mode=mode).region(arg)
                    assert isinstance(f.to_xarray(), xr.Dataset)
                    assert is_list_of_strings(f.uri)
                except ErddapServerError:
                    # Test is passed even if something goes wrong with the erddap server
                    pass
                except ArgovisServerError:
                    # Test is passed even if something goes wrong with the argovis server
                    pass
                except Exception:
                    raise

    @requires_connected_erddap_phy
    def test_float_erddap(self):
        self.__test_float('erddap')

    @requires_localftp
    def test_float_localftp(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            self.__test_float('localftp')

    @requires_connected_argovis
    def test_float_argovis(self):
        self.__test_float('argovis')

    @requires_connected_erddap_phy
    def test_profile_erddap(self):
        self.__test_profile('erddap')

    @requires_localftp
    def test_profile_localftp(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            self.__test_profile('localftp')

    @requires_connected_argovis
    def test_profile_argovis(self):
        self.__test_profile('argovis')

    @requires_connected_erddap_phy
    def test_region_erddap(self):
        self.__test_region('erddap')

    @requires_localftp
    def test_region_localftp(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            self.__test_region('localftp')

    @requires_connected_argovis
    def test_region_argovis(self):
        self.__test_region('argovis')

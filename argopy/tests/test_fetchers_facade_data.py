import numpy as np
import xarray as xr

import pytest
import warnings

import argopy
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import (
    InvalidFetcherAccessPoint,
    InvalidFetcher,
    ErddapServerError,
    ArgovisServerError,
    DataNotFound
)
from aiohttp.client_exceptions import ServerDisconnectedError
from argopy.utilities import is_list_of_strings
from . import (
    AVAILABLE_SOURCES,
    requires_fetcher,
    requires_connected_erddap_phy,
    requires_localftp,
    requires_connected_argovis,
)


def safe_to_server_errors(test_func):
    """ Test fixture to make sure we don't fail because of an error from the server, not our Fault ! """

    def test_wrapper(fix):
        try:
            test_func(fix)
        except ErddapServerError as e:
            # Test is passed when something goes wrong because of the erddap server
            warnings.warn("\nSomething happened on erddap that should not: %s" % str(e.args))
            pass
        except ArgovisServerError as e:
            # Test is passed when something goes wrong because of the argovis server
            warnings.warn("\nSomething happened on argovis that should not: %s" % str(e.args))
            pass
        except DataNotFound as e:
            # We make sure that data requested by tests are available from API, so this must be a server side error.
            warnings.warn("\nSomething happened on server: %s" % str(e.args))
            pass
        except ServerDisconnectedError as e:
            # We make sure that data requested by tests are available from API, so this must be a server side error.
            warnings.warn("\nWe were disconnected from server !\n%s" % str(e.args))
            pass
        except Exception:
            raise

    return test_wrapper



def test_invalid_fetcher():
    with pytest.raises(InvalidFetcher):
        ArgoDataFetcher(src="invalid_fetcher").to_xarray()


@requires_fetcher
def test_invalid_accesspoint():
    src = list(AVAILABLE_SOURCES.keys())[0]  # Use the first valid data source
    with pytest.raises(InvalidFetcherAccessPoint):
        ArgoDataFetcher(
            src=src
        ).invalid_accesspoint.to_xarray()  # Can't get data if access point not defined first
    with pytest.raises(InvalidFetcherAccessPoint):
        ArgoDataFetcher(
            src=src
        ).to_xarray()  # Can't get data if access point not defined first


@requires_fetcher
def test_invalid_dataset():
    src = list(AVAILABLE_SOURCES.keys())[0]  # Use the first valid data source
    with pytest.raises(ValueError):
        ArgoDataFetcher(src=src, ds='dummy_ds')


@requires_fetcher
class Test_AllBackends:
    """ Test main API facade for all available fetching backends and default dataset """

    local_ftp = argopy.tutorial.open_dataset("localftp")[0]

    # todo Determine the list of output format to test
    # what else beyond .to_xarray() ?

    fetcher_opts = {}

    mode = ["standard", "expert"]

    # Define API entry point options to tests:
    args = {}
    args["float"] = [[2901623], [2901623, 6901929]]
    args["profile"] = [[2901623, 12], [2901623, np.arange(12, 14)], [6901929, [1, 6]]]
    args["region"] = [
        [-60, -55, 40.0, 45.0, 0.0, 10.0],
        [-60, -55, 40.0, 45.0, 0.0, 10.0, "2007-08-01", "2007-09-01"],
    ]

    def __test_float(self, bk, **ftc_opts):
        """ Test float for a given backend """
        for arg in self.args["float"]:
            for mode in self.mode:
                options = {**self.fetcher_opts, **ftc_opts}
                f = ArgoDataFetcher(src=bk, mode=mode, **options).float(arg)
                assert isinstance(f.to_xarray(), xr.Dataset)
                assert is_list_of_strings(f.uri)

    def __test_profile(self, bk):
        """ Test float for a given backend """
        for arg in self.args["profile"]:
            for mode in self.mode:
                f = ArgoDataFetcher(src=bk, mode=mode).profile(*arg)
                assert isinstance(f.to_xarray(), xr.Dataset)
                assert is_list_of_strings(f.uri)

    def __test_region(self, bk):
        """ Test float for a given backend """
        for arg in self.args["region"]:
            for mode in self.mode:
                f = ArgoDataFetcher(src=bk, mode=mode).region(arg)
                assert isinstance(f.to_xarray(), xr.Dataset)
                assert is_list_of_strings(f.uri)

    @requires_connected_erddap_phy
    @safe_to_server_errors
    def test_float_erddap(self):
        self.__test_float("erddap")

    @requires_localftp
    @safe_to_server_errors
    def test_float_localftp(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            self.__test_float("localftp")

    @requires_connected_argovis
    @safe_to_server_errors
    def test_float_argovis(self):
        self.__test_float("argovis")

    @requires_connected_erddap_phy
    @safe_to_server_errors
    def test_profile_erddap(self):
        self.__test_profile("erddap")

    @requires_localftp
    @safe_to_server_errors
    def test_profile_localftp(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            self.__test_profile("localftp")

    @requires_connected_argovis
    @safe_to_server_errors
    def test_profile_argovis(self):
        self.__test_profile("argovis")

    @requires_connected_erddap_phy
    @safe_to_server_errors
    def test_region_erddap(self):
        self.__test_region("erddap")

    @requires_localftp
    @safe_to_server_errors
    def test_region_localftp(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            self.__test_region("localftp")

    @requires_connected_argovis
    @safe_to_server_errors
    def test_region_argovis(self):
        self.__test_region("argovis")

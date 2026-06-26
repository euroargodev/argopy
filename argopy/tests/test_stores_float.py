import pytest
import tempfile
from typing import Generator, Any

import xarray as xr
import pandas as pd
import shutil
import logging
from urllib.parse import urlparse
import random

import argopy
from argopy.errors import OptionValueError
from argopy.utils.checkers import is_list_of_strings, is_cyc, is_wmo

from argopy.stores.float.implementations.offline.float import (
    FloatStore as ArgoFloatOffline,
)
from argopy.stores.float.implementations.online.float import (
    FloatStore as ArgoFloatOnline,
)

from argopy.tests.helpers.mocked_http import mocked_httpserver, mocked_server_address
from argopy.tests.helpers.utils import patch_ftp, has_connection, has_s3

log = logging.getLogger("argopy.tests.floatstore")

skip_online = pytest.mark.skipif(0, reason="Skipped tests for online implementation")
skip_offline = pytest.mark.skipif(0, reason="Skipped tests for offline implementation")
skip_spec = pytest.mark.skipif(0, reason="Skipped tests for specification")

"""
List GDAC hosts to be tested. 
Since the ArgoFloat class is compatible with hosts from local, http, ftp or s3 protocols, we try to test them all.
We use 2 dictionaries to be able to test implementations separately.
"""
VALID_LOCAL_HOSTS = {
    "local": argopy.tutorial.open_dataset("gdac")[0],  # Use local files
}
VALID_REMOTE_HOSTS = {
    "http1": mocked_server_address,  # Use the mocked http server
    # 'http2': 'https://data-argo.ifremer.fr',
    "ftp": "MOCKFTP",  # keyword to use a fake/mocked ftp server (running on localhost)
}

if has_s3:
    VALID_REMOTE_HOSTS.update({"s3": "s3://argo-gdac-sandbox/pub"})

VALID_HOSTS = VALID_LOCAL_HOSTS.copy()
VALID_HOSTS.update(VALID_REMOTE_HOSTS)

"""
List WMO to be tested, one for each mission
"""
WMO_CORE = [13857]
WMO_BGC = [3902131]
VALID_WMO = WMO_CORE + WMO_BGC


def id_for_host(host):
    """Get a short name for GDAC host to populate scenarios IDs"""
    if host == "MOCKFTP":
        return "ftp_mocked"
    elif "localhost" in host or "127.0.0.1" in host:
        return "http_mocked"
    else:
        return (lambda x: "local" if x == "" else x)(urlparse(host).scheme)


@skip_offline
class Test_FloatStore_Offline:
    """
    Tests methods and attributes specific to the Offline implementation

    Note that by directly using the implementation, and not the facade, un-cached extensions are not available (eg: 'plot'). Which is ok because extensions should have their own tests suite.
    """

    floatstore = ArgoFloatOffline

    scenarios = [
        (wmo, h, cache)
        for wmo in VALID_WMO
        for h in VALID_LOCAL_HOSTS.keys()
        for cache in [False, True]
    ]
    scenarios_ids = [
        "wmo=%i, host='%s', %s"
        % (
            opts[0],
            id_for_host(VALID_LOCAL_HOSTS[opts[1]]),
            "cached" if opts[2] else "no cache",
        )
        for opts in scenarios
    ]

    #############
    # UTILITIES #
    #############

    def setup_class(self):
        """setup any state specific to the execution of the given class"""
        self.cachedir = tempfile.mkdtemp()

    def teardown_class(self):
        """Cleanup once we are finished."""

        def remove_test_dir():
            shutil.rmtree(self.cachedir)

        remove_test_dir()

    @pytest.fixture
    def af(self, request) -> Generator[ArgoFloatOffline, Any, None]:
        """Fixture to create a Float store instance for a given wmo and host"""
        log.debug("-" * 50)
        # log.debug(request)
        wmo = request.param[0]
        host = patch_ftp(VALID_LOCAL_HOSTS[request.param[1]])
        cache = request.param[2]

        xfail, reason = False, ""

        store_args = {"host": host}
        if cache:
            store_args = {
                **store_args,
                **{"cache": True, "cachedir": self.cachedir},
            }

        try:
            log.debug(f"Instantiating a FloatStore with: {store_args}")
            this_af = self.floatstore(wmo, **store_args)
        except Exception:
            if xfail:
                pytest.xfail(reason)
            else:
                raise

        yield this_af

    #########
    # TESTS #
    #########

    def test_remotehost(self, mocked_httpserver):
        with pytest.raises(OptionValueError):
            self.floatstore(VALID_WMO[0], host=mocked_server_address)

    @pytest.mark.parametrize("af", scenarios, indirect=True, ids=scenarios_ids)
    def test_load_metadata(self, af):
        af.load_metadata()
        assert hasattr(af, "metadata")
        assert isinstance(af.metadata, dict)

    @pytest.mark.parametrize("af", scenarios, indirect=True, ids=scenarios_ids)
    def test_load_dac(self, af):
        af.load_dac()
        assert hasattr(af, "dac")
        assert isinstance(af.dac, str)


@skip_online
class Test_FloatStore_Online:
    """
    Tests methods and attributes specific to the Online implementation

    Note that by directly using the implementation, and not the facade, un-cached extensions are not available (eg: 'plot'). Which is ok because extensions should have their own tests suite.
    """

    floatstore = ArgoFloatOnline

    scenarios = [
        (wmo, h, cache)
        for wmo in VALID_WMO
        for h in VALID_REMOTE_HOSTS.keys()
        for cache in [False, True]
    ]
    scenarios_ids = [
        "wmo=%i, host='%s', %s"
        % (
            opts[0],
            id_for_host(VALID_REMOTE_HOSTS[opts[1]]),
            "cached" if opts[2] else "no cache",
        )
        for opts in scenarios
    ]

    #############
    # UTILITIES #
    #############

    def setup_class(self):
        """setup any state specific to the execution of the given class"""
        self.cachedir = tempfile.mkdtemp()

    def teardown_class(self):
        """Cleanup once we are finished."""

        def remove_test_dir():
            shutil.rmtree(self.cachedir)

        remove_test_dir()

    def _patch_host(self, host):
        if "s3" in host and not has_connection:
            log.info("Skip this test with 's3' because there is no internet connection")
            pytest.skip(
                "Skip this test with 's3' because there is no internet connection"
            )
        return patch_ftp(host)

    @pytest.fixture
    def af(self, request) -> Generator[ArgoFloatOnline, Any, None]:
        """Fixture to create a Float store instance for a given wmo and host"""
        log.debug("-" * 50)
        # log.debug(request)
        wmo = request.param[0]
        host = self._patch_host(VALID_REMOTE_HOSTS[request.param[1]])
        cache = request.param[2]

        xfail, reason = False, ""
        if not has_s3 and "s3" in host:
            xfail, reason = True, "s3fs not available"

        store_args = {
            "host": host,
            "eafleetmonitoring_server": mocked_server_address,
            # also use mocked server for Euro-Argo meta data API calls
        }
        if cache:
            store_args = {
                **store_args,
                **{"cache": True, "cachedir": self.cachedir},
            }

        try:
            log.debug(f"Instantiating a FloatStore with: {store_args}")
            this_af = self.floatstore(wmo, **store_args)
        except Exception:
            if xfail:
                pytest.xfail(reason)
            else:
                raise

        yield this_af

    #########
    # TESTS #
    #########
    @pytest.mark.parametrize("af", scenarios, indirect=True, ids=scenarios_ids)
    def test_load_metadata(self, af):
        af.load_metadata()
        assert hasattr(af, "metadata")
        assert isinstance(af.metadata, dict)

    @pytest.mark.parametrize("af", scenarios, indirect=True, ids=scenarios_ids)
    def test_load_dac(self, af):
        af.load_dac()
        assert hasattr(af, "dac")
        assert isinstance(af.dac, str)

    @pytest.mark.parametrize("af", scenarios, indirect=True, ids=scenarios_ids)
    def test_load_technicaldata(self, af):
        af.load_technicaldata()
        assert hasattr(af, "technicaldata")
        assert isinstance(af.technicaldata, dict)

    @pytest.mark.parametrize("af", scenarios, indirect=True, ids=scenarios_ids)
    def test_api_point(self, af):
        assert isinstance(af.api_point, dict)
        assert "meta" in af.api_point
        assert "technical" in af.api_point


@skip_spec
class Test_FloatStore_Spec:
    """
    Tests for methods and attributes shared by all implementations, hence from the specification prototype.

    The instance fixture `af` will rely on the appropriate implementation for a given host.

    But note that by directly using the implementation, and not the facade, un-cached extensions are not available (eg: 'plot'). Which is ok because extensions should have their own tests suite.
    """

    scenarios = [
        (wmo, h, cache)
        for wmo in VALID_WMO
        for h in VALID_HOSTS.keys()
        for cache in [False, True]
    ]
    scenarios_ids = [
        "wmo=%i, host='%s', %s"
        % (
            opts[0],
            id_for_host(VALID_HOSTS[opts[1]]),
            "cached" if opts[2] else "no cache",
        )
        for opts in scenarios
    ]

    scenarios_bgc = [
        (wmo, h, cache)
        for wmo in WMO_BGC
        for h in VALID_HOSTS.keys()
        for cache in [False, True]
    ]
    scenarios_ids_bgc = [
        "wmo=%i, host='%s', %s"
        % (
            opts[0],
            id_for_host(VALID_HOSTS[opts[1]]),
            "cached" if opts[2] else "no cache",
        )
        for opts in scenarios_bgc
    ]

    scenarios_core = [
        (wmo, h, cache)
        for wmo in WMO_BGC
        for h in VALID_HOSTS.keys()
        for cache in [False, True]
    ]
    scenarios_ids_core = [
        "wmo=%i, host='%s', %s"
        % (
            opts[0],
            id_for_host(VALID_HOSTS[opts[1]]),
            "cached" if opts[2] else "no cache",
        )
        for opts in scenarios_core
    ]
    #############
    # UTILITIES #
    #############

    def setup_class(self):
        """setup any state specific to the execution of the given class"""
        self.cachedir = tempfile.mkdtemp()

    def teardown_class(self):
        """Cleanup once we are finished."""

        def remove_test_dir():
            shutil.rmtree(self.cachedir)

        remove_test_dir()

    def _patch_host(self, host):
        if "s3" in host and not has_connection:
            log.info("Skip this test with 's3' because there is no internet connection")
            pytest.skip(
                "Skip this test with 's3' because there is no internet connection"
            )
        return patch_ftp(host)

    @pytest.fixture
    def af(self, request) -> Generator[ArgoFloatOnline, Any, None]:
        """Fixture to create a Float store instance for a given wmo and host"""
        wmo = request.param[0]
        host = self._patch_host(VALID_HOSTS[request.param[1]])
        cache = request.param[2]

        xfail, reason = False, ""
        if not has_s3 and "s3" in host:
            xfail, reason = True, "s3fs not available"

        if host in VALID_LOCAL_HOSTS.values():
            floatstore = ArgoFloatOffline
            store_args = {"host": host}
        else:
            floatstore = ArgoFloatOnline
            store_args = {
                "host": host,
                "eafleetmonitoring_server": mocked_server_address,
                # also use mocked server for Euro-Argo meta data API calls
            }

        if cache:
            store_args = {
                **store_args,
                **{"cache": True, "cachedir": self.cachedir},
            }

        try:
            log.debug(f"Instantiating a FloatStore with: {store_args}")
            this_af = floatstore(wmo, **store_args)
        except Exception:
            if xfail:
                pytest.xfail(reason)
            else:
                raise

        yield this_af

    #########
    # TESTS #
    #########
    @pytest.mark.parametrize("af", scenarios, indirect=True, ids=scenarios_ids)
    def test_attributes(self, mocked_httpserver, af):
        assert is_wmo(af.WMO)

        assert hasattr(af, "dac")
        assert isinstance(af.dac, str)

        assert hasattr(af, "metadata")
        assert isinstance(af.metadata, dict)

        assert is_cyc(af.CYCLE_NUMBERS)
        assert isinstance(af.N_CYCLES, int)

        assert isinstance(af.path, str)
        assert isinstance(af.host_sep, str)
        assert isinstance(af.host_protocol, str)

    @pytest.mark.parametrize("af", scenarios, indirect=True, ids=scenarios_ids)
    def test_list_directories(self, mocked_httpserver, af):

        assert isinstance(af.ls_datasets(), dict)
        assert is_list_of_strings(af._ls())

        assert isinstance(af.ls_profiles(), dict)
        assert is_list_of_strings(af._lsp())

        assert isinstance(af.profiles_to_dataframe(), pd.DataFrame)

    @pytest.mark.parametrize("af", scenarios, indirect=True, ids=scenarios_ids)
    def test_open_dataset(self, mocked_httpserver, af):
        lds = af.ls_datasets()
        ds_key, _ = random.choice(list(lds.items()))
        assert isinstance(af[ds_key], xr.Dataset)
        assert isinstance(af.dataset(ds_key), xr.Dataset)
        assert isinstance(af.open_dataset(ds_key), xr.Dataset)

        with pytest.raises(ValueError):
            af.open_dataset("dummy_ds_key")

    @pytest.mark.parametrize("af", scenarios, indirect=True, ids=scenarios_ids)
    def test_open_profile(self, mocked_httpserver, af):
        lds = af.ls_profiles()
        ds_key, _ = random.choice(list(lds.items()))
        assert isinstance(af[ds_key], xr.Dataset)
        assert isinstance(af.profile(ds_key), xr.Dataset)
        assert isinstance(af.open_profile(ds_key), xr.Dataset)

        with pytest.raises(ValueError):
            af.open_profile("dummy_ds_key")

    @pytest.mark.parametrize("af", scenarios, indirect=True, ids=scenarios_ids)
    def test_open_profiles(self, mocked_httpserver, af):
        ds_list = af.open_profiles(af.CYCLE_NUMBERS[0:2])
        assert all([isinstance(ds, xr.Dataset) for ds in ds_list])

    # @pytest.mark.parametrize("af", scenarios, indirect=True, ids=scenarios_ids)
    # def test_get_item(self, mocked_httpserver, af):
    #     assert isinstance(af[1], xr.Dataset)
    #     assert all([isinstance(ds, xr.Dataset) for ds in af[1:3]])
    #     assert all([isinstance(ds, xr.Dataset) for ds in af[1:5:2]])
    #
    # @pytest.mark.parametrize("af", scenarios_bgc, indirect=True, ids=scenarios_ids_bgc)
    # def test_get_item_bgc(self, mocked_httpserver, af):
    #     for dataset in ['B', 'S']:
    #         assert isinstance(af[1, dataset], xr.Dataset)
    #         assert all([isinstance(ds, xr.Dataset) for ds in af[1:3, dataset]])
    #         assert all([isinstance(ds, xr.Dataset) for ds in af[1:5:2, dataset]])

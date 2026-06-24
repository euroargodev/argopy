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
Since the class is compatible with host from local, http, ftp or s3 protocols, we try to test them all:
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

"""
List WMO to be tested, one for each mission
"""
VALID_WMO = [13857, 3902131]  # core, bgc


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


VALID_HOSTS = VALID_LOCAL_HOSTS.copy()
VALID_HOSTS.update(VALID_REMOTE_HOSTS)


@skip_spec
class Test_FloatStore_Spec:
    """
    Tests specification methods and attributes of the facade, for both the offline/online implementations
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
    def test_instance(self, mocked_httpserver, af):
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

        assert isinstance(af.ls_datasets(), dict)
        assert is_list_of_strings(af.ls())

        assert isinstance(af.ls_profiles(), dict)
        assert is_list_of_strings(af.lsp())
        assert isinstance(af.describe_profiles(), pd.DataFrame)

    @pytest.mark.parametrize("af", scenarios, indirect=True, ids=scenarios_ids)
    def test_open_dataset(self, mocked_httpserver, af):
        lds = af.ls_datasets()
        dsname, _ = random.choice(list(lds.items()))
        assert isinstance(af.open_dataset(dsname), xr.Dataset)

        with pytest.raises(ValueError):
            af.open_dataset("dummy_dsname")

    @pytest.mark.parametrize("af", scenarios, indirect=True, ids=scenarios_ids)
    def test_open_profile(self, mocked_httpserver, af):
        lds = af.ls_profiles()
        dsname, _ = random.choice(list(lds.items()))
        # dsname = list(lds)[0]
        assert isinstance(af.open_profile(dsname), xr.Dataset)

        with pytest.raises(ValueError):
            af.open_dataset("dummy_dsname")

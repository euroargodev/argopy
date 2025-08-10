import pytest
import tempfile

import xarray as xr
import pandas as pd
import importlib
import shutil
import logging
from urllib.parse import urlparse
import random

import argopy
from argopy.utils.checkers import is_list_of_strings

from argopy.stores.float.implementations.offline.float import FloatStore as ArgoFloatOffline
from argopy.stores.float.implementations.online.float import FloatStore as ArgoFloatOnline

from mocked_http import mocked_httpserver, mocked_server_address
from utils import patch_ftp


log = logging.getLogger("argopy.tests.floatstore")

skip_online = pytest.mark.skipif(0, reason="Skipped tests for online implementation")
skip_offline = pytest.mark.skipif(0, reason="Skipped tests for offline implementation")


"""
List GDAC hosts to be tested. 
Since the class is compatible with host from local, http, ftp or s3 protocols, we try to test them all:
"""
VALID_HOSTS = {
    'local': argopy.tutorial.open_dataset("gdac")[0],  # Use local files
    'http1': mocked_server_address,  # Use the mocked http server
    # 'http2': 'https://data-argo.ifremer.fr',
    'ftp': "MOCKFTP",  # keyword to use a fake/mocked ftp server (running on localhost)
}

HAS_S3FS = importlib.util.find_spec("s3fs") is not None
if HAS_S3FS:
    VALID_HOSTS.update({'s3': 's3://argo-gdac-sandbox/pub'})

"""
List WMO to be tested, one for each mission
"""
VALID_WMO = [13857, 3902131]


@skip_offline
class Test_FloatStore_Offline():
    host = VALID_HOSTS['local']

    @pytest.mark.parametrize("wmo", VALID_WMO, indirect=False)
    def test_init(self, wmo):
        ArgoFloatOffline(wmo, host=self.host)

    def test_dac_notfound(self):
        with pytest.raises(ValueError):
            ArgoFloatOffline(123456, host=self.host)


def id_for_host(host):
    """Get a short name for scenarios IDs, given a FTP host"""
    if host == "MOCKFTP":
        return "ftp_mocked"
    elif "localhost" in host or "127.0.0.1" in host:
        return "http_mocked"
    else:
        return (lambda x: "local" if x == "" else x)(urlparse(host).scheme)


@skip_online
class Test_FloatStore_Online():
    floatstore = ArgoFloatOnline

    scenarios = [(wmo, h, cache) for wmo in VALID_WMO for h in VALID_HOSTS.keys() for cache in [True, False]]
    scenarios_ids = [
        "wmo=%i, host='%s', %s" % (opts[0], id_for_host(VALID_HOSTS[opts[1]]), 'cached' if opts[2] else 'no cache')
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

    def _patch_ftp(self, ftp):
        return patch_ftp(ftp)

    def call_floatstore(self, WMO, store_args, xfail=False, reason="?"):
        def core(WMO, fargs):
            try:
                log.debug(fargs)
                af = self.floatstore(WMO, **fargs)
            except Exception:
                if xfail:
                    pytest.xfail(reason)
                else:
                    raise
            return af

        return core(WMO, store_args)

    def get_a_floatstore(self, WMO, host, cache=False, **kwargs):
        store_args = {'host': host}
        if cache:
            store_args = {
                **store_args,
                **{"cache": True, "cachedir": self.cachedir},
            }
        log.debug(store_args)
        return self.call_floatstore(WMO, store_args, **kwargs)

    @pytest.fixture
    def store_maker(self, request):
        """Fixture to create a Float store instance for a given wmo and host"""
        log.debug("-"*50)
        log.debug(request)
        wmo = request.param[0]
        host = self._patch_ftp(VALID_HOSTS[request.param[1]])
        cache = request.param[2]

        xfail, reason = False, ""
        if not HAS_S3FS and 's3' in host:
            xfail, reason = True, 's3fs not available'
        # elif 's3' in host:
        #     xfail, reason = True, 's3 is experimental'

        yield self.get_a_floatstore(wmo, host=host, cache=cache, xfail=xfail, reason=reason)

    def assert_float(self, this_af):
        assert isinstance(this_af.load_index(), ArgoFloatOnline)

        assert hasattr(this_af, "dac")
        assert isinstance(this_af.dac, str)

        assert hasattr(this_af, "metadata")
        assert isinstance(this_af.metadata, dict)

        assert hasattr(this_af, "technicaldata")
        assert isinstance(this_af.technicaldata, dict)

        assert hasattr(this_af, "api_point")
        assert isinstance(this_af.api_point, dict)

        assert isinstance(this_af.N_CYCLES, int)
        assert isinstance(this_af.path, str)
        assert isinstance(this_af.host_sep, str)
        assert isinstance(this_af.host_protocol, str)

        assert isinstance(this_af.ls_dataset(), dict)
        assert is_list_of_strings(this_af.ls())

        assert is_list_of_strings(this_af.lsprofiles())
        assert isinstance(this_af.describe_profiles(), pd.DataFrame)

    def assert_open_dataset(self, this_af):
        lds = this_af.ls_dataset()
        dsname, _ = random.choice(list(lds.items()))
        assert isinstance(this_af.open_dataset(dsname), xr.Dataset)

        with pytest.raises(ValueError):
            this_af.open_dataset('dummy_dsname')

    #########
    # TESTS #
    #########
    @pytest.mark.parametrize(
        "store_maker", scenarios, indirect=True, ids=scenarios_ids
    )
    def test_wmo(self, mocked_httpserver, store_maker):
        self.assert_float(store_maker)

    @pytest.mark.parametrize(
        "store_maker", scenarios, indirect=True, ids=scenarios_ids
    )
    def test_open_dataset(self, mocked_httpserver, store_maker):
        self.assert_open_dataset(store_maker)

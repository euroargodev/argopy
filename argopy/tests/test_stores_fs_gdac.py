import importlib
from urllib.parse import urlparse
import pytest
import tempfile
import shutil
import xarray as xr
import logging

import argopy
from argopy.stores import gdacfs
from mocked_http import mocked_httpserver, mocked_server_address
from utils import patch_ftp


log = logging.getLogger("argopy.tests.gdacfs")


"""
List gdac hosts to be tested. 
Since the GDAC fs is compatible with host from local, http, ftp or s3 protocols, we try to test them all:
"""
VALID_HOSTS = {
    'local': argopy.tutorial.open_dataset("gdac")[0],  # Use local files
    'http': mocked_server_address,  # Use the mocked http server
    'ftp': "MOCKFTP",  # keyword to use a fake/mocked ftp server (running on localhost)
}

HAS_S3FS = importlib.util.find_spec("s3fs") is not None
if HAS_S3FS:
    VALID_HOSTS.update({'s3': 's3://argo-gdac-sandbox/pub'})

def id_for_host(host):
    """Get a short name for scenarios IDs, given a FTP host"""
    if host == "MOCKFTP":
        return "ftp_mocked"
    elif "localhost" in host or "127.0.0.1" in host:
        return "http_mocked"
    else:
        return (lambda x: "local" if x == "" else x)(urlparse(host).scheme)


class Test_Gdacfs:
    scenarios = [(h, cache) for h in VALID_HOSTS.keys() for cache in [False]]
    scenarios_ids = [
        "host='%s', %s" % (id_for_host(VALID_HOSTS[opts[0]]), 'cached' if opts[1] else 'no cache')
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
        log.debug(ftp)
        return patch_ftp(ftp)

    def call_gdacfs(self, host, xfail=False, reason="?"):
        def core(host):
            try:
                af = gdacfs(host)
            except Exception:
                if xfail:
                    pytest.xfail(reason)
                else:
                    raise
            return af

        return core(host)

    def get_a_gdacfs(self, host, **kwargs):
        return self.call_gdacfs(host, **kwargs)

    @pytest.fixture
    def store_maker(self, request):
        """Fixture to create a GDAC store instance for a given host"""
        host = self._patch_ftp(VALID_HOSTS[request.param[0]])
        log.debug(host)
        # cache = request.param[1]

        xfail, reason = False, ""
        if not HAS_S3FS and 's3' in host:
            xfail, reason = True, 's3fs not available'
        # elif 's3' in host:
        #     xfail, reason = True, 's3 is experimental'

        yield self.get_a_gdacfs(host=host, xfail=xfail, reason=reason)

    def assert_fs(self, fs):
        assert isinstance(fs.open_dataset("dac/aoml/13857/13857_meta.nc"), xr.Dataset)
        assert fs.info("dac/aoml/13857/13857_meta.nc")['size'] == 25352

    #########
    # TESTS #
    #########
    @pytest.mark.parametrize(
        "store_maker", scenarios, indirect=True, ids=scenarios_ids
    )
    def test_implementation(self, mocked_httpserver, store_maker):
        self.assert_fs(store_maker)

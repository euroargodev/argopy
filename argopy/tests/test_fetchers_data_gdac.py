"""
Test the GDAC data fetcher backend

Here we try an approach based on fixtures and pytest parametrization
to make more explicit the full list of scenario tested.
"""
import sys

import xarray as xr

import pytest
import shutil
from urllib.parse import urlparse
import logging
import importlib

import argopy
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import (
    CacheFileNotFound,
)
from argopy.utils.checkers import is_list_of_strings, check_gdac_path
from utils import requires_gdac, create_temp_folder, patch_ftp
from mocked_http import mocked_httpserver
from mocked_http import mocked_server_address as MOCKHTTP


log = logging.getLogger("argopy.tests.data.gdac")


"""
List GDAC hosts to be tested. 
Since the fetcher is compatible with host from local, http or ftp protocols, we try to test them all:
"""
HOSTS = [
         argopy.tutorial.open_dataset("gdac")[0],
         MOCKHTTP,
         'MOCKFTP',
        ]

HAS_S3FS = importlib.util.find_spec("s3fs") is not None
if HAS_S3FS:
    # todo Create a mocked server for s3 tests
    HOSTS.append("s3://argo-gdac-sandbox/pub")  # todo: How do we mock a s3 server ?


"""
List access points to be tested.
For each access points, we list 1-to-2 scenario to make sure all possibilities are tested
"""
ACCESS_POINTS = [
    {"float": [13857]},
    {"profile": [13857, 90]},
    {"region": [-20, -16., 0, 1, 0, 100.]},
    {"region": [-20, -16., 0, 1, 0, 100., "1997-07-01", "1997-09-01"]},
    ]

"""
List parallel methods to be tested.
"""
valid_parallel_opts = [
    {"parallel": "thread"},
    # {"parallel": True, "parallel_method": "thread"},  # opts0
    # {"parallel": True, "parallel_method": "process"}  # opts1
]

"""
List user modes to be tested
"""
USER_MODES = ['standard', 'expert', 'research']


@requires_gdac
def create_fetcher(fetcher_args, access_point):
    """ Create a fetcher for a given set of facade options and access point

    """
    def core(fargs, apts):
        try:
            f = ArgoDataFetcher(**fargs)
            if "float" in apts:
                f = f.float(apts['float'])
            elif "profile" in apts:
                f = f.profile(*apts['profile'])
            elif "region" in apts:
                f = f.region(apts['region'])
        except Exception:
            raise
        return f
    fetcher = core(fetcher_args, access_point)
    return fetcher


def assert_fetcher(server, this_fetcher, cacheable=False):
    """Assert a data fetcher.

        This should be used by all tests
    """
    try:
        assert isinstance(this_fetcher.to_xarray(errors='raise'), xr.Dataset)
    except PermissionError:
        if sys.platform.startswith('win32') or sys.platform.startswith('cygwin'):
            pytest.xfail("Fails because of Windows permissions error that can't be fixed (e.g. https://github.com/python/cpython/issues/66305)")
        else:
            raise

    core = this_fetcher.fetcher
    assert is_list_of_strings(core.uri)
    assert (core.N_RECORDS >= 1)  # Make sure we loaded the index file content
    assert (core.N_FILES >= 1)  # Make sure we found results
    if cacheable:
        assert is_list_of_strings(core.cachepath)


def gdac_shortname(gdac):
    """Get a short name for scenarios IDs, given a FTP host"""
    if gdac == 'MOCKFTP':
        return 'ftp_mocked'
    elif 'localhost' in gdac or '127.0.0.1' in gdac:
        return 'http_mocked'
    else:
        return (lambda x: 'file' if x == "" else x)(urlparse(gdac).scheme)

"""
Make a list of VALID host/dataset/access_points to be tested
"""
VALID_ACCESS_POINTS, VALID_ACCESS_POINTS_IDS = [], []
for host in HOSTS:
    for mode in USER_MODES:
        for ap in ACCESS_POINTS:
            VALID_ACCESS_POINTS.append({'host': host, 'ds': 'phy', 'mode': mode, 'access_point': ap})
            VALID_ACCESS_POINTS_IDS.append("host='%s', ds='%s', mode='%s', %s" % (gdac_shortname(host), 'phy', mode, ap))



@requires_gdac
class TestBackend:
    src = 'gdac'

    #############
    # UTILITIES #
    #############

    def setup_class(self):
        """setup any state specific to the execution of the given class"""
        # Create the cache folder here, so that it's not the same for the pandas and pyarrow tests
        self.cachedir = create_temp_folder().folder

    def _patch_gdac(self, gdac):
        return patch_ftp(gdac)

    def _setup_fetcher(self, this_request, cached=False, parallel=False):
        """Helper method to set up options for a fetcher creation"""
        gdac = this_request.param['host']
        access_point = this_request.param['access_point']
        N_RECORDS = None if 'tutorial' in gdac or 'MOCK' in gdac else 100  # Make sure we're not going to load the full index

        fetcher_args = {"src": self.src,
                         "gdac": self._patch_gdac(gdac),
                         "ds": this_request.param['ds'],
                         "mode": this_request.param['mode'],
                         "cache": cached,
                         "cachedir": self.cachedir,
                         "parallel": False,
                         "N_RECORDS": N_RECORDS,
                         }

        if not cached:
            # cache is False by default, so we don't need to clutter the arguments list
            del fetcher_args["cache"]
            del fetcher_args["cachedir"]
        if not parallel:
            # parallel is False by default, so we don't need to clutter the arguments list
            del fetcher_args["parallel"]

        if not check_gdac_path(fetcher_args['gdac']):
            pytest.xfail("Fails because %s cannot be reached" % fetcher_args['gdac'])
        else:
            return fetcher_args, access_point

    @pytest.fixture
    def fetcher(self, request, mocked_httpserver):
        """ Fixture to create a GDAC fetcher for a given host and access point """
        fetcher_args, access_point = self._setup_fetcher(request, cached=False)
        yield create_fetcher(fetcher_args, access_point)

    @pytest.fixture
    def cached_fetcher(self, request):
        """ Fixture to create a cached FTP fetcher for a given host and access point """
        fetcher_args, access_point = self._setup_fetcher(request, cached=True)
        yield create_fetcher(fetcher_args, access_point)

    def teardown_class(self):
        """Cleanup once we are finished."""
        def remove_test_dir():
            shutil.rmtree(self.cachedir)
        remove_test_dir()

    #########
    # TESTS #
    #########
    @pytest.mark.parametrize("fetcher", VALID_ACCESS_POINTS, indirect=True, ids=VALID_ACCESS_POINTS_IDS)
    def test_fetching(self, mocked_httpserver, fetcher):
        assert_fetcher(mocked_httpserver, fetcher, cacheable=False)

    @pytest.mark.parametrize("cached_fetcher", VALID_ACCESS_POINTS, indirect=True, ids=VALID_ACCESS_POINTS_IDS)
    def test_fetching_cached(self, mocked_httpserver, cached_fetcher):
        # Assert the fetcher (this trigger data fetching, hence caching as well):
        assert_fetcher(mocked_httpserver, cached_fetcher, cacheable=True)
        # and we also make sure we can clear the cache:
        cached_fetcher.clear_cache()
        with pytest.raises(CacheFileNotFound):
            cached_fetcher.fetcher.cachepath

    def test_uri_mono2multi(self, mocked_httpserver):
        ap = [v for v in ACCESS_POINTS if 'region' in v.keys()][0]
        f = create_fetcher({"src": self.src, "gdac": HOSTS[0], "N_RECORDS": 100}, ap).fetcher
        assert is_list_of_strings(f.uri_mono2multi(f.uri))

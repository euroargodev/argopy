import pandas as pd

import pytest
import tempfile
import shutil
from fsspec.core import split_protocol

import argopy
from argopy import IndexFetcher as ArgoIndexFetcher
from argopy.errors import (
    CacheFileNotFound,
    FileSystemHasNoCache,
    FtpPathError
)
from argopy.utilities import isconnected
from argopy.options import check_gdac_path
from . import (
    requires_ftp,
    safe_to_server_errors
)

import logging


log = logging.getLogger("argopy.tests.index.ftp")
skip_for_debug = pytest.mark.skipif(True, reason="Taking too long !")


"""
List ftp hosts to be tested. 
Since the fetcher is compatible with host from local, http or ftp protocols, we
try to test them all:
"""
host_list = [argopy.tutorial.open_dataset("localftp")[0],
             'https://data-argo.ifremer.fr',
             'ftp://ftp.ifremer.fr/ifremer/argo',
             # 'ftp://usgodae.org/pub/outgoing/argo',  # ok, but takes too long to respond, slow down CI
             ]
# Make sure hosts are valid and available at test time:
valid_hosts = [h for h in host_list if isconnected(h) and check_gdac_path(h, errors='ignore')]

"""
List access points to be tested.
For each access points, we list 2 scenarios to make sure all possibilities are tested
"""
valid_access_points = [
    {"float": [13857]},
    # {"float": [2901746, 4902252]},
    # {"profile": [2901746, 90]},
    {"profile": [13857, 90]},
    # {"profile": [6901929, np.arange(12, 14)]},
    # {"region": [-58.3, -58, 40.1, 40.3]},
    {"region": [-20, -16., 0, 1.]},
    {"region": [-20, -16., 0, 1., "1997-07-01", "1997-09-01"]},
    # {"region": [-60, -58, 40.0, 45.0, "2007-08-01", "2007-09-01"]},
    ]


def create_fetcher(fetcher_args, access_point):
    """ Create a fetcher for given facade options and access point """
    f = ArgoIndexFetcher(**fetcher_args)
    if "float" in access_point:
        return f.float(access_point['float'])
    elif "profile" in access_point:
        return f.profile(*access_point['profile'])
    elif "region" in access_point:
        return f.region(access_point['region'])


def assert_fetcher(this_fetcher, cachable=False):
    """ Assert structure of a fetcher """
    assert isinstance(this_fetcher.to_dataframe(), pd.core.frame.DataFrame)
    # assert is_list_of_strings(this_fetcher.uri)
    assert (this_fetcher.N_RECORDS >= 1)  # Make sure we loaded the index file content
    assert (this_fetcher.N_FILES >= 1)  # Make sure we found results
    if cachable:
        assert isinstance(this_fetcher.cachepath, str)


@skip_for_debug
@requires_ftp
class Test_Backend:
    src = 'ftp'

    # Create list of tests scenarios
    # combine all hosts with all access points:
    fixtures = [(h, ap) for h in valid_hosts for ap in valid_access_points]
    fixtures_ids = ["%s, %s" % (fix[0], list(fix[1].keys())[0]) for fix in fixtures]
    fixtures_ids_short = [
        "%s, %s" % ((lambda x: 'file' if x is None else x)(split_protocol(fix[0])[0]), list(fix[1].keys())[0]) for fix in
        fixtures]

    @pytest.fixture
    def _fetcher(self, request):
        """ Fixture to create a FTP fetcher for a given host and access point """
        if isinstance(request.param, tuple):
            if 'tutorial' not in request.param[0]:
                N_RECORDS = 100
            else:
                N_RECORDS = None
            fetcher_args = {"src": self.src, "ftp": request.param[0], "cache": False, "N_RECORDS": N_RECORDS}
            yield create_fetcher(fetcher_args, request.param[1]).fetcher
        else:
            if 'tutorial' not in request.param:
                N_RECORDS = 100
            else:
                N_RECORDS = None
            fetcher_args = {"src": self.src, "ftp": request.param, "cache": False, "N_RECORDS": N_RECORDS}
            # log.debug(fetcher_args)
            # log.debug(valid_access_points[0])
            yield create_fetcher(fetcher_args, valid_access_points[0]).fetcher  # Use 1st valid access point

    @pytest.fixture
    def _cached_fetcher(self, request):
        """ Fixture to create a FTP cached fetcher for a given host and access point """
        testcachedir = tempfile.mkdtemp()
        # log.debug(type(request.param))
        if isinstance(request.param, tuple):
            if 'tutorial' not in request.param[0]:
                N_RECORDS = 100
            else:
                N_RECORDS = None
            fetcher_args = {"src": self.src, "ftp": request.param[0], "cache": True, "cachedir": testcachedir, "N_RECORDS": N_RECORDS}
            yield create_fetcher(fetcher_args, request.param[1]).fetcher
        else:
            if 'tutorial' not in request.param:
                N_RECORDS = 100
            else:
                N_RECORDS = None
            fetcher_args = {"src": self.src, "ftp": request.param, "cache": True, "cachedir": testcachedir, "N_RECORDS": N_RECORDS}
            # log.debug(fetcher_args)
            # log.debug(valid_access_points[0])
            yield create_fetcher(fetcher_args, valid_access_points[0]).fetcher  # Use 1st valid access point
        shutil.rmtree(testcachedir)

    @safe_to_server_errors
    def test_nocache(self):
        this_fetcher = create_fetcher({"src": self.src, "ftp": valid_hosts[0]}, valid_access_points[0]).fetcher
        with pytest.raises(FileSystemHasNoCache):
            this_fetcher.cachepath

    @pytest.mark.parametrize("_fetcher", valid_hosts, indirect=True)
    def test_hosts_valid(self, _fetcher):
        @safe_to_server_errors
        def test(this_fetcher):
            assert (this_fetcher.N_RECORDS >= 1)  # Make sure we loaded the index file content
        test(_fetcher)

    @pytest.mark.parametrize("ftp_host", ['invalid', 'https://invalid_ftp', 'ftp://invalid_ftp'], indirect=False)
    def test_hosts_invalid(self, ftp_host):
        # Invalid servers:
        with pytest.raises(FtpPathError):
            create_fetcher({"src": self.src, "ftp": ftp_host}, valid_access_points[0])

    @pytest.mark.parametrize("_fetcher", fixtures, indirect=True, ids=fixtures_ids_short)
    def test_fetching(self, _fetcher):
        @safe_to_server_errors
        def test(this_fetcher):
            assert_fetcher(this_fetcher, cachable=False)
        test(_fetcher)

    @pytest.mark.parametrize("_cached_fetcher", fixtures, indirect=True, ids=fixtures_ids_short)
    def test_fetching_cached(self, _cached_fetcher):
        @safe_to_server_errors
        def test(this_fetcher):
            # Assert the fetcher (this trigger data fetching, hence caching as well):
            assert_fetcher(this_fetcher, cachable=True)

            # Make sure we can clear the cache:
            this_fetcher.clear_cache()
            with pytest.raises(CacheFileNotFound):
                this_fetcher.cachepath

        test(_cached_fetcher)

"""
Test the "GDAC ftp" data fetcher backend

Here we try an approach based on fixtures and pytest parametrization
to make more explicit the full list of scenario tested.
"""
import xarray as xr

import pytest
import tempfile
import shutil
from fsspec.core import split_protocol

import argopy
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import (
    CacheFileNotFound,
    FileSystemHasNoCache,
    FtpPathError,
    InvalidMethod
)
from argopy.utilities import is_list_of_strings, isconnected
from argopy.options import check_gdac_path
from . import (
    requires_ftp,
    safe_to_server_errors
)

import logging


log = logging.getLogger("argopy.tests.data.ftp")


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
    {"float": [4902252]},
    # {"float": [2901746, 4902252]},
    {"profile": [2901746, 90]},
    # {"profile": [6901929, np.arange(12, 14)]},
    # {"region": [-58.3, -58, 40.1, 40.3, 0, 100.]},
    {"region": [-60, -58, 40.0, 45.0, 0, 100., "2007-08-01", "2007-09-01"]},
    ]

"""
List parallel methods to be tested.
"""
valid_parallel_opts = [
    # {"parallel": "thread"},
    {"parallel": True, "parallel_method": "thread"},  # opts0
    {"parallel": True, "parallel_method": "process"}  # opts1
]


def create_fetcher(fetcher_args, access_point):
    """ Create a fetcher for given facade options and access point """
    f = ArgoDataFetcher(**fetcher_args)
    if "float" in access_point:
        return f.float(access_point['float'])
    elif "profile" in access_point:
        return f.profile(*access_point['profile'])
    elif "region" in access_point:
        return f.region(access_point['region'])


def assert_fetcher(this_fetcher, cachable=False):
    """ Assert structure of a fetcher """
    assert isinstance(this_fetcher.to_xarray(), xr.Dataset)
    assert is_list_of_strings(this_fetcher.uri)
    assert (this_fetcher.N_RECORDS >= 1)  # Make sure we loaded the index file content
    assert (this_fetcher.N_FILES >= 1)  # Make sure we found results
    if cachable:
        assert is_list_of_strings(this_fetcher.cachepath)


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
            fetcher_args = {"src": self.src, "ftp": request.param[0], "cache": False}
            yield create_fetcher(fetcher_args, request.param[1]).fetcher
        else:
            fetcher_args = {"src": self.src, "ftp": request.param, "cache": False}
            # log.debug(fetcher_args)
            # log.debug(valid_access_points[0])
            yield create_fetcher(fetcher_args, valid_access_points[0]).fetcher  # Use 1st valid access point

    @pytest.fixture
    def _cached_fetcher(self, request):
        """ Fixture to create a FTP cached fetcher for a given host and access point """
        testcachedir = tempfile.mkdtemp()
        # log.debug(type(request.param))
        if isinstance(request.param, tuple):
            fetcher_args = {"src": self.src, "ftp": request.param[0], "cache": True, "cachedir": testcachedir}
            yield create_fetcher(fetcher_args, request.param[1]).fetcher
        else:
            fetcher_args = {"src": self.src, "ftp": request.param, "cache": True, "cachedir": testcachedir}
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

    @pytest.mark.parametrize("_cached_fetcher", fixtures, indirect=True, ids=fixtures_ids_short)
    def test_fetching_cached(self, _cached_fetcher):
        @safe_to_server_errors
        def test(this_fetcher):
            # Before data fetching, the cache does not exist yet:
            with pytest.raises(CacheFileNotFound):
                this_fetcher.cachepath

            # Assert the fetcher (this trigger data fetching, hence caching as well):
            assert_fetcher(this_fetcher, cachable=True)

            # Finally make sure we can clear the cache:
            this_fetcher.clear_cache()
            with pytest.raises(CacheFileNotFound):
                this_fetcher.cachepath

        test(_cached_fetcher)

    @pytest.mark.parametrize("_fetcher", fixtures, indirect=True, ids=fixtures_ids_short)
    def test_fetching(self, _fetcher):
        @safe_to_server_errors
        def test(this_fetcher):
            assert_fetcher(this_fetcher, cachable=False)
        test(_fetcher)


@requires_ftp
class Test_BackendParallel:
    src = 'ftp'

    # Create list of tests scenarios
    # combine all hosts with all access points and valid parallel options:
    # fixtures = [(h, mth, ap) for h in valid_hosts for ap in valid_access_points for mth in valid_parallel_opts]
    fixtures = [(h, mth, ap) for h in valid_hosts for ap in valid_access_points if 'float' in ap for mth in
                valid_parallel_opts]
    fixtures_ids = [
        "%s, %s, %s" % (
            fix[0],
            (lambda x: x['parallel_method'] if 'parallel_method' in x else x['parallel'])(fix[1]),
            list(fix[2].keys())[0])
        for fix in fixtures]
    fixtures_ids_short = [
        "%s, %s, %s" % (
            (lambda x: 'file' if x is None else x)(split_protocol(fix[0])[0]),
            (lambda x: x['parallel_method'] if 'parallel_method' in x else x['parallel'])(fix[1]),
            list(fix[2].keys())[0])
        for fix in fixtures]

    @pytest.fixture
    def _fetcher(self, request):
        """ Fixture to create a FTP fetcher for a given host and access point """
        fetcher_args = {"src": self.src, "ftp": request.param[0], "cache": False, **request.param[1]}
        yield create_fetcher(fetcher_args, request.param[2]).fetcher

    @pytest.mark.parametrize("opts", [
        {"parallel": True, "parallel_method": "invalid"},  # opts0
        {"parallel": "invalid"}  # opts1
        ], indirect=False)
    def test_methods_invalid(self, opts):
        with pytest.raises(InvalidMethod):
            this_fetcher = create_fetcher({**{"src": self.src}, **opts}, valid_access_points[0]).fetcher
            assert_fetcher(this_fetcher)

    @pytest.mark.parametrize("_fetcher", fixtures, ids=fixtures_ids_short, indirect=True)
    def test_fetching(self, _fetcher):
        @safe_to_server_errors
        def test(this_fetcher):
            assert_fetcher(this_fetcher)
        test(_fetcher)

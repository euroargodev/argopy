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
from argopy.utilities import is_list_of_strings, isconnected
from argopy.options import check_gdac_path
from utils import (
    requires_gdac,
    safe_to_server_errors,
    fct_safe_to_server_errors
)

import logging
import warnings

log = logging.getLogger("argopy.tests.index.gdac")
skip_for_debug = pytest.mark.skipif(False, reason="Taking too long !")


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


@requires_gdac
def create_fetcher(fetcher_args, access_point, xfail=False):
    """ Create a fetcher for given facade options and access point

        Use xfail=True when a test with this fetcher is expected to fail
    """
    def core(fargs, apts):
        try:
            f = ArgoIndexFetcher(**fargs)
            if "float" in apts:
                f = f.float(apts['float'])
            elif "profile" in apts:
                f = f.profile(*apts['profile'])
            elif "region" in apts:
                f = f.region(apts['region'])
        except Exception:
            raise
        return f
    return fct_safe_to_server_errors(core)(fetcher_args, access_point, xfail=xfail)


def assert_fetcher(this_fetcher, cacheable=False):
    """ Assert structure of a fetcher """
    assert isinstance(this_fetcher.to_dataframe(), pd.core.frame.DataFrame)
    # assert is_list_of_strings(this_fetcher.uri)
    assert (this_fetcher.N_RECORDS >= 1)  # Make sure we loaded the index file content
    assert (this_fetcher.N_FILES >= 1)  # Make sure we found results
    if cacheable:
        assert is_list_of_strings(this_fetcher.cachepath)


@skip_for_debug
@requires_gdac
class TestBackend:
    src = 'gdac'

    # Create list of tests scenarios
    # combine all hosts with all access points:
    scenarios = [(h, ap) for h in valid_hosts for ap in valid_access_points]
    # scenarios = [(valid_hosts[0], valid_access_points[0])]
    # scenarios = [(valid_hosts[1], ap) for ap in valid_access_points]
    # scenarios_ids = ["%s, %s" % (fix[0], list(fix[1].keys())[0]) for fix in scenarios]
    scenarios_ids = [
        "%s, %s" % ((lambda x: 'file' if x is None else x)(split_protocol(fix[0])[0]), list(fix[1].keys())[0]) for fix
        in
        scenarios]

    def setup_class(self):
        """setup any state specific to the execution of the given class"""
        # Create the cache folder here, so that it's not the same for the pandas and pyarrow tests
        self.cachedir = tempfile.mkdtemp()

    def _setup_fetcher(self, this_request, cached=False):
        """Helper method to set-up options for a fetcher creation"""
        if isinstance(this_request.param, tuple):
            ftp = this_request.param[0]
            access_point = this_request.param[1]
        else:
            ftp = this_request.param
            access_point = valid_access_points[0]  # Use 1st valid access point

        N_RECORDS = None if 'tutorial' in ftp else 100  # Make sure we're not going to load the full index
        fetcher_args = {"src": self.src, "ftp": ftp, "cache": False, "N_RECORDS": N_RECORDS}
        if cached:
            fetcher_args = {**fetcher_args, **{"cache": True, "cachedir": self.cachedir}}
            # fetcher_args = {**fetcher_args, **{"cache": True, "cachedir": tempfile.mkdtemp()}}
        if not isconnected(fetcher_args['ftp']+"/"+"ar_index_global_prof.txt"):
            pytest.xfail("Fails because %s not available" % fetcher_args['ftp'])
        else:
            return fetcher_args, access_point

    @pytest.fixture
    def _make_a_fetcher(self, request):
        """ Fixture to create a FTP fetcher for a given host and access point """
        fetcher_args, access_point = self._setup_fetcher(request, cached=False)
        yield create_fetcher(fetcher_args, access_point).fetcher

    @pytest.fixture
    def _make_a_cached_fetcher(self, request):
        """ Fixture to create a cached FTP fetcher for a given host and access point """
        fetcher_args, access_point = self._setup_fetcher(request, cached=True)
        yield create_fetcher(fetcher_args, access_point).fetcher

    # @skip_for_debug
    @safe_to_server_errors
    def test_nocache(self):
        this_fetcher = create_fetcher({"src": self.src, "ftp": valid_hosts[0]}, valid_access_points[0]).fetcher
        with pytest.raises(FileSystemHasNoCache):
            this_fetcher.cachepath

    # @skip_for_debug
    @pytest.mark.parametrize("_make_a_fetcher", valid_hosts, indirect=True)
    def test_hosts(self, _make_a_fetcher):
        @safe_to_server_errors
        def test(this_fetcher):
            assert (this_fetcher.N_RECORDS >= 1)  # Make sure we loaded the index file content
        test(_make_a_fetcher)

    # @skip_for_debug
    @pytest.mark.parametrize("ftp_host", ['invalid', 'https://invalid_ftp', 'ftp://invalid_ftp'], indirect=False)
    def test_hosts_invalid(self, ftp_host):
        # Invalid servers:
        with pytest.raises(FtpPathError):
            create_fetcher({"src": self.src, "ftp": ftp_host}, valid_access_points[0], xfail=True)

    # @skip_for_debug
    @pytest.mark.parametrize("_make_a_fetcher", scenarios, indirect=True, ids=scenarios_ids)
    def test_fetching(self, _make_a_fetcher):
        @safe_to_server_errors
        def test(this_fetcher):
            assert_fetcher(this_fetcher, cacheable=False)
        test(_make_a_fetcher)

    # @skip_for_debug
    @pytest.mark.parametrize("_make_a_cached_fetcher", scenarios, indirect=True, ids=scenarios_ids)
    def test_fetching_cached(self, _make_a_cached_fetcher):
        @safe_to_server_errors
        def test(this_fetcher):
            # Assert the fetcher (this trigger data fetching, hence caching as well):
            assert_fetcher(this_fetcher, cacheable=True)

            # Make sure we can clear the cache:
            this_fetcher.clear_cache()
            with pytest.raises(CacheFileNotFound):
                this_fetcher.cachepath

        test(_make_a_cached_fetcher)

    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self, request):
        """Cleanup once we are finished."""
        def remove_test_dir():
            # warnings.warn("\n%s" % argopy.lscache(self.cachedir))  # This could be useful to debug tests
            shutil.rmtree(self.cachedir)
        request.addfinalizer(remove_test_dir)


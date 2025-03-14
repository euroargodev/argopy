import pandas as pd

import pytest
import shutil
from urllib.parse import urlparse
import logging

import argopy
from argopy import IndexFetcher as ArgoIndexFetcher
from argopy.errors import CacheFileNotFound, FileSystemHasNoCache, GdacPathError
from argopy.utils.checkers import isconnected, is_list_of_strings

from utils import requires_gdac, create_temp_folder, patch_ftp
from mocked_http import mocked_httpserver
from mocked_http import mocked_server_address as MOCKHTTP


log = logging.getLogger("argopy.tests.index.gdac")
skip_for_debug = pytest.mark.skipif(True, reason="Taking too long !")


"""
List GDAC hosts to be tested. 
Since the fetcher is compatible with host from local, http or ftp protocols, we
try to test them all:
"""
VALID_HOSTS = [
    argopy.tutorial.open_dataset("gdac")[0],
    #'https://data-argo.ifremer.fr',
    MOCKHTTP,
    # 'ftp://ftp.ifremer.fr/ifremer/argo',
    "MOCKFTP",
]

"""
List access points to be tested.
For each access points, we list 1-to-2 scenario to make sure all possibilities are tested
"""
VALID_ACCESS_POINTS = [
    {"float": [13857]},
    {"profile": [13857, 90]},
    {"region": [-20, -16.0, 0, 1]},
    {"region": [-20, -16.0, 0, 1, "1997-07-01", "1997-09-01"]},
]

"""
List parallel methods to be tested.
"""
valid_parallel_opts = [
    {"parallel": "thread"},
    # {"parallel": True, "parallel_method": "thread"},  # opts0
    # {"parallel": True, "parallel_method": "process"}  # opts1
]


@requires_gdac
def create_fetcher(fetcher_args, access_point, xfail=False):
    """Create a fetcher for a given set of facade options and access point

    Use xfail=True when a test with this fetcher is expected to fail
    """

    def core(fargs, apts):
        try:
            f = ArgoIndexFetcher(**fargs)
            if "float" in apts:
                f = f.float(apts["float"])
            elif "profile" in apts:
                f = f.profile(*apts["profile"])
            elif "region" in apts:
                f = f.region(apts["region"])
        except Exception:
            raise
        return f

    fetcher = core(fetcher_args, access_point).fetcher
    return fetcher


def assert_fetcher(mocked_erddapserver, this_fetcher, cacheable=False):
    """Assert a data fetcher.

    This should be used by all tests
    """
    assert isinstance(this_fetcher.to_dataframe(), pd.core.frame.DataFrame)
    assert this_fetcher.N_RECORDS >= 1  # Make sure we loaded the index file content
    assert this_fetcher.N_FILES >= 1  # Make sure we found results
    if cacheable:
        assert is_list_of_strings(this_fetcher.cachepath)


def gdac_shortname(gdac):
    """Get a short name for scenarios IDs, given a FTP host"""
    if gdac == "MOCKFTP":
        return "ftp_mocked"
    elif "localhost" in gdac or "127.0.0.1" in gdac:
        return "http_mocked"
    else:
        return (lambda x: "file" if x == "" else x)(urlparse(gdac).scheme)


@requires_gdac
class TestBackend:
    src = "gdac"

    # Create list of tests scenarios
    # combine all hosts with all access points:
    scenarios = [(h, ap) for h in VALID_HOSTS for ap in VALID_ACCESS_POINTS]

    scenarios_ids = [
        "%s, %s" % (gdac_shortname(fix[0]), list(fix[1].keys())[0]) for fix in scenarios
    ]

    #############
    # UTILITIES #
    #############

    def setup_class(self):
        """setup any state specific to the execution of the given class"""
        # Create the cache folder here, so that it's not the same for the pandas and pyarrow tests
        self.cachedir = create_temp_folder().folder

    def _patch_gdac(self, gdac):
        return patch_ftp(gdac)

    def _setup_fetcher(self, this_request, cached=False):
        """Helper method to set up options for a fetcher creation"""
        if isinstance(this_request.param, tuple):
            gdac = this_request.param[0]
            access_point = this_request.param[1]
        else:
            gdac = this_request.param
            access_point = VALID_ACCESS_POINTS[0]  # Use 1st valid access point

        N_RECORDS = (
            None if "tutorial" in gdac or "MOCK" in gdac else 100
        )  # Make sure we're not going to load the full index
        fetcher_args = {
            "src": self.src,
            "gdac": self._patch_gdac(gdac),
            "cache": False,
            "N_RECORDS": N_RECORDS,
        }

        if cached:
            fetcher_args = {
                **fetcher_args,
                **{"cache": True, "cachedir": self.cachedir},
            }
        if not isconnected(fetcher_args["gdac"]):
            pytest.xfail("Fails because %s not available" % fetcher_args["gdac"])
        else:
            return fetcher_args, access_point

    @pytest.fixture
    def _make_a_fetcher(self, request, mocked_httpserver):
        """Fixture to create a GDAC fetcher for a given host and access point"""
        fetcher_args, access_point = self._setup_fetcher(request, cached=False)
        yield create_fetcher(fetcher_args, access_point)

    @pytest.fixture
    def _make_a_cached_fetcher(self, request):
        """Fixture to create a cached FTP fetcher for a given host and access point"""
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
    def test_nocache(self, mocked_httpserver):
        this_fetcher = create_fetcher(
            {
                "src": self.src,
                "gdac": self._patch_gdac(VALID_HOSTS[0]),
                "N_RECORDS": 10,
            },
            VALID_ACCESS_POINTS[0],
        )
        with pytest.raises(FileSystemHasNoCache):
            this_fetcher.cachepath

    @pytest.mark.parametrize(
        "_make_a_fetcher",
        VALID_HOSTS,
        indirect=True,
        ids=["%s" % gdac_shortname(host) for host in VALID_HOSTS],
    )
    def test_hosts(self, mocked_httpserver, _make_a_fetcher):
        assert _make_a_fetcher.N_RECORDS >= 1

    @pytest.mark.parametrize(
        "gdac_host",
        ["invalid", "https://invalid_gdac", "ftp://invalid_ftp"],
        indirect=False,
    )
    def test_hosts_invalid(self, gdac_host):
        # Invalid servers:
        with pytest.raises(GdacPathError):
            create_fetcher({"src": self.src, "gdac": gdac_host}, VALID_ACCESS_POINTS[0])

    @pytest.mark.parametrize(
        "_make_a_fetcher", scenarios, indirect=True, ids=scenarios_ids
    )
    def test_fetching(self, mocked_httpserver, _make_a_fetcher):
        assert_fetcher(mocked_httpserver, _make_a_fetcher, cacheable=False)

    @pytest.mark.parametrize(
        "_make_a_cached_fetcher", scenarios, indirect=True, ids=scenarios_ids
    )
    def test_fetching_cached(self, mocked_httpserver, _make_a_cached_fetcher):
        # Assert the fetcher (this trigger data fetching, hence caching as well):
        assert_fetcher(mocked_httpserver, _make_a_cached_fetcher, cacheable=True)
        # and we also make sure we can clear the cache:
        _make_a_cached_fetcher.clear_cache()
        with pytest.raises(CacheFileNotFound):
            _make_a_cached_fetcher.cachepath

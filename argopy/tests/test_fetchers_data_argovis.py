import logging

from argopy import DataFetcher as ArgoDataFetcher
from argopy.utils.checkers import is_list_of_strings

import pytest
import xarray as xr
from utils import (
    requires_argovis,
    create_temp_folder,
)

from mocked_http import mocked_server_address
from mocked_http import mocked_httpserver as mocked_argovisserver

import shutil
from collections import ChainMap


log = logging.getLogger("argopy.tests.data.argovis")

USE_MOCKED_SERVER = True

"""
List access points to be tested for each datasets: phy and ref.
For each access points, we list 1-to-2 scenario to make sure all possibilities are tested
"""
ACCESS_POINTS = [
    {"phy": [
            {"float": 1901393},
            {"float": [1901393, 6902746]},
            {"profile": [6902746, 34]},
            {"profile": [6902746, [1, 12]]},
            {"region": [-70, -65, 35.0, 40.0, 0, 10.0, "2012-01", "2012-03"]},
            {"region": [-70, -65, 35.0, 40.0, 0, 10.0, "2012-01", "2012-06"]},
    ]},
]
PARALLEL_ACCESS_POINTS = [
    {"phy": [
        {"float": [1900468, 1900117, 1900386]},
        {"region": [-70, -65, 35.0, 40.0, 0, 10.0, "2012-01", "2012-06"]},
    ]},
]

"""
List user modes to be tested
"""
USER_MODES = ['standard']


"""
Make a list of VALID dataset/access_points to be tested
"""
VALID_ACCESS_POINTS, VALID_ACCESS_POINTS_IDS = [], []
for entry in ACCESS_POINTS:
    for ds in entry:
        for mode in USER_MODES:
            for ap in entry[ds]:
                VALID_ACCESS_POINTS.append({'ds': ds, 'mode': mode, 'access_point': ap})
                VALID_ACCESS_POINTS_IDS.append("ds='%s', mode='%s', %s" % (ds, mode, ap))


VALID_PARALLEL_ACCESS_POINTS, VALID_PARALLEL_ACCESS_POINTS_IDS = [], []
for entry in PARALLEL_ACCESS_POINTS:
    for ds in entry:
        for mode in USER_MODES:
            for ap in entry[ds]:
                VALID_PARALLEL_ACCESS_POINTS.append({'ds': ds, 'mode': mode, 'access_point': ap})
                VALID_PARALLEL_ACCESS_POINTS_IDS.append("ds='%s', mode='%s', %s" % (ds, mode, ap))


def create_fetcher(fetcher_args, access_point):
    """ Create a fetcher for a given set of facade options and access point """
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


def assert_fetcher(mocked_argovisserver, this_fetcher, cacheable=False):
    """Assert a data fetcher.

        This should be used by all tests asserting a fetcher
    """
    def assert_all(this_fetcher, cacheable):
        # We use the facade to test 'to_xarray' in order to make sure to test all filters required by user mode
        ds = this_fetcher.to_xarray(errors='raise')
        assert isinstance(ds, xr.Dataset)
        #
        core = this_fetcher.fetcher
        assert is_list_of_strings(core.uri)
        if cacheable:
            assert is_list_of_strings(core.cachepath)

        # log.debug("In assert, this fetcher is in '%s' user mode" % this_fetcher._mode)
        if this_fetcher._mode == 'standard':
            assert 'PRES_ADJUSTED' not in ds
        else:
            raise ValueError("Un-expected user mode for argovis, only support 'standard'")

    try:
        assert_all(this_fetcher, cacheable)
    except:
        raise


@requires_argovis
class Test_Backend:
    """ Test ERDDAP data fetching backend """
    src = 'argovis'

    #############
    # UTILITIES #
    #############

    def setup_class(self):
        """setup any state specific to the execution of the given class"""
        # Create the cache folder here, so that it's not the same for the pandas and pyarrow tests
        self.cachedir = create_temp_folder().folder

    def _setup_fetcher(self, this_request, cached=False, parallel=False):
        """Helper method to set up options for a fetcher creation"""
        defaults_args = {"src": self.src,
                         "cache": cached,
                         "cachedir": self.cachedir,
                         "parallel": parallel,
                         }
        if USE_MOCKED_SERVER:
            defaults_args['server'] = mocked_server_address

        dataset = this_request.param['ds']
        user_mode = this_request.param['mode']
        access_point = this_request.param['access_point']

        fetcher_args = ChainMap(defaults_args, {"ds": dataset, 'mode': user_mode})
        if not cached:
            # cache is False by default, so we don't need to clutter the arguments list
            del fetcher_args["cache"]
            del fetcher_args["cachedir"]
        if not parallel:
            # parallel is False by default, so we don't need to clutter the arguments list
            del fetcher_args["parallel"]

        # log.debug("Setting up a new fetcher with the following arguments:")
        # log.debug(fetcher_args)
        return fetcher_args, access_point

    @pytest.fixture
    def fetcher(self, request):
        """ Fixture to create a data fetcher for a given dataset and access point """
        fetcher_args, access_point = self._setup_fetcher(request, cached=False)
        yield create_fetcher(fetcher_args, access_point)

    @pytest.fixture
    def cached_fetcher(self, request):
        """ Fixture to create a cached data fetcher for a given dataset and access point """
        fetcher_args, access_point = self._setup_fetcher(request, cached=True)
        yield create_fetcher(fetcher_args, access_point)

    @pytest.fixture
    def parallel_fetcher(self, request):
        """ Fixture to create a parallel data fetcher for a given dataset and access point """
        fetcher_args, access_point = self._setup_fetcher(request, parallel="thread")
        yield create_fetcher(fetcher_args, access_point)

    def teardown_class(self):
        """Cleanup once we are finished."""
        def remove_test_dir():
            shutil.rmtree(self.cachedir)
        remove_test_dir()

    #########
    # TESTS #
    #########
    @pytest.mark.parametrize("fetcher", VALID_ACCESS_POINTS,
                             indirect=True,
                             ids=VALID_ACCESS_POINTS_IDS)
    def test_fetching(self, mocked_argovisserver, fetcher):
        assert_fetcher(mocked_argovisserver, fetcher, cacheable=False)

    @pytest.mark.parametrize("cached_fetcher", VALID_ACCESS_POINTS,
                             indirect=True,
                             ids=VALID_ACCESS_POINTS_IDS)
    def test_fetching_cached(self, mocked_argovisserver, cached_fetcher):
        assert_fetcher(mocked_argovisserver, cached_fetcher, cacheable=True)

    @pytest.mark.parametrize("parallel_fetcher", VALID_PARALLEL_ACCESS_POINTS,
                             indirect=True,
                             ids=VALID_PARALLEL_ACCESS_POINTS_IDS)
    def test_fetching_parallel_thread(self, mocked_argovisserver, parallel_fetcher):
        assert_fetcher(mocked_argovisserver, parallel_fetcher, cacheable=False)

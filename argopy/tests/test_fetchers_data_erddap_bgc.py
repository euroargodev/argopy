import logging
import numpy as np

from argopy import DataFetcher as ArgoDataFetcher
from argopy.utils.checkers import is_list_of_strings
from argopy.stores import indexstore_pd as ArgoIndex  # make sure to work with the Pandas index store with erddap-bgc

import pytest
import xarray as xr
from utils import (
    requires_erddap,
)
from mocked_http import mocked_server_address
from mocked_http import mocked_httpserver as mocked_erddapserver

import tempfile
import shutil
from collections import ChainMap


log = logging.getLogger("argopy.tests.data.erddap")

USE_MOCKED_SERVER = True

"""
List access points to be tested for each datasets: bgc.
For each access points, we list 1-to-2 scenario to make sure all possibilities are tested
"""
ACCESS_POINTS = [
    {"bgc": [
        {"float": 5903248},
        {"float": [5903248, 6904241]},
        {"profile": [5903248, 34]},
        {"profile": [5903248, np.arange(12, 14)]},
        {"region": [-55, -47, 55, 57, 0, 10]},
        {"region": [-55, -47, 55, 57, 0, 10, "2022-05-1", "2023-07-01"]},
    ]},
]
PARALLEL_ACCESS_POINTS = [
    {"bgc": [
        {"float": [5903248, 6904241]},
        {"region": [-55, -47, 55, 57, 0, 10, "2022-05-1", "2023-07-01"]},
    ]},
]

"""
List user modes to be tested
"""
# USER_MODES = ['standard', 'expert', 'research']
USER_MODES = ['expert']

"""
List of 'params' fetcher arguments to be tested
"""
PARAMS = ['all', 'DOXY']

"""
Make a list of VALID dataset/access_points to be tested
"""
VALID_ACCESS_POINTS, VALID_ACCESS_POINTS_IDS = [], []
for entry in ACCESS_POINTS:
    for ds in entry:
        for mode in USER_MODES:
            for params in PARAMS:
                for ap in entry[ds]:
                    VALID_ACCESS_POINTS.append({'ds': ds, 'mode': mode, 'params': params, 'access_point': ap})
                    VALID_ACCESS_POINTS_IDS.append("ds='%s', mode='%s', params='%s', %s" % (ds, mode, params, ap))


VALID_PARALLEL_ACCESS_POINTS, VALID_PARALLEL_ACCESS_POINTS_IDS = [], []
for entry in PARALLEL_ACCESS_POINTS:
    for ds in entry:
        for mode in USER_MODES:
            for params in PARAMS:
                for ap in entry[ds]:
                    VALID_PARALLEL_ACCESS_POINTS.append({'ds': ds, 'mode': mode, 'params': params, 'access_point': ap})
                    VALID_PARALLEL_ACCESS_POINTS_IDS.append("ds='%s', mode='%s', params='%s', %s" % (ds, mode, params, ap))


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


def assert_fetcher(mocked_erddapserver, this_fetcher, cacheable=False):
    """Assert a data fetcher.

        This should be used by all tests asserting a fetcher
    """
    def assert_all(this_fetcher, cacheable):

        # We use the facade to test 'to_xarray' in order to make sure to test all filters required by user mode
        ds = this_fetcher.to_xarray(errors='raise')
        assert isinstance(ds, xr.Dataset)

        # Then apply checks on erddap fetcher:
        core = this_fetcher.fetcher
        assert is_list_of_strings(core.uri)
        assert (core.N_POINTS >= 1)  # Make sure we found results
        if cacheable:
            assert is_list_of_strings(core.cachepath)

        # log.debug("In assert, this fetcher is in '%s' user mode" % this_fetcher._mode)
        if this_fetcher._mode == 'expert':
            assert 'PRES_ADJUSTED' in ds

        elif this_fetcher._mode == 'standard':
            assert 'PRES_ADJUSTED' not in ds

        elif this_fetcher._mode == 'research':
            assert 'PRES_ADJUSTED' not in ds
            assert 'PRES_QC' not in ds

    try:
        assert_all(this_fetcher, cacheable)
    except Exception as e:
        if this_fetcher._mode not in ['expert']:
            pytest.xfail("BGC is not yet supported in '%s' user mode" % this_fetcher._mode)
        else:
            log.debug("Fetcher instance assert false because: %s" % e)
            assert False


@requires_erddap
class Test_Backend:
    """ Test ERDDAP data fetching backend """
    src = 'erddap'

    #############
    # UTILITIES #
    #############

    def setup_class(self):
        """setup any state specific to the execution of the given class"""
        # Create the cache folder here, so that it's not the same for the pandas and pyarrow tests
        self.cachedir = tempfile.mkdtemp()

    def _setup_fetcher(self, this_request, cached=False, parallel=False):
        """Helper method to set up options for a fetcher creation"""
        defaults_args = {"src": self.src,
                         "cache": cached,
                         "cachedir": self.cachedir,
                         "parallel": parallel,
                         }

        if USE_MOCKED_SERVER:
            defaults_args['server'] = mocked_server_address
            defaults_args['indexfs'] = ArgoIndex(
                                            host=mocked_server_address,
                                            index_file='argo_synthetic-profile_index.txt',
                                            cache=cached,
                                            cachedir=self.cachedir,
                                            timeout=5,
                                        )

        dataset = this_request.param['ds']
        user_mode = this_request.param['mode']
        params = this_request.param['params']
        measured = this_request.param['measured'] if 'measured' in this_request.param else None

        access_point = this_request.param['access_point']

        fetcher_args = ChainMap(defaults_args, {"ds": dataset, 'mode': user_mode, 'params': params, 'measured': measured})
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
        """ Fixture to create a ERDDAP data fetcher for a given dataset and access point """
        fetcher_args, access_point = self._setup_fetcher(request, cached=False)
        yield create_fetcher(fetcher_args, access_point)

    @pytest.fixture
    def cached_fetcher(self, request):
        """ Fixture to create a cached ERDDAP data fetcher for a given dataset and access point """
        fetcher_args, access_point = self._setup_fetcher(request, cached=True)
        yield create_fetcher(fetcher_args, access_point)

    @pytest.fixture
    def parallel_fetcher(self, request):
        """ Fixture to create a parallel ERDDAP data fetcher for a given dataset and access point """
        fetcher_args, access_point = self._setup_fetcher(request, parallel="erddap")
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
    def test_fetching(self, mocked_erddapserver, fetcher):
        assert_fetcher(mocked_erddapserver, fetcher, cacheable=False)

    @pytest.mark.parametrize("cached_fetcher", VALID_ACCESS_POINTS,
                             indirect=True,
                             ids=VALID_ACCESS_POINTS_IDS)
    def test_fetching_cached(self, mocked_erddapserver, cached_fetcher):
        assert_fetcher(mocked_erddapserver, cached_fetcher, cacheable=True)

    @pytest.mark.parametrize("parallel_fetcher", VALID_PARALLEL_ACCESS_POINTS,
                             indirect=True,
                             ids=VALID_PARALLEL_ACCESS_POINTS_IDS)
    def test_fetching_parallel(self, mocked_erddapserver, parallel_fetcher):
        assert_fetcher(mocked_erddapserver, parallel_fetcher, cacheable=False)

    @pytest.mark.parametrize("measured", [None, 'all', 'DOXY'],
                             indirect=False,
                             ids=["measured=%s" % m for m in [None, 'all', 'DOXY']]
                             )
    def test_fetching_measured(self, mocked_erddapserver, measured):
        class this_request:
            param = {
                'ds': 'bgc',
                'mode': 'expert',
                'params': 'all',
                'measured': measured,
                'access_point': {"float": [5903248]},
            }
        fetcher_args, access_point = self._setup_fetcher(this_request)
        fetcher = create_fetcher(fetcher_args, access_point)
        assert_fetcher(mocked_erddapserver, fetcher)

    # @pytest.mark.parametrize("measured", ['all'],
    #                          indirect=False,
    #                          ids=["measured=%s" % m for m in ['all']],
    #                          )
    # def test_fetching_failed_measured(self, mocked_erddapserver, measured):
    #     class this_request:
    #         param = {
    #             'ds': 'bgc',
    #             'mode': 'expert',
    #             'params': 'all',
    #             'measured': measured,
    #             'access_point': {"float": [6904240]},
    #         }
    #     fetcher_args, access_point = self._setup_fetcher(this_request)
    #     fetcher = create_fetcher(fetcher_args, access_point)
    #     with pytest.raises(ValueError):
    #         fetcher.to_xarray()

import pytest

dask = pytest.importorskip("dask", reason="Requires 'Dask' and 'distributed'")
distributed = pytest.importorskip("distributed", reason="Requires 'Dask' and 'distributed'")
from dask.distributed import Client


import logging
from argopy import DataFetcher
from collections import ChainMap
import xarray as xr

from mocked_http import mocked_server_address, mocked_httpserver
from utils import (
    requires_argovis,
    requires_erddap,
    requires_gdac,
)

log = logging.getLogger("argopy.tests.dask")
USE_MOCKED_SERVER = True

"""
List data sources to be tested

We use these 3 data sources in order to test the 3 dask client implementation of the corresponding file store methods:

erddap : httpstore.open_mfdataset
argovis: httstore.open_mfjson
gdac: filestore.open_mfdataset

"""
SRC_LIST = ["erddap", "argovis", "gdac"]


"""
List access points to be tested for each datasets: phy.
For each access points, we list 1-to-2 scenario to make sure all possibilities are tested
"""
PARALLEL_ACCESS_POINTS = [
    {
        "phy": [
            {"region": [-60, -55, 40.0, 45.0, 0.0, 20.0, "2007-08-01", "2007-09-01"]},
        ]
    },
]

"""
List user modes to be tested
"""
USER_MODES = ["standard"]  # Because it's the only available with argovis

"""
Make a list of VALID dataset/access_points to be tested
"""
VALID_PARALLEL_ACCESS_POINTS, VALID_PARALLEL_ACCESS_POINTS_IDS = [], []
for entry in PARALLEL_ACCESS_POINTS:
    for src in SRC_LIST:
        for ds in entry:
            for mode in USER_MODES:
                for ap in entry[ds]:
                    VALID_PARALLEL_ACCESS_POINTS.append(
                        {"src": src, "ds": ds, "mode": mode, "access_point": ap}
                    )
                    VALID_PARALLEL_ACCESS_POINTS_IDS.append(
                        "src='%s', ds='%s', mode='%s', %s" % (src, ds, mode, ap)
                    )


def create_fetcher(fetcher_args, access_point):
    """Create a fetcher for a given set of facade options and access point"""

    def core(fargs, apts):
        try:
            f = DataFetcher(**fargs)
            if "float" in apts:
                f = f.float(apts["float"])
            elif "profile" in apts:
                f = f.profile(*apts["profile"])
            elif "region" in apts:
                f = f.region(apts["region"])
        except Exception:
            raise
        return f

    fetcher = core(fetcher_args, access_point)
    return fetcher


@requires_erddap
@requires_gdac
@requires_argovis
class Test_Backend:
    """Test Dask cluster parallelization"""

    #############
    # UTILITIES #
    #############
    def setup_class(self):
        """setup any state specific to the execution of the given class"""
        # Create the cache folder here, so that it's not the same for the pandas and pyarrow tests
        self.client = Client(processes=True)
        log.debug("Dask dashboard: %s" % self.client.dashboard_link)

    def _test2fetcherargs(self, this_request):
        """Helper method to set up options for a fetcher creation"""
        defaults_args = {
            "parallel": self.client,
            "chunks_maxsize": {"lon": 2.5, "lat": 2.5},
        }
        if USE_MOCKED_SERVER:
            defaults_args["server"] = mocked_server_address

        src = this_request.param["src"]
        dataset = this_request.param["ds"]
        user_mode = this_request.param["mode"]
        access_point = this_request.param["access_point"]

        fetcher_args = ChainMap(
            defaults_args,
            {
                "src": src,
                "ds": dataset,
                "mode": user_mode,
            },
        )

        # log.debug("Setting up fetcher arguments:%s" % fetcher_args)
        return fetcher_args, access_point

    @pytest.fixture
    def fetcher(self, request):
        """Fixture to create a data fetcher for a given dataset and access point"""
        fetcher_args, access_point = self._test2fetcherargs(request)
        yield create_fetcher(fetcher_args, access_point)

    def teardown_class(self):
        """Cleanup once we are finished."""
        self.client.close()

    #########
    # TESTS #
    #########
    @pytest.mark.parametrize(
        "fetcher",
        VALID_PARALLEL_ACCESS_POINTS,
        indirect=True,
        ids=VALID_PARALLEL_ACCESS_POINTS_IDS,
    )
    def test_parallel_data_fetching(self, mocked_httpserver, fetcher):
        assert len(fetcher.uri) > 1

        ds = fetcher.to_xarray()
        assert isinstance(ds, xr.Dataset)

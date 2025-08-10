import os

import pytest
import tempfile

import numpy as np
import pandas as pd
import importlib
import shutil
import logging
from urllib.parse import urlparse

import argopy
from argopy.errors import (
    GdacPathError,
    OptionValueError,
    InvalidDatasetStructure,
)
from argopy.utils.checkers import is_list_of_strings, is_wmo
from argopy.stores.index import indexstore_pd
from argopy.stores import ArgoFloat
from utils import create_temp_folder
from mocked_http import mocked_httpserver, mocked_server_address
from utils import patch_ftp


log = logging.getLogger("argopy.tests.indexstores")

has_pyarrow = importlib.util.find_spec("pyarrow") is not None
skip_nopyarrow = pytest.mark.skipif(not has_pyarrow, reason="Requires pyarrow")

skip_pandas = pytest.mark.skipif(0, reason="Skipped tests for Pandas backend")
skip_pyarrow = pytest.mark.skipif(0, reason="Skipped tests for Pyarrow backend")
skip_CORE = pytest.mark.skipif(0, reason="Skipped tests for CORE index")
skip_BGCs = pytest.mark.skipif(0, reason="Skipped tests for BGC synthetic index")
skip_BGCb = pytest.mark.skipif(0, reason="Skipped tests for BGC bio index")

"""
List gdac hosts to be tested. 
Since the fetcher is compatible with host from local, http, ftp or s3 protocols, we try to test them all:
"""
VALID_HOSTS = [
    argopy.tutorial.open_dataset("gdac")[0],  # Use local files
    mocked_server_address,  # Use the mocked http server
    "MOCKFTP",  # keyword to use a fake/mocked ftp server (running on localhost)
]

HAS_S3FS = importlib.util.find_spec("s3fs") is not None
if HAS_S3FS:
    # todo Create a mocked server for s3 tests
    VALID_HOSTS.append("s3://argo-gdac-sandbox/pub/idx")

"""
List index searches to be tested.
"""
VALID_SEARCHES = [
    # {"wmo": [13857]},
    {"wmo": [3902131]},  # BGC
    {"wmo": [6901929, 2901623]},
    {"cyc": [5]},
    {"cyc": [5, 45]},
    # {"wmo_cyc": [13857, 2]},
    {"wmo_cyc": [3902131, 2]},  # BGC
    {"date": [-60, -40, 40.0, 60.0, "2007-08-01", "2007-09-01"]},
    {"lon": [-60, -40, 40.0, 60.0, "2007-08-01", "2007-09-01"]},
    {"lat": [-60, -40, 40.0, 60.0, "2007-08-01", "2007-09-01"]},
    {"lon_lat": [-60, -40, 40.0, 60.0, "2007-08-01", "2007-09-01"]},
    {"box": [-60, -40, 40.0, 60.0, "2007-08-01", "2007-09-01"]},
    {"profiler_type": [845]},
    {"profiler_label": 'ARVOR'},
]
VALID_SEARCHES_LOGICAL = [
    {"params": ["C1PHASE_DOXY", "DOWNWELLING_PAR"]},
    {"parameter_data_mode": {'DOXY': ['R', 'A']}},
    {"parameter_data_mode": {'DOXY': ['R', 'A'], 'BBP700': 'A'}},
]


def run_a_search(idx_maker, fetcher_args, search_point, xfail=False, reason="?"):
    """Create and run a search on a given index store

    Use xfail=True when a test with this is expected to fail
    """

    def core(fargs, apts):
        # log.debug(apts)
        if "nrows" in apts:
            nrows = apts["nrows"]
        else:
            nrows = None
        try:
            idx = idx_maker(**fargs)
            if "wmo" in apts:
                idx.query.wmo(apts["wmo"], nrows=nrows)
            if "cyc" in apts:
                idx.query.cyc(apts["cyc"], nrows=nrows)
            if "wmo_cyc" in apts:
                idx.query.wmo_cyc(apts["wmo_cyc"][0], apts["wmo_cyc"][1], nrows=nrows)
            if "date" in apts:
                idx.query.date(apts["date"], nrows=nrows)
            if "lon" in apts:
                idx.query.lon(apts["lon"], nrows=nrows)
            if "lat" in apts:
                idx.query.lat(apts["lat"], nrows=nrows)
            if "lon_lat" in apts:
                idx.query.lon_lat(apts["lon_lat"], nrows=nrows)
            if "box" in apts:
                idx.query.box(apts["box"], nrows=nrows)
            if "params" in apts:
                if np.any(
                        [key in idx.convention_title for key in ["Bio", "Synthetic"]]
                ):
                    if "logical" in apts:
                        logical = apts["logical"]
                    else:
                        logical = "and"
                    idx.query.params(apts["params"], nrows=nrows, logical=logical)
                else:
                    pytest.skip("For BGC index only")
            if "parameter_data_mode" in apts:
                if np.any(
                        [key in idx.convention_title for key in ["Bio", "Synthetic"]]
                ):
                    if "logical" in apts:
                        logical = apts["logical"]
                    else:
                        logical = "and"
                    idx.query.parameter_data_mode(apts["parameter_data_mode"], nrows=nrows, logical=logical)
                else:
                    pytest.skip("For BGC index only")
            if "profiler_type" in apts:
                idx.query.profiler_type(apts["profiler_type"], nrows=nrows)
            if "profiler_label" in apts:
                idx.query.profiler_label(apts["profiler_label"], nrows=nrows)
        except:
            if xfail:
                pytest.xfail(reason)
            else:
                raise
        return idx

    return core(fetcher_args, search_point)


def ftp_shortname(ftp):
    """Get a short name for scenarios IDs, given a FTP host"""
    if ftp == "MOCKFTP":
        return "ftp_mocked"
    elif "localhost" in ftp or "127.0.0.1" in ftp:
        return "http_mocked"
    else:
        return (lambda x: "file" if x == "" else x)(urlparse(ftp).scheme)


class IndexStore_test_proto:
    host, flist = argopy.tutorial.open_dataset("gdac")

    search_scenarios = [(h, ap) for h in VALID_HOSTS for ap in VALID_SEARCHES]
    search_scenarios = [
        (h, ap, n) for h in VALID_HOSTS for ap in VALID_SEARCHES for n in [None, 2]
    ]
    search_scenarios_ids = [
        "%s, %s, nrows=%s" % (ftp_shortname(fix[0]),
                              "%s[n=%i]" % (list(fix[1].keys())[0], len(fix[1][list(fix[1].keys())[0]])),
                              str(fix[2]))
        for fix in search_scenarios
    ]

    search_scenarios_bool = [
        (h, ap, n, b)
        for h in VALID_HOSTS
        for ap in VALID_SEARCHES_LOGICAL
        for n in [None, 2]
        for b in ["and", "or"]
    ]
    search_scenarios_bool_ids = [
        "%s, %s, nrows=%s, logical='%s'"
        % (ftp_shortname(fix[0]),
           "%s[n=%i]" % (list(fix[1].keys())[0], len(fix[1][list(fix[1].keys())[0]])),
           str(fix[2]),
           str(fix[3]))
        for fix in search_scenarios_bool
    ]

    #############
    # UTILITIES #
    #############

    def setup_class(self):
        """setup any state specific to the execution of the given class"""
        # Create the cache folder here, so that it's not the same for the pandas and pyarrow tests
        self.cachedir = create_temp_folder().folder

    def teardown_class(self):
        """Cleanup once we are finished."""

        def remove_test_dir():
            shutil.rmtree(self.cachedir)

        remove_test_dir()

    def _patch_ftp(self, ftp):
        return patch_ftp(ftp)

    def create_store(self, store_args, xfail=False, reason="?"):
        def core(fargs):
            try:
                idx = self.indexstore(**fargs)
            except Exception:
                if xfail:
                    pytest.xfail(reason)
                else:
                    raise
            return idx

        return core(store_args)

    def _setup_store(self, this_request, cached=False):
        """Helper method to set up options for an index store creation"""
        index_file = self.index_file
        convention = None
        if hasattr(this_request, "param"):
            if isinstance(this_request.param, tuple):
                host = this_request.param[0]
            else:
                host = this_request.param
        else:
            host = this_request["param"]["host"]
            index_file = this_request["param"]["index_file"]
            convention = this_request["param"]["convention"]
        N_RECORDS = (
            None if "tutorial" in host or "MOCK" in host else 100
        )  # Make sure we're not going to load the full index
        fetcher_args = {
            "host": self._patch_ftp(host),
            "index_file": index_file,
            "cache": False,
            "convention": convention,
        }
        if cached:
            fetcher_args = {
                **fetcher_args,
                **{"cache": True, "cachedir": self.cachedir},
            }
        return fetcher_args, N_RECORDS

    def new_idx(self, cache=False, cachedir=None, **kwargs):
        host = kwargs["host"] if "host" in kwargs else self.host
        index_file = kwargs["index_file"] if "index_file" in kwargs else self.index_file
        convention = kwargs["convention"] if "convention" in kwargs else None
        fetcher_args, N_RECORDS = self._setup_store(
            {
                "param": {
                    "host": host,
                    "index_file": index_file,
                    "convention": convention,
                }
            },
            cached=cache,
        )
        idx = self.create_store(fetcher_args)
        return idx

    @pytest.fixture
    def a_store(self, request):
        """Fixture to create an index store for a given host."""
        fetcher_args, N_RECORDS = self._setup_store(request)
        xfail, reason = False, ""
        if not HAS_S3FS and 's3' in fetcher_args['host']:
            xfail, reason = True, 's3fs not available'
        elif 's3' in fetcher_args['host']:
            xfail, reason = True, 's3 is experimental'
        yield self.create_store(fetcher_args, xfail=xfail, reason=reason).load(nrows=N_RECORDS)

    @pytest.fixture
    def a_search(self, request):
        """Fixture to create an Index fetcher for a given host and access point"""
        host = request.param[0]
        srch = request.param[1]
        nrows = request.param[2]
        srch["nrows"] = nrows
        if len(request.param) == 4:
            logical = request.param[3]
            srch["logical"] = logical
        # log.debug("a_search: %s, %s, %s" % (self.index_file, srch, xfail))

        xfail, reason = False, ""
        if not HAS_S3FS and 's3' in host:
            xfail, reason = True, 's3fs not available'
        elif 's3' in host:
            xfail, reason = True, 's3 is experimental'

        yield run_a_search(self.new_idx, {"host": host, "cache": True}, srch, xfail=xfail, reason=reason)

    def assert_index(self, this_idx, cacheable=False):
        assert hasattr(this_idx, "index")
        assert this_idx.shape[0] == this_idx.index.shape[0]
        assert this_idx.N_RECORDS == this_idx.index.shape[0]
        assert (
                is_list_of_strings(this_idx.uri_full_index)
                and len(this_idx.uri_full_index) == this_idx.N_RECORDS
        )
        if cacheable:
            assert is_list_of_strings(this_idx.cachepath("index"))

    def assert_search(self, this_idx, cacheable=False):
        assert hasattr(this_idx, "search")
        assert this_idx.N_MATCH == this_idx.search.shape[0]
        assert this_idx.N_FILES == this_idx.N_MATCH
        assert (
                is_list_of_strings(this_idx.uri) and len(this_idx.uri) == this_idx.N_MATCH
        )
        if cacheable:
            assert is_list_of_strings(this_idx.cachepath("search"))

    #########
    # TESTS #
    #########

    @pytest.mark.parametrize(
        "a_store",
        VALID_HOSTS,
        indirect=True,
        ids=["%s" % ftp_shortname(ftp) for ftp in VALID_HOSTS],
    )
    def test_hosts(self, mocked_httpserver, a_store):
        self.assert_index(
            a_store
        )

    @pytest.mark.parametrize(
        "ftp_host",
        ["invalid", "https://invalid_ftp", "ftp://invalid_ftp"],
        indirect=False,
    )
    def test_hosts_invalid(self, ftp_host):
        # Invalid servers:
        with pytest.raises(GdacPathError):
            self.indexstore(host=ftp_host)

    def test_index(self):
        def new_idx():
            return self.indexstore(
                host=self.host, index_file=self.index_file, cache=False
            )

        self.assert_index(new_idx().load())
        self.assert_index(new_idx().load(force=True))

        N = np.random.randint(1, 100 + 1)
        idx = new_idx().load(nrows=N)
        self.assert_index(idx)
        assert idx.index.shape[0] == N
        # Since no search was triggered:
        assert idx.N_FILES == idx.N_RECORDS

        with pytest.raises(OptionValueError):
            idx = self.indexstore(
                host=self.host, index_file="ar_greylist.txt", cache=False
            )

    @pytest.mark.parametrize(
        "a_search", search_scenarios, indirect=True, ids=search_scenarios_ids
    )
    def test_a_search(self, mocked_httpserver, a_search):
        self.assert_search(a_search, cacheable=False)

    @pytest.mark.parametrize(
        "a_search", search_scenarios_bool, indirect=True, ids=search_scenarios_bool_ids
    )
    def test_a_search_with_logical(self, mocked_httpserver, a_search):
        self.assert_search(a_search, cacheable=False)

    def test_to_dataframe_index(self):
        idx = self.new_idx()
        assert isinstance(idx.to_dataframe(), pd.core.frame.DataFrame)

        df = idx.to_dataframe(index=True)
        assert df.shape[0] == idx.N_RECORDS

        df = idx.to_dataframe()
        assert df.shape[0] == idx.N_RECORDS

        N = np.random.randint(1, 20 + 1)
        df = idx.to_dataframe(index=True, nrows=N)
        assert df.shape[0] == N

    def test_to_dataframe_search(self):
        idx = self.new_idx()
        wmo = [s["wmo"] for s in VALID_SEARCHES if "wmo" in s.keys()][0]
        idx = idx.query.wmo(wmo)

        df = idx.to_dataframe()
        assert isinstance(df, pd.core.frame.DataFrame)
        assert df.shape[0] == idx.N_MATCH

        N = np.random.randint(1, 10 + 1)
        df = idx.to_dataframe(nrows=N)
        assert df.shape[0] == N

    def test_caching_index(self):
        idx = self.new_idx(cache=True)
        idx.load(nrows=None if "tutorial" in idx.host or "MOCK" in idx.host else 100)
        self.assert_index(idx, cacheable=True)

    def test_caching_search(self):
        idx = self.new_idx(cache=True)
        wmo = [s["wmo"] for s in VALID_SEARCHES if "wmo" in s.keys()][0]
        idx.query.wmo(wmo)
        self.assert_search(idx, cacheable=True)

    @pytest.mark.parametrize(
        "index",
        [False, True],
        indirect=False,
        ids=["index=%s" % i for i in [False, True]],
    )
    def test_read_wmo(self, index):
        wmo = [s["wmo"] for s in VALID_SEARCHES if "wmo" in s.keys()][0]
        idx = self.new_idx().query.wmo(wmo)
        WMOs = idx.read_wmo(index=index)
        if index:
            assert all([is_wmo(w) for w in WMOs])
        else:
            assert len(WMOs) == len(wmo)

    @pytest.mark.parametrize(
        "index",
        [False, True],
        indirect=False,
        ids=["index=%s" % i for i in [False, True]],
    )
    def test_read_dac_wmo(self, index):
        wmo = [s["wmo"] for s in VALID_SEARCHES if "wmo" in s.keys()][0]
        idx = self.new_idx().query.wmo(wmo)
        DAC_WMOs = idx.read_dac_wmo(index=index)
        assert isinstance(DAC_WMOs, tuple)
        for row in DAC_WMOs:
            assert isinstance(row[0], str)
            assert is_wmo(row[1])

    @pytest.mark.parametrize(
        "index",
        [False, True],
        indirect=False,
        ids=["index=%s" % i for i in [False, True]],
    )
    def test_read_params(self, index):
        wmo = [s["wmo"] for s in VALID_SEARCHES if "wmo" in s.keys()][0]
        idx = self.new_idx().query.wmo(wmo)
        if self.network == "bgc":
            params = idx.read_params(index=index)
            assert is_list_of_strings(params)
        else:
            with pytest.raises(InvalidDatasetStructure):
                idx.read_params(index=index)

    @pytest.mark.parametrize(
        "index",
        [False, True],
        indirect=False,
        ids=["index=%s" % i for i in [False, True]],
    )
    def test_read_domain(self, index):
        wmo = [s["wmo"] for s in VALID_SEARCHES if "wmo" in s.keys()][0]
        idx = self.new_idx().query.wmo(wmo)
        domain = idx.read_domain(index=index)
        assert isinstance(domain, list)
        assert len(domain) == 6
        assert domain[1]>domain[0]
        assert domain[3]>domain[2]
        assert domain[5]>domain[4]

    @pytest.mark.parametrize(
        "index",
        [False, True],
        indirect=False,
        ids=["index=%s" % i for i in [False, True]],
    )
    def test_read_files(self, index):
        wmo = [s["wmo"] for s in VALID_SEARCHES if "wmo" in s.keys()][0]
        idx = self.new_idx().query.wmo(wmo)
        files = idx.read_files(index=index)
        assert isinstance(files, list)
        if index==True:
            assert len(files) == idx.N_RECORDS
        else:
            assert len(files) == idx.N_MATCH

    @pytest.mark.parametrize(
        "index",
        [False, True],
        indirect=False,
        ids=["index=%s" % i for i in [False, True]],
    )
    def test_records_per_wmo(self, index):
        wmo = [s["wmo"] for s in VALID_SEARCHES if "wmo" in s.keys()][0]
        idx = self.new_idx().query.wmo(wmo)
        C = idx.records_per_wmo(index=index)
        for w in C:
            assert str(C[w]).isdigit()

    def test_to_indexfile(self):
        # Create a store and make a simple float search:
        idx0 = self.new_idx()
        wmo = [s["wmo"] for s in VALID_SEARCHES if "wmo" in s.keys()][0]
        idx0 = idx0.query.wmo(wmo)

        # Then save this search as a new Argo index file:
        tf = tempfile.NamedTemporaryFile(delete=False)
        new_indexfile = idx0.to_indexfile(tf.name)

        # Finally try to load the new index file, like it was an official one:
        idx = self.new_idx(
            host=os.path.dirname(new_indexfile),
            index_file=os.path.basename(new_indexfile),
            convention=idx0.convention,
        )
        self.assert_index(idx.load())

        # Cleanup
        tf.close()

    @pytest.mark.parametrize(
        "chunksize",
        [None, 1],
        indirect=False,
        ids=["chunksize=%s" % i for i in [None, 1]],
    )
    def test_iterfloats(self, chunksize):
        # Create a store and make a simple float search:
        idx0 = self.new_idx()
        wmo = [s["wmo"] for s in VALID_SEARCHES if "wmo" in s.keys()][0]
        idx0 = idx0.query.wmo(wmo)
        if chunksize is None:
            assert all([isinstance(float, ArgoFloat) for float in idx0.iterfloats(chunksize=chunksize)])
        else:
            for chunk in idx0.iterfloats(chunksize=chunksize):
                assert len(chunk) == chunksize
                assert all([isinstance(float, ArgoFloat) for float in chunk])

    def test_dateline_search(self):
        idx = self.new_idx()
        with argopy.set_options(longitude_convention='360'):
            BOX = [170, 190., -90, 90, '2020-01', '2021-01']
            idx.query.lon(BOX)
            self.assert_search(idx)

            idx.query.lon_lat(BOX)
            self.assert_search(idx)

            idx.query.box(BOX)
            self.assert_search(idx)

############################
# TESTS FOR PANDAS BACKEND #
############################

@skip_pandas
@skip_CORE
class Test_IndexStore_pandas_CORE(IndexStore_test_proto):
    network = "core"
    indexstore = indexstore_pd
    index_file = "ar_index_global_prof.txt"


@skip_pandas
@skip_BGCs
class Test_IndexStore_pandas_BGC_synthetic(IndexStore_test_proto):
    network = "bgc"
    indexstore = indexstore_pd
    index_file = "argo_synthetic-profile_index.txt"


@skip_pandas
@skip_BGCb
class Test_IndexStore_pandas_BGC_bio(IndexStore_test_proto):
    network = "bgc"
    indexstore = indexstore_pd
    index_file = "argo_bio-profile_index.txt"


#############################
# TESTS FOR PYARROW BACKEND #
#############################

@skip_nopyarrow
@skip_pyarrow
@skip_CORE
class Test_IndexStore_pyarrow_CORE(IndexStore_test_proto):
    network = "core"
    from argopy.stores.index import indexstore_pa

    indexstore = indexstore_pa
    index_file = "ar_index_global_prof.txt"


@skip_nopyarrow
@skip_pyarrow
@skip_BGCs
class Test_IndexStore_pyarrow_BGC_bio(IndexStore_test_proto):
    network = "bgc"
    from argopy.stores.index import indexstore_pa

    indexstore = indexstore_pa
    index_file = "argo_bio-profile_index.txt"


@skip_nopyarrow
@skip_pyarrow
@skip_BGCb
class Test_IndexStore_pyarrow_BGC_synthetic(IndexStore_test_proto):
    network = "bgc"
    from argopy.stores.index import indexstore_pa

    indexstore = indexstore_pa
    index_file = "argo_synthetic-profile_index.txt"

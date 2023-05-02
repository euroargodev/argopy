import os

import pytest
import tempfile

import numpy as np
import pandas as pd
from fsspec.registry import known_implementations
import importlib
import shutil
import logging
from urllib.parse import urlparse

import argopy
from argopy.stores import (
    indexfilter_wmo,
    indexfilter_box,
    indexstore,
)
from argopy.errors import (
    FtpPathError,
    OptionValueError,
)
from argopy.utilities import (
    is_list_of_strings,
)
from argopy.stores.argo_index_pd import indexstore_pandas
from utils import requires_connection
from mocked_http import mocked_httpserver, mocked_server_address


log = logging.getLogger("argopy.tests.indexstores")

has_pyarrow = importlib.util.find_spec('pyarrow') is not None
skip_pyarrow = pytest.mark.skipif(not has_pyarrow, reason="Requires pyarrow")

skip_this = pytest.mark.skipif(0, reason="Skipped temporarily")
skip_for_debug = pytest.mark.skipif(False, reason="Taking too long !")


@skip_this
class Test_IndexFilter_WMO:
    kwargs = [
        {"WMO": 6901929},
        {"WMO": [6901929, 2901623]},
        {"CYC": 1},
        {"CYC": [1, 6]},
        {"WMO": 6901929, "CYC": 36},
        {"WMO": 6901929, "CYC": [5, 45]},
        {"WMO": [6901929, 2901623], "CYC": 2},
        {"WMO": [6901929, 2901623], "CYC": [2, 23]},
        {},
    ]

    def test_implementation(self):
        for kw in self.kwargs:
            filt = indexfilter_wmo(**kw)
            assert isinstance(filt, argopy.stores.argo_index.indexfilter_wmo)

    def test_filters_uri(self):
        for kw in self.kwargs:
            filt = indexfilter_wmo(**kw)
            assert isinstance(filt.uri, str)

    def test_filters_sha(self):
        for kw in self.kwargs:
            filt = indexfilter_wmo(**kw)
            assert isinstance(filt.sha, str) and len(filt.sha) == 64

    @requires_connection
    def test_filters_run(self):
        ftproot, flist = argopy.tutorial.open_dataset("gdac")
        index_file = os.path.sep.join([ftproot, "ar_index_global_prof.txt"])
        for kw in self.kwargs:
            filt = indexfilter_wmo(**kw)
            with open(index_file, "r") as f:
                results = filt.run(f)
                if results:
                    assert isinstance(results, str)
                else:
                    assert results is None


@skip_this
@requires_connection
class Test_Legacy_IndexStore:
    ftproot, flist = argopy.tutorial.open_dataset("gdac")
    index_file = os.path.sep.join([ftproot, "ar_index_global_prof.txt"])

    kwargs_wmo = [
        {"WMO": 6901929},
        {"WMO": [6901929, 2901623]},
        {"CYC": 1},
        {"CYC": [1, 6]},
        {"WMO": 6901929, "CYC": 36},
        {"WMO": 6901929, "CYC": [5, 45]},
        {"WMO": [6901929, 2901623], "CYC": 2},
        {"WMO": [6901929, 2901623], "CYC": [2, 23]},
        {},
    ]

    kwargs_box = [
        {"BOX": [-60, -40, 40.0, 60.0]},
        {"BOX": [-60, -40, 40.0, 60.0, "2007-08-01", "2007-09-01"]},
    ]

    def test_implementation(self):
        assert isinstance(indexstore(), argopy.stores.argo_index.indexstore)
        assert isinstance(indexstore(cache=True), argopy.stores.argo_index.indexstore)
        assert isinstance(
            indexstore(cache=True, cachedir="."), argopy.stores.argo_index.indexstore
        )
        assert isinstance(
            indexstore(index_file="toto.txt"), argopy.stores.argo_index.indexstore
        )

    def test_search_wmo(self):
        for kw in self.kwargs_wmo:
            df = indexstore(cache=False, index_file=self.index_file).read_csv(
                indexfilter_wmo(**kw)
            )
            assert isinstance(df, pd.core.frame.DataFrame)

    def test_search_box(self):
        for kw in self.kwargs_box:
            df = indexstore(cache=False, index_file=self.index_file).read_csv(
                indexfilter_box(**kw)
            )
            assert isinstance(df, pd.core.frame.DataFrame)


"""
List gdac hosts to be tested. 
Since the fetcher is compatible with host from local, http or ftp protocols, we
try to test them all:
"""
VALID_HOSTS = [argopy.tutorial.open_dataset("gdac")[0],
             #'https://data-argo.ifremer.fr',
             mocked_server_address,
             # 'ftp://ftp.ifremer.fr/ifremer/argo',
             # 'ftp://usgodae.org/pub/outgoing/argo',  # ok, but takes too long to respond, slow down CI
             'MOCKFTP',  # keyword to use a fake/mocked ftp server (running on localhost)
             ]

"""
List index searches to be tested.
"""
VALID_SEARCHES = [
    # {"wmo": [13857]},
    {"wmo": [3902131]},  # BGC
    # {"wmo": [6901929, 2901623]},
    {"cyc": [5, 45]},
    # {"wmo_cyc": [13857, 2]},
    {"wmo_cyc": [3902131, 2]},  # BGC
    {"tim": [-60, -40, 40.0, 60.0, "2007-08-01", "2007-09-01"]},
    {"lat_lon": [-60, -40, 40.0, 60.0, "2007-08-01", "2007-09-01"]},
    {"lat_lon_tim": [-60, -40, 40.0, 60.0, "2007-08-01", "2007-09-01"]},
    {"params": ['C1PHASE_DOXY', 'DOWNWELLING_PAR']},
    ]

def run_a_search(idx_maker, fetcher_args, search_point, xfail=False, reason='?'):
    """ Create and run a search on a given index store

        Use xfail=True when a test with this is expected to fail
    """
    def core(fargs, apts):
        try:
            idx = idx_maker(**fargs)
            if "wmo" in apts:
                idx.search_wmo(apts['wmo'])
            if "cyc" in apts:
                idx.search_cyc(apts['cyc'])
            if "wmo_cyc" in apts:
                idx.search_wmo_cyc(apts['wmo_cyc'][0], apts['wmo_cyc'][1])
            if "tim" in apts:
                idx.search_tim(apts['tim'])
            if "lat_lon" in apts:
                idx.search_lat_lon(apts['lat_lon'])
            if "lat_lon_tim" in apts:
                idx.search_lat_lon_tim(apts['lat_lon_tim'])
            if "params" in apts:
                idx.search_params(apts['params'])
        except:
            if xfail:
                pytest.xfail(reason)
            else:
                raise
        return idx

    return core(fetcher_args, search_point)


def ftp_shortname(ftp):
    """Get a short name for scenarios IDs, given a FTP host"""
    if ftp == 'MOCKFTP':
        return 'ftp_mocked'
    elif 'localhost' in ftp or '127.0.0.1' in ftp:
        return 'http_mocked'
    else:
        return (lambda x: 'file' if x == "" else x)(urlparse(ftp).scheme)


class IndexStore_test_proto:
    host, flist = argopy.tutorial.open_dataset("gdac")

    search_scenarios = [(h, ap) for h in VALID_HOSTS for ap in VALID_SEARCHES]
    search_scenarios_ids = [
        "%s, %s" % (ftp_shortname(fix[0]), list(fix[1].keys())[0]) for fix
        in
        search_scenarios]

    #############
    # UTILITIES #
    #############

    def setup_class(self):
        """setup any state specific to the execution of the given class"""
        # Create the cache folder here, so that it's not the same for the pandas and pyarrow tests
        self.cachedir = tempfile.mkdtemp()

    def teardown_class(self):
        """Cleanup once we are finished."""
        def remove_test_dir():
            shutil.rmtree(self.cachedir)
        remove_test_dir()

    def _patch_ftp(self, ftp):
        """Patch Mocked FTP server keyword"""
        if ftp == 'MOCKFTP':
            return pytest.MOCKFTP  # this was set in conftest.py
        else:
            return ftp

    def create_store(self, store_args, xfail=False):
        def core(fargs):
            try:
                idx = self.indexstore(**fargs)
            except Exception:
                raise
            return idx
        return core(store_args)

    def _setup_store(self, this_request, cached=False):
        """Helper method to set up options for an index store creation"""
        index_file = self.index_file
        convention = None
        if hasattr(this_request, 'param'):
            if isinstance(this_request.param, tuple):
                host = this_request.param[0]
            else:
                host = this_request.param
        else:
            host = this_request['param']['host']
            index_file = this_request['param']['index_file']
            convention = this_request['param']['convention']
        N_RECORDS = None if 'tutorial' in host or 'MOCK' in host else 100  # Make sure we're not going to load the full index
        fetcher_args = {"host": self._patch_ftp(host), "index_file": index_file, "cache": False, "convention": convention}
        if cached:
            fetcher_args = {**fetcher_args, **{"cache": True, "cachedir": self.cachedir}}
        return fetcher_args, N_RECORDS

    def new_idx(self, cache=False, cachedir=None, **kwargs):
        host = kwargs['host'] if 'host' in kwargs else self.host
        index_file = kwargs['index_file'] if 'index_file' in kwargs else self.index_file
        convention = kwargs['convention'] if 'convention' in kwargs else None
        fetcher_args, N_RECORDS = self._setup_store({'param': {'host': host, 'index_file': index_file, 'convention': convention}}, cached=cache)
        idx = self.create_store(fetcher_args).load(nrows=N_RECORDS)
        return idx

    @pytest.fixture
    def a_store(self, request):
        """Fixture to create an index store for a given host."""
        fetcher_args, N_RECORDS = self._setup_store(request)
        # yield self.indexstore(**fetcher_args).load(nrows=N_RECORDS)
        yield self.create_store(fetcher_args).load(nrows=N_RECORDS)

    @pytest.fixture
    def a_search(self, request):
        """ Fixture to create a FTP fetcher for a given host and access point """
        host = request.param[0]
        srch = request.param[1]
        xfail = self.index_file == 'ar_index_global_prof.txt' and 'params' in srch
        reason = "'params' search only available to BGC profile index" if xfail else '?'
        # log.debug("a_search: %s, %s, %s" % (self.index_file, srch, xfail))
        yield run_a_search(self.new_idx, {'host': host, 'cache': True}, srch, xfail=xfail, reason=reason)

    def assert_index(self, this_idx, cacheable=False):
        assert hasattr(this_idx, 'index')
        assert this_idx.shape[0] == this_idx.index.shape[0]
        assert this_idx.N_RECORDS == this_idx.index.shape[0]
        assert is_list_of_strings(this_idx.uri_full_index) and len(this_idx.uri_full_index) == this_idx.N_RECORDS
        if cacheable:
            assert is_list_of_strings(this_idx.cachepath('index'))

    def assert_search(self, this_idx, cacheable=False):
        assert hasattr(this_idx, 'search')
        assert this_idx.N_MATCH == this_idx.search.shape[0]
        assert this_idx.N_FILES == this_idx.N_MATCH
        assert is_list_of_strings(this_idx.uri) and len(this_idx.uri) == this_idx.N_MATCH
        if cacheable:
            assert is_list_of_strings(this_idx.cachepath('search'))

    #########
    # TESTS #
    #########

    @pytest.mark.parametrize("a_store", VALID_HOSTS,
                             indirect=True,
                             ids=["%s" % ftp_shortname(ftp) for ftp in VALID_HOSTS])
    def test_hosts(self, mocked_httpserver, a_store):
        self.assert_index(a_store) # assert (this_store.N_RECORDS >= 1)  # Make sure we loaded the index file content

    @pytest.mark.parametrize("ftp_host", ['invalid', 'https://invalid_ftp', 'ftp://invalid_ftp'], indirect=False)
    def test_hosts_invalid(self, ftp_host):
        # Invalid servers:
        with pytest.raises(FtpPathError):
            self.indexstore(host=ftp_host)

    def test_index(self):
        def new_idx():
            return self.indexstore(host=self.host, index_file=self.index_file, cache=False)
        self.assert_index(new_idx().load())
        self.assert_index(new_idx().load(force=True))

        N = np.random.randint(1, 100+1)
        idx = new_idx().load(nrows=N)
        self.assert_index(idx)
        assert idx.index.shape[0] == N
        # Since no search was triggered:
        assert idx.N_FILES == idx.N_RECORDS

        with pytest.raises(OptionValueError):
            idx = self.indexstore(host=self.host, index_file="ar_greylist.txt", cache=False)

    @pytest.mark.parametrize("a_search", search_scenarios, indirect=True, ids=search_scenarios_ids)
    def test_search(self, mocked_httpserver, a_search):
        self.assert_search(a_search, cacheable=False)

    def test_to_dataframe_index(self):
        idx = self.new_idx()
        assert isinstance(idx.to_dataframe(), pd.core.frame.DataFrame)

        df = idx.to_dataframe(index=True)
        assert df.shape[0] == idx.N_RECORDS

        df = idx.to_dataframe()
        assert df.shape[0] == idx.N_RECORDS

        N = np.random.randint(1, 20+1)
        df = idx.to_dataframe(index=True, nrows=N)
        assert df.shape[0] == N

    def test_to_dataframe_search(self):
        idx = self.new_idx()
        wmo = [s['wmo'] for s in VALID_SEARCHES if 'wmo' in s.keys()][0]
        idx = idx.search_wmo(wmo)

        df = idx.to_dataframe()
        assert isinstance(df, pd.core.frame.DataFrame)
        assert df.shape[0] == idx.N_MATCH

        N = np.random.randint(1,10+1)
        df = idx.to_dataframe(nrows=N)
        assert df.shape[0] == N

    def test_caching_index(self):
        idx = self.new_idx(cache=True)
        self.assert_index(idx, cacheable=True)

    def test_caching_search(self):
        idx = self.new_idx(cache=True)
        wmo = [s['wmo'] for s in VALID_SEARCHES if 'wmo' in s.keys()][0]
        idx.search_wmo(wmo)
        self.assert_search(idx, cacheable=True)

    def test_read_wmo(self):
        wmo = [s['wmo'] for s in VALID_SEARCHES if 'wmo' in s.keys()][0]
        idx = self.new_idx().search_wmo(wmo)
        assert len(idx.read_wmo()) == len(wmo)

    def test_records_per_wmo(self):
        wmo = [s['wmo'] for s in VALID_SEARCHES if 'wmo' in s.keys()][0]
        idx = self.new_idx().search_wmo(wmo)
        C = idx.records_per_wmo()
        for w in C:
            assert str(C[w]).isdigit()

    def test_to_indexfile(self):
        # Create a store and make a simple float search:
        idx0 = self.new_idx()
        wmo = [s['wmo'] for s in VALID_SEARCHES if 'wmo' in s.keys()][0]
        idx0 = idx0.search_wmo(wmo)

        # Then save this search as a new Argo index file:
        tf = tempfile.NamedTemporaryFile(delete=False)
        new_indexfile = idx0.to_indexfile(tf.name)

        # Finally try to load the new index file, like it was an official one:
        idx = self.new_idx(host=os.path.dirname(new_indexfile),
                           index_file=os.path.basename(new_indexfile),
                           convention=idx0.convention)
        self.assert_index(idx.load())

        # Cleanup
        tf.close()


@skip_this
class Test_IndexStore_pandas_CORE(IndexStore_test_proto):
    indexstore = indexstore_pandas
    index_file = "ar_index_global_prof.txt"


@skip_this
@skip_pyarrow
class Test_IndexStore_pyarrow_CORE(IndexStore_test_proto):
    from argopy.stores.argo_index_pa import indexstore_pyarrow
    indexstore = indexstore_pyarrow
    index_file = "ar_index_global_prof.txt"


@skip_this
class Test_IndexStore_pandas_BGC(IndexStore_test_proto):
    indexstore = indexstore_pandas
    index_file = "argo_bio-profile_index.txt"


@skip_this
@skip_pyarrow
class Test_IndexStore_pyarrow_BGC(IndexStore_test_proto):
    from argopy.stores.argo_index_pa import indexstore_pyarrow
    indexstore = indexstore_pyarrow
    index_file = "argo_bio-profile_index.txt"

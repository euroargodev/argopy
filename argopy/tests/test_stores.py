import os
import pytest
import tempfile

import numpy as np
import xarray as xr
import pandas as pd
import fsspec
from fsspec.registry import known_implementations
import aiohttp
import importlib

import argopy
from argopy.stores import (
    filestore,
    httpstore,
    indexfilter_wmo,
    indexfilter_box,
    indexstore,
)
from argopy.stores.filesystems import new_fs
from argopy.options import OPTIONS
from argopy.errors import FileSystemHasNoCache, CacheFileNotFound, InvalidDatasetStructure, FtpPathError
from . import requires_connection, requires_connected_argovis, safe_to_server_errors
from argopy.utilities import (
    is_list_of_datasets,
    is_list_of_dicts,
    modified_environ,
    is_list_of_strings,
    check_wmo,
)

has_pyarrow = (spec := importlib.util.find_spec('pyarrow')) is not None
if has_pyarrow:
    from argopy.stores.argo_index_pa import indexstore_pyarrow as new_indexstore
else:
    from argopy.stores.argo_index_pa import indexstore_pandas as new_indexstore


skip_this = pytest.mark.skipif(True, reason="Skipped temporarily")


@skip_this
class Test_new_fs:
    id_implementation = lambda y, x: [k for k, v in known_implementations.items()  # noqa: E731
                                       if x.__class__.__name__ == v['class'].split('.')[-1]]
    is_initialised = lambda y, x: ((x is None) or (x == []))  # noqa: E731

    def test_default(self):
        fs, cache_registry = new_fs()
        assert self.id_implementation(fs) is not None
        assert self.is_initialised(cache_registry)

    def test_cache_type(self):
        fs, cache_registry = new_fs(cache=True)
        assert self.id_implementation(fs) == ['filecache']


@skip_this
@requires_connection
class Test_FileStore:
    ftproot = argopy.tutorial.open_dataset("localftp")[0]
    csvfile = os.path.sep.join([ftproot, "ar_index_global_prof.txt"])

    def test_creation(self):
        fs = filestore(cache=False)
        assert isinstance(fs.fs, fsspec.implementations.local.LocalFileSystem)

    def test_nocache(self):
        fs = filestore(cache=False)
        with pytest.raises(FileSystemHasNoCache):
            fs.cachepath("dummy_uri")

    def test_cache(self):
        fs = filestore(cache=True)
        assert isinstance(fs.fs, fsspec.implementations.cached.WholeFileCacheFileSystem)

    def test_nocachefile(self):
        fs = filestore(cache=True)
        with pytest.raises(CacheFileNotFound):
            fs.cachepath("dummy_uri")

    def test_glob(self):
        fs = filestore()
        assert isinstance(fs.glob(os.path.sep.join([self.ftproot, "dac/*"])), list)

    def test_open_dataset(self):
        ncfile = os.path.sep.join([self.ftproot, "dac/aoml/5900446/5900446_prof.nc"])
        fs = filestore()
        assert isinstance(fs.open_dataset(ncfile), xr.Dataset)

    def test_open_mfdataset(self):
        fs = filestore()
        ncfiles = fs.glob(
            os.path.sep.join([self.ftproot, "dac/aoml/5900446/profiles/*_1*.nc"])
        )[0:2]
        for method in ["seq", "thread", "process"]:
            for progress in [True, False]:
                assert isinstance(
                    fs.open_mfdataset(ncfiles, method=method, progress=progress),
                    xr.Dataset,
                )
                assert is_list_of_datasets(
                    fs.open_mfdataset(
                        ncfiles, method=method, progress=progress, concat=False
                    )
                )

    def test_read_csv(self):
        fs = filestore()
        assert isinstance(
            fs.read_csv(self.csvfile, skiprows=8, header=0), pd.core.frame.DataFrame
        )

    def test_cachefile(self):
        with tempfile.TemporaryDirectory() as cachedir:
            fs = filestore(cache=True, cachedir=cachedir)
            fs.read_csv(self.csvfile, skiprows=8, header=0)
            assert isinstance(fs.cachepath(self.csvfile), str)

    def test_clear_cache(self):
        with tempfile.TemporaryDirectory() as cachedir:
            # Create dummy data to read and cache:
            uri = os.path.abspath("dummy_fileA.txt")
            with open(uri, "w") as fp:
                fp.write("Hello world!")
            # Create store:
            fs = filestore(cache=True, cachedir=cachedir)
            # Then we read some dummy data from the dummy file to trigger caching
            with fs.open(uri, "r") as fp:
                fp.read()
            assert isinstance(fs.cachepath(uri), str)
            assert os.path.isfile(fs.cachepath(uri))
            # Now, we can clear the cache:
            fs.clear_cache()
            # And verify it does not exist anymore:
            with pytest.raises(CacheFileNotFound):
                fs.cachepath(uri)
            os.remove(uri)  # Delete dummy file


@skip_this
@requires_connection
class Test_HttpStore:
    def test_creation(self):
        fs = httpstore(cache=False)
        assert isinstance(fs.fs, fsspec.implementations.http.HTTPFileSystem)

    def test_trust_env(self):
        with modified_environ(HTTP_PROXY='http://dummy_proxy'):
            with argopy.set_options(trust_env=True):
                fs = httpstore(cache=False)
                with pytest.raises(aiohttp.client_exceptions.ClientConnectorError):
                    uri = "http://github.com/euroargodev/argopy-data/raw/master/ftp/dac/csiro/5900865/5900865_prof.nc"
                    fs.open_dataset(uri)

    def test_nocache(self):
        fs = httpstore(cache=False)
        with pytest.raises(FileSystemHasNoCache):
            fs.cachepath("dummy_uri")

    def test_cachable(self):
        fs = httpstore(cache=True)
        assert isinstance(fs.fs, fsspec.implementations.cached.WholeFileCacheFileSystem)

    def test_nocachefile(self):
        fs = httpstore(cache=True)
        with pytest.raises(CacheFileNotFound):
            fs.cachepath("dummy_uri")

    @safe_to_server_errors
    def test_cache_a_file(self):
        uri = "https://github.com/euroargodev/argopy-data/raw/master/ftp/ar_index_global_prof.txt"
        with tempfile.TemporaryDirectory() as cachedir:
            fs = httpstore(cache=True, cachedir=cachedir, timeout=OPTIONS['api_timeout'])
            fs.read_csv(uri, skiprows=8, header=0)
            assert isinstance(fs.cachepath(uri), str)

    @safe_to_server_errors
    def test_clear_cache(self):
        uri = "https://github.com/euroargodev/argopy-data/raw/master/ftp/ar_index_global_prof.txt"
        with tempfile.TemporaryDirectory() as cachedir:
            fs = httpstore(cache=True, cachedir=cachedir, timeout=OPTIONS['api_timeout'])
            fs.read_csv(uri, skiprows=8, header=0)
            assert isinstance(fs.cachepath(uri), str)
            assert os.path.isfile(fs.cachepath(uri))
            fs.clear_cache()
            with pytest.raises(CacheFileNotFound):
                fs.cachepath(uri)

    @safe_to_server_errors
    def test_open_dataset(self):
        uri = "https://github.com/euroargodev/argopy-data/raw/master/ftp/dac/csiro/5900865/5900865_prof.nc"
        fs = httpstore(timeout=OPTIONS['api_timeout'])
        assert isinstance(fs.open_dataset(uri), xr.Dataset)

    @safe_to_server_errors
    def test_open_mfdataset(self):
        fs = httpstore(timeout=OPTIONS['api_timeout'])
        uri = [
            "https://github.com/euroargodev/argopy-data/raw/master/ftp/dac/csiro/5900865/profiles/D5900865_00%i.nc"
            % i
            for i in [1, 2]
        ]
        for method in ["seq", "thread"]:
            for progress in [True, False]:
                assert isinstance(
                    fs.open_mfdataset(uri, method=method, progress=progress), xr.Dataset
                )
                assert is_list_of_datasets(
                    fs.open_mfdataset(
                        uri, method=method, progress=progress, concat=False
                    )
                )

    @requires_connected_argovis
    @safe_to_server_errors
    def test_open_json(self):
        uri = "https://argovis.colorado.edu/catalog/mprofiles/?ids=['6902746_12']"
        fs = httpstore(timeout=OPTIONS['api_timeout'])
        assert is_list_of_dicts(fs.open_json(uri))

    @requires_connected_argovis
    @safe_to_server_errors
    def test_open_mfjson(self):
        fs = httpstore(timeout=OPTIONS['api_timeout'])
        uri = [
            "https://argovis.colorado.edu/catalog/mprofiles/?ids=['6902746_%i']" % i
            for i in [12, 13]
        ]
        for method in ["seq", "thread"]:
            for progress in [True, False]:
                lst = fs.open_mfjson(uri, method=method, progress=progress)
                assert all(is_list_of_dicts(x) for x in lst)

    @safe_to_server_errors
    def test_read_csv(self):
        uri = "https://github.com/euroargodev/argopy-data/raw/master/ftp/ar_index_global_prof.txt"
        fs = httpstore(timeout=OPTIONS['api_timeout'])
        assert isinstance(
            fs.read_csv(uri, skiprows=8, header=0), pd.core.frame.DataFrame
        )


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

    def test_creation(self):
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
        ftproot, flist = argopy.tutorial.open_dataset("localftp")
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
    ftproot, flist = argopy.tutorial.open_dataset("localftp")
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

    def test_creation(self):
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


class Test_IndexStore:
    host, flist = argopy.tutorial.open_dataset("localftp")
    index_file = "ar_index_global_prof.txt"

    kwargs_wmo = [
        {"WMO": 6901929},
        {"WMO": [6901929, 2901623]},
        # {"CYC": 1},
        # {"CYC": [1, 6]},
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

    def test_creation(self):
        assert isinstance(new_indexstore(), argopy.stores.argo_index_pa.ArgoIndexStoreProto)
        assert isinstance(new_indexstore(cache=True), argopy.stores.argo_index_pa.ArgoIndexStoreProto)
        assert isinstance(
            new_indexstore(cache=True, cachedir="."), argopy.stores.argo_index_pa.ArgoIndexStoreProto
        )
        assert isinstance(
            new_indexstore(host=self.host, index_file=self.index_file), argopy.stores.argo_index_pa.ArgoIndexStoreProto
        )
        with pytest.raises(FtpPathError):
            new_indexstore(host=self.host, index_file="dummy_index.txt")
        with pytest.raises(FtpPathError):
            new_indexstore(host="ftp://ftp.dummy.fr/argo")
        with pytest.raises(FtpPathError):
            new_indexstore(host="s3://dummy_cloud_store")

    def test_index(self):
        def new_idx():
            return new_indexstore(host=self.host, index_file=self.index_file, cache=False)
        assert hasattr(new_idx().load(), 'index')
        assert hasattr(new_idx().load(force=True), 'index')

        N = np.random.randint(1,100+1)
        idx = new_idx().load(nrows=N)
        assert hasattr(idx, 'index') and idx.index.shape[0] == N
        assert idx.shape[0] == idx.index.shape[0]
        assert idx.N_RECORDS == idx.index.shape[0]
        assert idx.N_FILES == idx.N_RECORDS
        assert is_list_of_strings(idx.uri_full_index) and len(idx.uri_full_index) == idx.N_RECORDS

        with pytest.raises(InvalidDatasetStructure):
            idx = new_indexstore(host=self.host, index_file="ar_greylist.txt", cache=False)
            idx.load()

    def test_index_to_dataframe(self):
        idx = new_indexstore(host=self.host, index_file=self.index_file, cache=False)
        idx.load()
        assert isinstance(idx.to_dataframe(), pd.core.frame.DataFrame)

        df = idx.to_dataframe(index=True)
        assert df.shape[0] == idx.N_RECORDS

        df = idx.to_dataframe()
        assert df.shape[0] == idx.N_RECORDS

        N = np.random.randint(1,100+1)
        df = idx.to_dataframe(index=True, nrows=N)
        assert df.shape[0] == N

    def test_search_wmo(self):
        for kw in self.kwargs_wmo:
            def new_idx(kw):
                idx = new_indexstore(host=self.host, index_file=self.index_file, cache=False).load()
                return idx.search_wmo(check_wmo(kw))

            idx = new_idx(kw['WMO'])
            assert hasattr(idx, 'search')
            assert idx.N_MATCH == idx.search.shape[0]
            assert idx.N_FILES == idx.N_MATCH
            assert is_list_of_strings(idx.uri) and len(idx.uri) == idx.N_MATCH

            # assert isinstance(df, pd.core.frame.DataFrame)

    # def test_search_box(self):
    #     for kw in self.kwargs_box:
    #         df = indexstore(cache=False, index_file=self.index_file).read_csv(
    #             indexfilter_box(**kw)
    #         )
    #         assert isinstance(df, pd.core.frame.DataFrame)

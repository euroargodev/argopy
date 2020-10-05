import os
import pytest
import tempfile

import xarray as xr
import pandas as pd
import fsspec
import argopy
from argopy.stores import (
    filestore,
    httpstore,
    indexfilter_wmo,
    indexfilter_box,
    indexstore,
)
from argopy.errors import FileSystemHasNoCache, CacheFileNotFound
from . import requires_connection
from argopy.utilities import is_list_of_datasets, is_list_of_dicts


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
            # Now, we can clear the cache:
            fs.clear_cache()
            # And verify it does not exist anymore:
            with pytest.raises(CacheFileNotFound):
                fs.cachepath(uri)
            os.remove(uri)


@requires_connection
class Test_HttpStore:
    def test_creation(self):
        fs = httpstore(cache=False)
        assert isinstance(fs.fs, fsspec.implementations.http.HTTPFileSystem)

    def test_nocache(self):
        fs = httpstore(cache=False)
        with pytest.raises(FileSystemHasNoCache):
            fs.cachepath("dummy_uri")

    def test_cache(self):
        fs = httpstore(cache=True)
        assert isinstance(fs.fs, fsspec.implementations.cached.WholeFileCacheFileSystem)

    def test_nocachefile(self):
        fs = httpstore(cache=True)
        with pytest.raises(CacheFileNotFound):
            fs.cachepath("dummy_uri")

    def test_open_dataset(self):
        uri = "https://github.com/euroargodev/argopy-data/raw/master/ftp/dac/csiro/5900865/5900865_prof.nc"
        fs = httpstore()
        assert isinstance(fs.open_dataset(uri), xr.Dataset)

    def test_open_mfdataset(self):
        fs = httpstore()
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

    def test_open_json(self):
        uri = "https://argovis.colorado.edu/catalog/mprofiles/?ids=['6902746_12']"
        fs = httpstore()
        assert is_list_of_dicts(fs.open_json(uri))

    def test_open_mfjson(self):
        fs = httpstore()
        uri = [
            "https://argovis.colorado.edu/catalog/mprofiles/?ids=['6902746_%i']" % i
            for i in [12, 13]
        ]
        for method in ["seq", "thread"]:
            for progress in [True, False]:
                lst = fs.open_mfjson(uri, method=method, progress=progress)
                assert all(is_list_of_dicts(x) for x in lst)

    def test_read_csv(self):
        uri = "https://github.com/euroargodev/argopy-data/raw/master/ftp/ar_index_global_prof.txt"
        fs = httpstore()
        assert isinstance(
            fs.read_csv(uri, skiprows=8, header=0), pd.core.frame.DataFrame
        )

    def test_cachefile(self):
        uri = "https://github.com/euroargodev/argopy-data/raw/master/ftp/ar_index_global_prof.txt"
        with tempfile.TemporaryDirectory() as cachedir:
            fs = httpstore(cache=True, cachedir=cachedir)
            fs.read_csv(uri, skiprows=8, header=0)
            assert isinstance(fs.cachepath(uri), str)


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
            assert isinstance(filt.uri(), str)

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


@requires_connection
class Test_IndexStore:
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

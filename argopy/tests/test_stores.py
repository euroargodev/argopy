import os
import shutil
import pytest
import unittest
from unittest import TestCase

import xarray as xr
import pandas as pd
import fsspec
import argopy
from argopy.stores import filestore, httpstore, indexfilter_wmo, indexstore
from argopy.errors import FileSystemHasNoCache, CacheFileNotFound, ErddapServerError
from argopy.fetchers import ArgoDataFetcher
from argopy.utilities import isconnected
CONNECTED = isconnected()


class FileStore(TestCase):
    def test_creation(self):
        fs = filestore(cache=0)
        assert isinstance(fs.fs, fsspec.implementations.local.LocalFileSystem)

    def test_nocache(self):
        fs = filestore(cache=0)
        with pytest.raises(FileSystemHasNoCache):
            fs.cachepath("dummy_uri")

    def test_cache(self):
        fs = filestore(cache=1)
        assert isinstance(fs.fs, fsspec.implementations.cached.WholeFileCacheFileSystem)

    def test_nocachefile(self):
        fs = filestore(cache=1)
        with pytest.raises(CacheFileNotFound):
            fs.cachepath("dummy_uri")

    def test_open_dataset(self):
        ftproot, flist = argopy.tutorial.open_dataset('localftp')
        ncfile = os.path.sep.join([ftproot, "dac/aoml/5900446/5900446_prof.nc"])
        fs = filestore()
        assert isinstance(fs.open_dataset(ncfile), xr.Dataset)

    def test_open_dataframe(self):
        ftproot, flist = argopy.tutorial.open_dataset('localftp')
        csvfile = os.path.sep.join([ftproot, "ar_index_global_prof.txt"])
        fs = filestore()
        assert isinstance(fs.open_dataframe(csvfile, skiprows=8, header=0), pd.core.frame.DataFrame)

    def test_cachefile(self):
        ftproot, flist = argopy.tutorial.open_dataset('localftp')
        csvfile = os.path.sep.join([ftproot, "ar_index_global_prof.txt"])
        testcachedir = os.path.expanduser(os.path.join("~", ".argopytest_tmp"))
        try:
            fs = filestore(cache=1, cachedir=testcachedir)
            fs.open_dataframe(csvfile, skiprows=8, header=0)
            assert isinstance(fs.cachepath(csvfile), str)
            shutil.rmtree(testcachedir)
        except Exception:
            shutil.rmtree(testcachedir)
            raise


class HttpStore(TestCase):
    def test_creation(self):
        fs = httpstore(cache=0)
        assert isinstance(fs.fs, fsspec.implementations.http.HTTPFileSystem)

    def test_nocache(self):
        fs = httpstore(cache=0)
        with pytest.raises(FileSystemHasNoCache):
            fs.cachepath("dummy_uri")

    def test_cache(self):
        fs = httpstore(cache=1)
        assert isinstance(fs.fs, fsspec.implementations.cached.WholeFileCacheFileSystem)

    def test_nocachefile(self):
        fs = httpstore(cache=1)
        with pytest.raises(CacheFileNotFound):
            fs.cachepath("dummy_uri")

    @unittest.skipUnless(CONNECTED, "httpstore requires an internet connection to open online resources")
    def test_open_dataset(self):
        uri = 'https://github.com/euroargodev/argopy-data/raw/master/ftp/dac/csiro/5900865/5900865_prof.nc'
        fs = httpstore()
        assert isinstance(fs.open_dataset(uri), xr.Dataset)

        # uri = 'https://github.com/dummy.nc'
        # fs = httpstore()
        # with pytest.raises(ErddapServerError):
        #     fs.open_dataset(uri)

    @unittest.skipUnless(CONNECTED, "httpstore requires an internet connection to open online resources")
    def test_open_dataframe(self):
        uri = 'https://github.com/euroargodev/argopy-data/raw/master/ftp/ar_index_global_prof.txt'
        fs = httpstore()
        assert isinstance(fs.open_dataframe(uri, skiprows=8, header=0), pd.core.frame.DataFrame)

        # uri = 'https://github.com/dummy.txt'
        # fs = httpstore()
        # with pytest.raises(ErddapServerError):
        #     fs.open_dataframe(uri)

    @unittest.skipUnless(CONNECTED, "httpstore requires an internet connection to open online resources")
    def test_cachefile(self):
        uri = 'https://github.com/euroargodev/argopy-data/raw/master/ftp/ar_index_global_prof.txt'
        testcachedir = os.path.expanduser(os.path.join("~", ".argopytest_tmp"))
        try:
            fs = httpstore(cache=1, cachedir=testcachedir)
            fs.open_dataframe(uri, skiprows=8, header=0)
            assert isinstance(fs.cachepath(uri), str)
            shutil.rmtree(testcachedir)
        except Exception:
            shutil.rmtree(testcachedir)
            raise


class IndexFilter_WMO(TestCase):
    kwargs = [{'WMO': 1900204},
              {'WMO': [1900204, 1900243]},
              {'CYC': 1},
              {'CYC': [1, 6]},
              {'WMO': 1900204, 'CYC': 36},
              {'WMO': 1900243, 'CYC': [5, 45]},
              {'WMO': [1900482, 2900738], 'CYC': 2},
              {'WMO': [1900482, 2900738], 'CYC': [2, 6]},
              {}]

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

    def test_filters_run(self):
        ftproot, flist = argopy.tutorial.open_dataset('localftp')
        index_file = os.path.sep.join([ftproot, "ar_index_global_prof.txt"])
        for kw in self.kwargs:
            filt = indexfilter_wmo(**kw)
            with open(index_file, "r") as f:
                results = filt.run(f)
                if results:
                    assert isinstance(results, str)
                else:
                    assert results is None


class IndexStore(TestCase):
    ftproot, flist = argopy.tutorial.open_dataset('localftp')
    index_file = os.path.sep.join([ftproot, "ar_index_global_prof.txt"])

    kwargs_wmo = [{'WMO': 1900204},
              {'WMO': [1900204, 1900243]},
              {'CYC': 1},
              {'CYC': [1, 6]},
              {'WMO': 1900204, 'CYC': 36},
              {'WMO': 1900243, 'CYC': [5, 45]},
              {'WMO': [1900482, 2900738], 'CYC': 2},
              {'WMO': [1900482, 2900738], 'CYC': [2, 6]},
              {}]

    def test_creation(self):
        assert isinstance(indexstore(), argopy.stores.argo_index.indexstore)
        assert isinstance(indexstore(cache=1), argopy.stores.argo_index.indexstore)
        assert isinstance(indexstore(cache=1, cachedir="."), argopy.stores.argo_index.indexstore)
        assert isinstance(indexstore(index_file="toto.txt"), argopy.stores.argo_index.indexstore)

    def test_search_wmo(self):
        for kw in self.kwargs_wmo:
            df = indexstore(cache=0, index_file=self.index_file).open_dataframe(indexfilter_wmo(**kw))
            assert isinstance(df, pd.core.frame.DataFrame)
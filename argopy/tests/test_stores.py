import os
import shutil
import pytest
import unittest
from unittest import TestCase

import xarray as xr
import pandas as pd
import fsspec
import argopy
from argopy.stores import filestore, httpstore
from argopy.errors import FileSystemHasNoCache, CacheFileNotFound
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

    @unittest.skipUnless(CONNECTED, "httpstore requires an internet connection to open online resources")
    def test_open_dataframe(self):
        uri = 'https://github.com/euroargodev/argopy-data/raw/master/ftp/ar_index_global_prof.txt'
        fs = httpstore()
        assert isinstance(fs.open_dataframe(uri, skiprows=8, header=0), pd.core.frame.DataFrame)

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
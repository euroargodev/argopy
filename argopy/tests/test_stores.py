import os
import shutil
import pytest
import unittest
from unittest import TestCase

import xarray as xr
import pandas as pd
import fsspec
import argopy
from argopy.stores import filestore
from argopy.errors import FileSystemHasNoCache, CacheFileNotFound

def test_filestore():
    fs = filestore(cache=0)
    assert isinstance(fs.fs, fsspec.implementations.local.LocalFileSystem)


def test_filestore_nocache():
    fs = filestore(cache=0)
    with pytest.raises(FileSystemHasNoCache):
        fs.cachepath("dummy_uri")


def test_filestore_cache():
    fs = filestore(cache=1)
    assert isinstance(fs.fs, fsspec.implementations.cached.WholeFileCacheFileSystem)


def test_filestore_nocachefile():
    fs = filestore(cache=1)
    with pytest.raises(CacheFileNotFound):
        fs.cachepath("dummy_uri")


def test_filestore_open_dataset():
    ftproot, flist = argopy.tutorial.open_dataset('localftp')
    ncfile = os.path.sep.join([ftproot, "dac/aoml/5900446/5900446_prof.nc"])
    fs = filestore()
    assert isinstance(fs.open_dataset(ncfile), xr.Dataset)


def test_filestore_open_dataframe():
    ftproot, flist = argopy.tutorial.open_dataset('localftp')
    csvfile = os.path.sep.join([ftproot, "ar_index_global_prof.txt"])
    fs = filestore()
    assert isinstance(fs.open_dataframe(csvfile, skiprows=8, header=0), pd.core.frame.DataFrame)


def test_filestore_cachefile():
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
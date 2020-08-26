import os
import numpy as np
import xarray as xr

import pytest
import unittest
import tempfile

import argopy
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import CacheFileNotFound, FileSystemHasNoCache, FtpPathError
from argopy.utilities import list_available_data_src, is_list_of_strings

AVAILABLE_SOURCES = list_available_data_src()


@unittest.skipUnless('localftp' in AVAILABLE_SOURCES, "requires localftp data fetcher")
class Backend(unittest.TestCase):
    """ Test main API facade for all available dataset and access points of the localftp fetching backend """
    src = 'localftp'
    local_ftp = argopy.tutorial.open_dataset('localftp')[0]

    def test_cachepath_notfound(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir, local_ftp=self.local_ftp):
                loader = ArgoDataFetcher(src=self.src, cache=True).profile(2901623, 12)
                with pytest.raises(CacheFileNotFound):
                    loader.fetcher.cachepath

    def test_nocache(self):
        with argopy.set_options(cachedir="dummy", local_ftp=self.local_ftp):
            loader = ArgoDataFetcher(src=self.src, cache=False).profile(2901623, 12)
            loader.to_xarray()
            with pytest.raises(FileSystemHasNoCache):
                loader.fetcher.cachepath

    def test_caching_float(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir, local_ftp=self.local_ftp):
                try:
                    loader = ArgoDataFetcher(src=self.src, cache=True).float(2901623)
                    # 1st call to load and save to cachedir:
                    ds = loader.to_xarray()
                    # 2nd call to load from cached file:
                    ds = loader.to_xarray()
                    assert isinstance(ds, xr.Dataset)
                    assert is_list_of_strings(loader.fetcher.uri)
                    assert is_list_of_strings(loader.fetcher.cachepath)
                except Exception:
                    raise

    def test_caching_profile(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir, local_ftp=self.local_ftp):
                loader = ArgoDataFetcher(src=self.src, cache=True).profile(2901623, 1)
                try:
                    # 1st call to load and save to cachedir:
                    ds = loader.to_xarray()
                    # 2nd call to load from cached file
                    ds = loader.to_xarray()
                    assert isinstance(ds, xr.Dataset)
                    assert is_list_of_strings(loader.fetcher.uri)
                    assert is_list_of_strings(loader.fetcher.cachepath)
                except Exception:
                    raise

    def test_caching_region(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir, local_ftp=self.local_ftp):
                loader = ArgoDataFetcher(src=self.src, cache=True).region([-60, -40, 40., 60., 0., 100., '2007-08-01', '2007-09-01'])
                try:
                    # 1st call to load and save to cachedir:
                    ds = loader.to_xarray()
                    # 2nd call to load from cached file
                    ds = loader.to_xarray()
                    assert isinstance(ds, xr.Dataset)
                    assert is_list_of_strings(loader.fetcher.uri)
                    assert is_list_of_strings(loader.fetcher.cachepath)
                except Exception:
                    raise

    def test_invalidFTPpath(self):
        with pytest.raises(ValueError):
            with argopy.set_options(local_ftp="dummy"):
                ArgoDataFetcher(src=self.src).profile(2901623, 12)

        with pytest.raises(FtpPathError):
            with argopy.set_options(local_ftp=os.path.sep.join([self.local_ftp, "dac"])):
                ArgoDataFetcher(src=self.src).profile(2901623, 12)

    def __testthis_profile(self, dataset):
        with argopy.set_options(local_ftp=self.local_ftp):
            fetcher_args = {"src": self.src, 'ds': dataset}
            for arg in self.args['profile']:
                try:
                    f = ArgoDataFetcher(**fetcher_args).profile(*arg)
                    assert isinstance(f.to_xarray(), xr.Dataset)
                    assert is_list_of_strings(f.fetcher.uri)
                except Exception:
                    print("ERROR LOCALFTP request:\n", f.fetcher.uri)
                    pass

    def __testthis_float(self, dataset):
        with argopy.set_options(local_ftp=self.local_ftp):
            fetcher_args = {"src": self.src, 'ds': dataset}
            for arg in self.args['float']:
                try:
                    f = ArgoDataFetcher(**fetcher_args).float(arg)
                    assert isinstance(f.to_xarray(), xr.Dataset)
                    assert is_list_of_strings(f.fetcher.uri)
                except Exception:
                    print("ERROR LOCALFTP request:\n", f.fetcher.uri)
                    pass

    def __testthis_region(self, dataset):
        with argopy.set_options(local_ftp=self.local_ftp):
            fetcher_args = {"src": self.src, 'ds': dataset}
            for arg in self.args['region']:
                try:
                    f = ArgoDataFetcher(**fetcher_args).region(arg)
                    assert isinstance(f.to_xarray(), xr.Dataset)
                    assert is_list_of_strings(f.fetcher.uri)
                except Exception:
                    print("ERROR LOCALFTP request:\n", f.fetcher.uri)
                    pass

    def __testthis(self, dataset):
        for access_point in self.args:
            if access_point == 'profile':
                self.__testthis_profile(dataset)
            elif access_point == 'float':
                self.__testthis_float(dataset)
            elif access_point == 'region':
                self.__testthis_region(dataset)

    def test_phy_float(self):
        self.args = {}
        self.args['float'] = [[2901623],
                              [6901929, 2901623]]
        self.__testthis('phy')

    def test_phy_profile(self):
        self.args = {}
        self.args['profile'] = [[2901623, 12],
                                [2901623, np.arange(12, 14)],
                                [2901623, [1, 6]]]
        self.__testthis('phy')

    def test_phy_region(self):
        self.args = {}
        self.args['region'] = [[-60, -40, 40., 60., 0., 100.],
                               [-60, -40, 40., 60., 0., 100., '2007-08-01', '2007-09-01']]
        self.__testthis('phy')


class BackendParallel(unittest.TestCase):
    """ This test backend for parallel requests """

    src = "localftp"
    local_ftp = argopy.tutorial.open_dataset('localftp')[0]

    requests = {}
    requests["region"] = [
        [-60, -40, 40., 60., 0., 100.],
        [-60, -40, 40., 60., 0., 100., "2007-08-01", "2007-09-01"],
    ]
    requests["wmo"] = [[
        5900446,
        5906072,
        6901929
    ]]

    def test_methods(self):
        args_list = [
            {"src": self.src, "parallel": "thread"},
            {"src": self.src, "parallel": True, "parallel_method": "thread"},
            {"src": self.src, "parallel": "process"},
            {"src": self.src, "parallel": True, "parallel_method": "process"},
        ]
        with argopy.set_options(local_ftp=self.local_ftp):
            for fetcher_args in args_list:
                loader = ArgoDataFetcher(**fetcher_args).float(self.requests["wmo"][0])
                assert isinstance(loader, argopy.fetchers.ArgoDataFetcher)

        args_list = [
            {"src": self.src, "parallel": "toto"},
            {"src": self.src, "parallel": True, "parallel_method": "toto"},
        ]
        with argopy.set_options(local_ftp=self.local_ftp):
            for fetcher_args in args_list:
                with pytest.raises(ValueError):
                    ArgoDataFetcher(**fetcher_args).float(self.requests["wmo"][0])

    def test_chunks_region(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            fetcher_args = {"src": self.src, "parallel": True, 'chunks':{'lon': 1, 'lat': 2, 'dpt': 1, 'time': 2}}
            for access_arg in self.requests["region"]:
                try:
                    f = ArgoDataFetcher(**fetcher_args).region(access_arg)
                    assert isinstance(f.to_xarray(), xr.Dataset)
                    assert is_list_of_strings(f.fetcher.uri)
                except Exception:
                    print("ERROR LOCALFTP request:\n", f.fetcher.uri)
                    pass

    def test_chunks_wmo(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            fetcher_args = {"src": self.src, "parallel": True, "chunks": {'wmo': 2}}
            for access_arg in self.requests["wmo"]:
                try:
                    # f = ArgoDataFetcher(**fetcher_args).float(access_arg)
                    f = ArgoDataFetcher(**fetcher_args).profile(access_arg, 1)
                    assert isinstance(f.to_xarray(), xr.Dataset)
                    assert is_list_of_strings(f.fetcher.uri)
                except Exception:
                    print("ERROR LOCALFTP request:\n", f.fetcher.uri)
                    pass


if __name__ == '__main__':
    unittest.main()

import xarray as xr
import tempfile

import pytest

import argopy
from argopy import IndexFetcher as ArgoIndexFetcher
from argopy.errors import FileSystemHasNoCache, CacheFileNotFound, DataNotFound
from . import requires_localftp_index, requires_connection


@requires_localftp_index
@requires_connection
class Test_Backend:
    """ Test localftp index fetcher """

    src = 'localftp'
    local_ftp = argopy.tutorial.open_dataset('localftp')[0]

    def test_cachepath_notfound(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir, local_ftp=self.local_ftp):
                loader = ArgoIndexFetcher(src=self.src, cache=True).profile(2901623, 12)
                with pytest.raises(CacheFileNotFound):
                    loader.fetcher.cachepath

    def test_nocache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir, local_ftp=self.local_ftp):
                loader = ArgoIndexFetcher(src=self.src, cache=False).profile(2901623, 12)
                loader.to_xarray()
                with pytest.raises(FileSystemHasNoCache):
                    loader.fetcher.cachepath

    def test_clearcache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir, local_ftp=self.local_ftp):
                loader = ArgoIndexFetcher(src=self.src, cache=True).profile(2901623, 12)
                try:
                    loader.to_xarray()
                    loader.clear_cache()
                    with pytest.raises(CacheFileNotFound):
                        loader.fetcher.cachepath
                except Exception:
                    raise

    def test_caching(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir, local_ftp=self.local_ftp):
                try:
                    loader = ArgoIndexFetcher(src=self.src, cache=True).float(6901929)
                    # 1st call to load and save to cache:
                    loader.to_xarray()
                    # 2nd call to load from cached file:
                    ds = loader.to_xarray()
                    assert isinstance(ds, xr.Dataset)
                    assert isinstance(loader.fetcher.cachepath, str)
                except Exception:
                    raise

    def test_noresults(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            with pytest.raises(DataNotFound):
                ArgoIndexFetcher(src=self.src).region([-70, -65, 30., 35., '2030-01-01', '2030-06-30']).to_dataframe()

    def __testthis(self):
        for access_point in self.args:

            if access_point == 'profile':
                for arg in self.args['profile']:
                    with argopy.set_options(local_ftp=self.local_ftp):
                        fetcher = ArgoIndexFetcher(src=self.src).profile(*arg).fetcher
                        try:
                            ds = fetcher.to_xarray()
                            assert isinstance(ds, xr.Dataset)
                        except Exception:
                            print("ERROR LOCALFTP request:\n", fetcher.cname())
                            pass

            if access_point == 'float':
                for arg in self.args['float']:
                    with argopy.set_options(local_ftp=self.local_ftp):
                        fetcher = ArgoIndexFetcher(src=self.src).float(arg).fetcher
                        try:
                            ds = fetcher.to_xarray()
                            assert isinstance(ds, xr.Dataset)
                        except Exception:
                            print("ERROR LOCALFTP request:\n", fetcher.cname())
                            pass

            if access_point == 'region':
                for arg in self.args['region']:
                    with argopy.set_options(local_ftp=self.local_ftp):
                        fetcher = ArgoIndexFetcher(src=self.src).region(arg).fetcher
                        try:
                            ds = fetcher.to_xarray()
                            assert isinstance(ds, xr.Dataset)
                        except Exception:
                            print("ERROR LOCALFTP request:\n", fetcher.cname())
                            pass

    def test_phy_float(self):
        self.args = {'float': [[2901623],
                               [2901623, 6901929]]}
        self.__testthis()

    def test_phy_profile(self):
        self.args = {'profile': [[6901929, 36],
                                 [6901929, [5, 45]]]}
        self.__testthis()

    def test_phy_region(self):
        self.args = {'region': [[-60, -40, 40., 60.],
                                [-60, -40, 40., 60., '2007-08-01', '2007-09-01']]}
        self.__testthis()

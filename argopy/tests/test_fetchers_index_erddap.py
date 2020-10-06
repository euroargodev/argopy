import xarray as xr

import pytest
import tempfile

import argopy
from argopy import IndexFetcher as ArgoIndexFetcher
from argopy.errors import (
    FileSystemHasNoCache,
    CacheFileNotFound
)
from . import requires_connected_erddap_index, safe_to_server_errors


@requires_connected_erddap_index
class Test_Backend:
    """ Test ERDDAP index fetching backend """

    src = "erddap"
    requests = {
        "float": [[2901623], [2901623, 6901929]],
        "region": [
            [-60, -40, 40.0, 60.0],
            [-60, -40, 40.0, 60.0, "2007-08-01", "2007-09-01"],
        ],
    }

    def test_cachepath_notfound(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                loader = ArgoIndexFetcher(src=self.src, cache=True).float(self.requests['float'][0])
                with pytest.raises(CacheFileNotFound):
                    loader.fetcher.cachepath

    @pytest.mark.skip(
        reason="Waiting for https://github.com/euroargodev/argopy/issues/16"
    )
    @safe_to_server_errors
    def test_nocache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                loader = ArgoIndexFetcher(src=self.src, cache=False).float(self.requests['float'][0])
                loader.to_xarray()
                with pytest.raises(FileSystemHasNoCache):
                    loader.fetcher.cachepath

    @pytest.mark.skip(
        reason="Waiting for https://github.com/euroargodev/argopy/issues/16"
    )
    @safe_to_server_errors
    def test_clearcache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                loader = ArgoIndexFetcher(src=self.src, cache=True).float(self.requests['float'][0])
                loader.to_xarray()
                loader.clear_cache()
                with pytest.raises(CacheFileNotFound):
                    loader.fetcher.cachepath

    @pytest.mark.skip(
        reason="Waiting for https://github.com/euroargodev/argopy/issues/16"
    )
    @safe_to_server_errors
    def test_caching(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                loader = ArgoIndexFetcher(src=self.src, cache=True).float(self.requests['float'][0])
                # 1st call to load and save to cache:
                loader.to_xarray()
                # 2nd call to load from cached file:
                ds = loader.to_xarray()
                assert isinstance(ds, xr.Dataset)
                assert isinstance(loader.fetcher.cachepath, str)

    def test_url(self):
        loader = ArgoIndexFetcher(src=self.src, cache=True).float(self.requests['float'][0])
        assert isinstance(loader.fetcher.url, str)
        # loader = ArgoIndexFetcher(src=self.src, cache=True).region([-60, -40, 40., 60., '2007-08-01', '2007-09-01'])
        # assert isinstance(loader.fetcher.url, str)

    def __testthis(self):
        for access_point in self.args:

            # Acces point not implemented yet for erddap
            # if access_point == 'profile':
            #     for arg in self.args['profile']:
            #         fetcher = ArgoIndexFetcher(src=self.src).profile(*arg).fetcher
            #         try:
            #             ds = fetcher.to_xarray()
            #             assert isinstance(ds, xr.Dataset)
            #         except ErddapServerError:
            #             # Test is passed even if something goes wrong with the erddap server
            #             pass
            #         except Exception:
            #             print("ERROR ERDDAP request:\n", fetcher.cname())
            #             pass

            if access_point == "float":
                for arg in self.args["float"]:
                    fetcher = ArgoIndexFetcher(src=self.src).float(arg).fetcher
                    ds = fetcher.to_xarray()
                    assert isinstance(ds, xr.Dataset)

            if access_point == "region":
                for arg in self.args["region"]:
                    fetcher = ArgoIndexFetcher(src=self.src).region(arg).fetcher
                    ds = fetcher.to_xarray()
                    assert isinstance(ds, xr.Dataset)

    @safe_to_server_errors
    def test_phy_float(self):
        self.args = {"float": self.requests["float"]}
        self.__testthis()

    # @pytest.mark.skip(reason="Waiting for https://github.com/euroargodev/argopy/issues/16")
    # def test_phy_profile(self):
    #     self.args = {'profile': [[6901929, 36],
    #                              [6901929, [5, 45]]]}
    #     self.__testthis()

    @pytest.mark.skip(
        reason="Waiting for https://github.com/euroargodev/argopy/issues/16"
    )
    @safe_to_server_errors
    def test_phy_region(self):
        self.args = {"region": self.requests["region"]}
        self.__testthis()

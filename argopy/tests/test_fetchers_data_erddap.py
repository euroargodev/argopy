import numpy as np
import xarray as xr

import pytest
import tempfile

import argopy
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import ErddapServerError, CacheFileNotFound, FileSystemHasNoCache
from argopy.utilities import is_list_of_strings
from . import (
    requires_connected_erddap,
    requires_connected_erddap_phy,
    requires_connected_erddap_bgc,
    requires_connected_erddap_ref,
)


@requires_connected_erddap
class Test_Backend:
    """ Test ERDDAP data fetching backend """

    src = "erddap"

    @requires_connected_erddap_phy
    def test_cachepath_notfound(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                loader = ArgoDataFetcher(src=self.src, cache=True).profile(6902746, 34)
                with pytest.raises(CacheFileNotFound):
                    loader.fetcher.cachepath

    @requires_connected_erddap_phy
    def test_nocache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                loader = ArgoDataFetcher(src=self.src, cache=False).profile(6902746, 34)
                loader.to_xarray()
                with pytest.raises(FileSystemHasNoCache):
                    loader.fetcher.cachepath

    @requires_connected_erddap_phy
    def test_clearcache(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                loader = ArgoDataFetcher(src=self.src, cache=True).float(6902746)
                try:
                    loader.to_xarray()
                    loader.clear_cache()
                    with pytest.raises(CacheFileNotFound):
                        loader.fetcher.cachepath
                except ErddapServerError:
                    # Test is passed even if something goes wrong with the erddap server
                    pass
                except Exception:
                    raise

    @requires_connected_erddap_phy
    def test_caching_float(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                f = ArgoDataFetcher(src=self.src, cache=True).float(1901393).fetcher
                try:
                    # 1st call to load and save to cache:
                    f.to_xarray()
                    # 2nd call to load from cached file:
                    ds = f.to_xarray()
                    assert isinstance(ds, xr.Dataset)
                    assert is_list_of_strings(f.uri)
                    assert is_list_of_strings(f.cachepath)
                except ErddapServerError:
                    # Test is passed even if something goes wrong with the erddap server
                    pass
                except Exception:
                    raise

    @requires_connected_erddap_phy
    def test_caching_profile(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                f = ArgoDataFetcher(src=self.src, cache=True).profile(6902746, 34).fetcher
                try:
                    # 1st call to load and save to cache:
                    f.to_xarray()
                    # 2nd call to load from cached file:
                    ds = f.to_xarray()
                    assert isinstance(ds, xr.Dataset)
                    assert is_list_of_strings(f.uri)
                    assert is_list_of_strings(f.cachepath)
                except ErddapServerError:
                    # Test is passed even if something goes wrong with the erddap server
                    pass
                except Exception:
                    raise

    @requires_connected_erddap_phy
    def test_caching_region(self):
        with tempfile.TemporaryDirectory() as testcachedir:
            with argopy.set_options(cachedir=testcachedir):
                f = ArgoDataFetcher(src=self.src, cache=True).region(
                    [-40, -30, 30, 40, 0, 100, "2011", "2012"]
                ).fetcher
                try:
                    # 1st call to load and save to cache:
                    f.to_xarray()
                    # 2nd call to load from cached file:
                    ds = f.to_xarray()
                    assert isinstance(ds, xr.Dataset)
                    assert is_list_of_strings(f.uri)
                    assert is_list_of_strings(f.cachepath)
                except ErddapServerError:
                    # Test is passed even if something goes wrong with the erddap server
                    pass
                except Exception:
                    raise

    @requires_connected_erddap_phy
    def test_N_POINTS(self):
        n = (
            ArgoDataFetcher(src=self.src)
            .region([-70, -65, 35.0, 40.0, 0, 10.0, "2012-01", "2013-12"])
            .fetcher.N_POINTS
        )
        assert isinstance(n, int)

    def __testthis_profile(self, dataset):
        fetcher_args = {"src": self.src, 'ds': dataset}
        for arg in self.args["profile"]:
            try:
                f = ArgoDataFetcher(**fetcher_args).profile(*arg).fetcher
                assert isinstance(f.to_xarray(), xr.Dataset)
                assert is_list_of_strings(f.uri)
            except ErddapServerError:
                # Test is passed even if something goes wrong with the erddap server
                pass
            except Exception:
                print("\nERROR with ERDDAP request:\n", "\n".join(f.uri))
                pass

    def __testthis_float(self, dataset):
        fetcher_args = {"src": self.src, 'ds': dataset}
        for arg in self.args["float"]:
            try:
                f = ArgoDataFetcher(**fetcher_args).float(arg).fetcher
                assert isinstance(f.to_xarray(), xr.Dataset)
                assert is_list_of_strings(f.uri)
            except ErddapServerError:
                # Test is passed even if something goes wrong with the erddap server
                pass
            except Exception:
                print("\nERROR with ERDDAP request:\n", "\n".join(f.uri))
                pass

    def __testthis_region(self, dataset):
        fetcher_args = {"src": self.src, 'ds': dataset}
        for arg in self.args["region"]:
            try:
                f = ArgoDataFetcher(**fetcher_args).region(arg).fetcher
                assert isinstance(f.to_xarray(), xr.Dataset)
                assert is_list_of_strings(f.uri)
            except ErddapServerError:
                # Test is passed even if something goes wrong with the erddap server
                pass
            except Exception:
                print("\nERROR with ERDDAP request:\n", "\n".join(f.uri))
                pass

    def __testthis(self, dataset):
        for access_point in self.args:
            if access_point == "profile":
                self.__testthis_profile(dataset)
            elif access_point == "float":
                self.__testthis_float(dataset)
            elif access_point == "region":
                self.__testthis_region(dataset)

    @requires_connected_erddap_phy
    def test_phy_float(self):
        self.args = {}
        self.args["float"] = [[1901393], [1901393, 6902746]]
        self.__testthis("phy")

    @requires_connected_erddap_phy
    def test_phy_profile(self):
        self.args = {}
        self.args["profile"] = [
            [6902746, 34],
            [6902746, np.arange(12, 13)],
            [6902746, [1, 12]],
        ]
        self.__testthis("phy")

    @requires_connected_erddap_phy
    def test_phy_region(self):
        self.args = {}
        self.args["region"] = [
            [-70, -65, 35.0, 40.0, 0, 10.0],
            [-70, -65, 35.0, 40.0, 0, 10.0, "2012-01", "2013-12"],
        ]
        self.__testthis("phy")

    @requires_connected_erddap_bgc
    def test_bgc_float(self):
        self.args = {}
        self.args["float"] = [[5903248], [7900596, 2902264]]
        self.__testthis("bgc")

    @requires_connected_erddap_bgc
    def test_bgc_profile(self):
        self.args = {}
        self.args["profile"] = [
            [5903248, 34],
            [5903248, np.arange(12, 14)],
            [5903248, [1, 12]],
        ]
        self.__testthis("bgc")

    @requires_connected_erddap_bgc
    def test_bgc_region(self):
        self.args = {}
        self.args["region"] = [
            [-70, -65, 35.0, 40.0, 0, 10.0],
            [-70, -65, 35.0, 40.0, 0, 10.0, "2012-01-1", "2012-12-31"],
        ]
        self.__testthis("bgc")

    @requires_connected_erddap_ref
    def test_ref_region(self):
        self.args = {}
        self.args["region"] = [
            [-70, -65, 35.0, 40.0, 0, 10.0],
            [-70, -65, 35.0, 40.0, 0, 10.0, "2012-01-01", "2012-12-31"],
        ]
        self.__testthis("ref")


@requires_connected_erddap_phy
class Test_BackendParallel:
    """ This test backend for parallel requests """

    src = "erddap"
    requests = {"region": [
        [-60, -55, 40.0, 45.0, 0.0, 20.0],
        [-60, -55, 40.0, 45.0, 0.0, 20.0, "2007-08-01", "2007-09-01"],
    ], "wmo": [[6902766, 6902772, 6902914]]}

    def test_methods(self):
        args_list = [
            {"src": self.src, "parallel": "thread"},
            {"src": self.src, "parallel": True, "parallel_method": "thread"},
        ]
        for fetcher_args in args_list:
            loader = ArgoDataFetcher(**fetcher_args).float(self.requests["wmo"][0])
            assert isinstance(loader, argopy.fetchers.ArgoDataFetcher)

        args_list = [
            {"src": self.src, "parallel": True, "parallel_method": "toto"},
            {"src": self.src, "parallel": "process"},
            {"src": self.src, "parallel": True, "parallel_method": "process"},
        ]
        for fetcher_args in args_list:
            with pytest.raises(ValueError):
                ArgoDataFetcher(**fetcher_args).float(self.requests["wmo"][0])

    def test_chunks_region(self):
        for access_arg in self.requests["region"]:
            fetcher_args = {
                "src": self.src,
                "parallel": True,
                "chunks": {"lon": 1, "lat": 2, "dpt": 2, "time": 1},
            }
            try:
                f = ArgoDataFetcher(**fetcher_args).region(access_arg).fetcher
                assert isinstance(f.to_xarray(), xr.Dataset)
                assert is_list_of_strings(f.uri)
                assert len(f.uri) == np.prod([v for k, v in fetcher_args['chunks'].items()])
            except ErddapServerError:
                # Test is passed even if something goes wrong with the erddap server
                pass
            except Exception:
                print("\nERDDAP request:\n", "\n".join(f.uri))
                pass

    def test_chunks_wmo(self):
        for access_arg in self.requests["wmo"]:
            fetcher_args = {"src": self.src, "parallel": True, "chunks_maxsize": {"wmo": 1}}
            try:
                f = ArgoDataFetcher(**fetcher_args).profile(access_arg, 12).fetcher
                assert isinstance(f.to_xarray(), xr.Dataset)
                assert is_list_of_strings(f.uri)
                assert len(f.uri) == len(access_arg)
            except ErddapServerError:
                # Test is passed even if something goes wrong with the erddap server
                pass
            except Exception as e:
                print("\nERDDAP request:\n", "\n".join(f.uri))
                raise

import os
import pandas as pd
import argopy
import tempfile
from argopy import DataFetcher as ArgoDataFetcher
from utils import (
    requires_gdac,
)
from argopy.utils.caching import lscache, clear_cache


@requires_gdac
def test_clear_cache():
    ftproot, flist = argopy.tutorial.open_dataset("gdac")
    with tempfile.TemporaryDirectory() as cachedir:
        with argopy.set_options(cachedir=cachedir):
            loader = ArgoDataFetcher(src="gdac", gdac=ftproot, cache=True).profile(2902696, 12)
            loader.to_xarray()
            clear_cache()
            assert os.path.exists(cachedir) is True
            assert len(os.listdir(cachedir)) == 0


@requires_gdac
def test_lscache():
    ftproot, flist = argopy.tutorial.open_dataset("gdac")
    with tempfile.TemporaryDirectory() as cachedir:
        with argopy.set_options(cachedir=cachedir):
            loader = ArgoDataFetcher(src="gdac", gdac=ftproot, cache=True).profile(2902696, 12)
            loader.to_xarray()
            result = lscache(cache_path=cachedir, prt=True)
            assert isinstance(result, str)

            result = lscache(cache_path=cachedir, prt=False)
            assert isinstance(result, pd.DataFrame)


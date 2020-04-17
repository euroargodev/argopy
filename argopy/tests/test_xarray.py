import os, sys
import pytest
import unittest
import xarray as xr
# import argopy
from argopy import DataFetcher as ArgoDataFetcher
# from argopy.options import OPTIONS

from argopy.utilities import list_available_data_src, isconnected
AVAILABLE_SOURCES = list_available_data_src()
CONNECTED = isconnected()

@unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
@unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
def test_point2profile2point():
    ds_pts = ArgoDataFetcher(src='erddap')\
                .region([-75, -55, 30., 40., 0, 100., '2011-01-01', '2011-01-15'])\
                .to_xarray()
    assert ds_pts.argo.point2profile().argo.profile2point().equals(ds_pts)

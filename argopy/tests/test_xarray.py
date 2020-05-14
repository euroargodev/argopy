import os, sys
import pytest
import unittest
import xarray as xr

from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import InvalidDatasetStructure, ErddapServerError

from argopy.utilities import list_available_data_src, isconnected, erddap_ds_exists
AVAILABLE_SOURCES = list_available_data_src()
CONNECTED = isconnected()
if CONNECTED:
    DSEXISTS = erddap_ds_exists(ds="ArgoFloats")
else:
    DSEXISTS = False

@unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
@unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
@unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
def test_point2profile():
    try:
        ds = ArgoDataFetcher(src='erddap')\
                    .region([-75, -55, 30., 40., 0, 100., '2011-01-01', '2011-01-15'])\
                    .to_xarray()
        assert 'N_PROF' in ds.argo.point2profile().dims
    except ErddapServerError:  # Test is passed when something goes wrong because of the erddap server, not our fault !
        pass

@unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
@unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
@unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
def test_profile2point():
    try:
        ds = ArgoDataFetcher(src='erddap')\
                    .region([-75, -55, 30., 40., 0, 100., '2011-01-01', '2011-01-15'])\
                    .to_xarray()
        with pytest.raises(InvalidDatasetStructure):
            ds.argo.profile2point()
    except ErddapServerError:  # Test is passed when something goes wrong because of the erddap server, not our fault !
        pass

@unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
@unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
@unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
def test_point2profile2point():
    try:
        ds_pts = ArgoDataFetcher(src='erddap') \
            .region([-75, -55, 30., 40., 0, 100., '2011-01-01', '2011-01-15']) \
            .to_xarray()
        assert ds_pts.argo.point2profile().argo.profile2point().equals(ds_pts)
    except ErddapServerError:  # Test is passed when something goes wrong because of the erddap server, not our fault !
        pass

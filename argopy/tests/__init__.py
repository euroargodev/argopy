# -*coding: UTF-8 -*-
"""

Test suite for argopy continuous integration

"""

import importlib

import xarray
import pandas
import pytest

print("xarray: %s, %s" % (xarray.__version__, xarray.__file__))
print("pandas: %s, %s" % (pandas.__version__, pandas.__file__))

def _importorskip(modname):
    try:
        mod = importlib.import_module(modname)
        has = True
    except ImportError:
        has = False
    func = pytest.mark.skipif(not has, reason="requires {}".format(modname))
    return has, func

def _connectskip(connected, msg):
    func = pytest.mark.skipif(not connected, reason="requires {}".format(msg))
    return connected, func


from argopy.utilities import list_available_data_src, isconnected, erddap_ds_exists
AVAILABLE_SOURCES = list_available_data_src()
CONNECTED = isconnected()
if CONNECTED:
    DSEXISTS = erddap_ds_exists(ds="ArgoFloats")
else:
    DSEXISTS = False


# @unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
has_erddap, requires_erddap = _connectskip('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")

# @unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
has_connection, requires_connection = _connectskip(CONNECTED, "requires an internet connection")

# @unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
has_erddap_phy, requires_erddap_phy = _connectskip(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")

has_connected_erddap_phy = has_connection and has_erddap and has_erddap_phy
requires_connected_erddap_phy = pytest.mark.skipif(
    not has_connected_erddap_phy, reason="requires a live core Argo dataset from Ifremer erddap server"
)




# has_netCDF4, requires_netCDF4 = _importorskip("netCDF4")
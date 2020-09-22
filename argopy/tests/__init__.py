# -*coding: UTF-8 -*-
"""

Test suite for argopy continuous integration

"""

import importlib
import pytest
import argopy
argopy.set_options(api_timeout=4 * 60)  # From Github actions, requests can take a while

def _importorskip(modname):
    try:
        importlib.import_module(modname)
        has = True
    except ImportError:
        has = False
    func = pytest.mark.skipif(not has, reason="requires {}".format(modname))
    return has, func

def _connectskip(connected, msg):
    func = pytest.mark.skipif(not connected, reason="requires {}".format(msg))
    return connected, func

def _xfail(name, msg):
    func = pytest.mark.xfail(run=name, reason=msg)
    return name, func

from argopy.utilities import list_available_data_src, isconnected, erddap_ds_exists
AVAILABLE_SOURCES = list_available_data_src()
CONNECTED = isconnected()
if CONNECTED:
    DSEXISTS = erddap_ds_exists(ds="ArgoFloats")
    DSEXISTS_bgc = erddap_ds_exists(ds="ArgoFloats-bio")
    DSEXISTS_ref = erddap_ds_exists(ds="ArgoFloats-ref")
else:
    DSEXISTS = False
    DSEXISTS_bgc = False
    DSEXISTS_ref = False

has_connection, requires_connection = _connectskip(CONNECTED, "requires an internet connection")

has_erddap, requires_erddap = _connectskip('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")

has_erddap_phy, requires_erddap_phy = _connectskip(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")

has_erddap_bgc, requires_erddap_bgc = _connectskip(DSEXISTS_bgc, "erddap requires a valid BGC Argo dataset from Ifremer server")

has_erddap_ref, requires_erddap_ref = _connectskip(DSEXISTS_ref, "erddap requires a valid Reference Argo dataset from Ifremer server")

has_connected_erddap = has_connection and has_erddap
requires_connected_erddap = pytest.mark.skipif(
    not has_connected_erddap, reason="requires a live Ifremer erddap server"
)
has_connected_erddap_phy = has_connection and has_erddap and has_erddap_phy
requires_connected_erddap_phy = pytest.mark.skipif(
    not has_connected_erddap_phy, reason="requires a live and valid core Argo dataset from Ifremer erddap server"
)
has_connected_erddap_bgc = has_connection and has_erddap and has_erddap_bgc
requires_connected_erddap_bgc = pytest.mark.skipif(
    not has_connected_erddap_bgc, reason="requires a live and valid BGC Argo dataset from Ifremer erddap server"
)
has_connected_erddap_ref = has_connection and has_erddap and has_erddap_ref
requires_connected_erddap_ref = pytest.mark.skipif(
    not has_connected_erddap_ref, reason="requires a live and valid Reference Argo dataset from Ifremer erddap server"
)

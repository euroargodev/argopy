# -*coding: UTF-8 -*-
"""

Test suite for argopy continuous integration

"""

import importlib
import pytest
import argopy
argopy.set_options(api_timeout=3 * 60)  # From Github actions, requests can take a while
argopy.show_versions()

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

from argopy.utilities import list_available_data_src, list_available_index_src, isconnected, erddap_ds_exists
AVAILABLE_SOURCES = list_available_data_src()
AVAILABLE_INDEX_SOURCES = list_available_index_src()
CONNECTED = isconnected()

has_fetcher, requires_fetcher = _connectskip(len(AVAILABLE_SOURCES) > 0, "requires at least one data fetcher")
has_fetcher_index, requires_fetcher_index = _connectskip(len(AVAILABLE_INDEX_SOURCES) > 0, "requires at least one index fetcher")

##########
# ERDDAP #
##########
if CONNECTED:
    DSEXISTS = erddap_ds_exists(ds="ArgoFloats")
    DSEXISTS_bgc = erddap_ds_exists(ds="ArgoFloats-bio")
    DSEXISTS_ref = erddap_ds_exists(ds="ArgoFloats-ref")
    DSEXISTS_index = erddap_ds_exists(ds="ArgoFloats-index")
else:
    DSEXISTS = False
    DSEXISTS_bgc = False
    DSEXISTS_ref = False
    DSEXISTS_index = False

has_connection, requires_connection = _connectskip(CONNECTED, "requires an internet connection")

has_erddap, requires_erddap = _connectskip('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")

has_erddap_phy, requires_erddap_phy = _connectskip(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")

has_erddap_bgc, requires_erddap_bgc = _connectskip(DSEXISTS_bgc, "erddap requires a valid BGC Argo dataset from Ifremer server")

has_erddap_ref, requires_erddap_ref = _connectskip(DSEXISTS_ref, "erddap requires a valid Reference Argo dataset from Ifremer server")

has_erddap_index, requires_erddap_index = _connectskip(DSEXISTS_index, "erddap requires a valid Argo Index from Ifremer server")

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
has_connected_erddap_index = has_connection and has_erddap and has_erddap_index
requires_connected_erddap_index = pytest.mark.skipif(
    not has_connected_erddap_index, reason="requires a live and valid Argo Index from Ifremer erddap server"
)

###########
# ARGOVIS #
###########
has_argovis, requires_argovis = _connectskip('argovis' in AVAILABLE_SOURCES, "requires argovis data fetcher")

has_connected_argovis = has_connection and has_argovis
requires_connected_argovis = pytest.mark.skipif(
    not has_connected_argovis, reason="requires a live Argovis server"
)

############
# LOCALFTP #
############
has_localftp, requires_localftp = _connectskip('localftp' in AVAILABLE_SOURCES, "requires localftp data fetcher")
has_localftp_index, requires_localftp_index = _connectskip('localftp' in AVAILABLE_INDEX_SOURCES, "requires localftp index fetcher")

# -*coding: UTF-8 -*-
"""

Test suite for argopy continuous integration

"""
import traceback
import importlib
import pytest
import fsspec
import argopy
from aiohttp.client_exceptions import ServerDisconnectedError, ClientResponseError
from packaging import version
import warnings
from argopy.errors import ErddapServerError, ArgovisServerError, DataNotFound
from argopy.utilities import (
    list_available_data_src,
    list_available_index_src,
    isconnected,
    erddap_ds_exists,
)

argopy.set_options(api_timeout=4 * 60)  # From Github actions, requests can take a while
argopy.show_options()
argopy.show_versions()


def _importorskip(modname):
    try:
        importlib.import_module(modname)  # noqa: E402
        has = True
    except ImportError:
        has = False
    func = pytest.mark.skipif(not has, reason="Requires {}".format(modname))
    return has, func


def _connectskip(connected, msg):
    func = pytest.mark.skipif(not connected, reason="Requires {}".format(msg))
    return connected, func


def _xfail(name, msg):
    func = pytest.mark.xfail(run=name, reason=msg)
    return name, func


AVAILABLE_SOURCES = list_available_data_src()
AVAILABLE_INDEX_SOURCES = list_available_index_src()
CONNECTED = isconnected()

has_fetcher, requires_fetcher = _connectskip(
    len(AVAILABLE_SOURCES) > 0, "at least one data fetcher"
)
has_fetcher_index, requires_fetcher_index = _connectskip(
    len(AVAILABLE_INDEX_SOURCES) > 0, "at least one index fetcher"
)

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

has_connection, requires_connection = _connectskip(
    CONNECTED, "an internet connection"
)

has_erddap, requires_erddap = _connectskip(
    "erddap" in AVAILABLE_SOURCES, "erddap data fetcher"
)

has_erddap_phy, requires_erddap_phy = _connectskip(
    DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server"
)

has_erddap_bgc, requires_erddap_bgc = _connectskip(
    DSEXISTS_bgc, "erddap requires a valid BGC Argo dataset from Ifremer server"
)

has_erddap_ref, requires_erddap_ref = _connectskip(
    DSEXISTS_ref, "erddap requires a valid Reference Argo dataset from Ifremer server"
)

has_erddap_index, requires_erddap_index = _connectskip(
    DSEXISTS_index, "erddap requires a valid Argo Index from Ifremer server"
)

has_connected_erddap = has_connection and has_erddap
requires_connected_erddap = pytest.mark.skipif(
    not has_connected_erddap, reason="Requires a live Ifremer erddap server"
)
has_connected_erddap_phy = has_connection and has_erddap and has_erddap_phy
requires_connected_erddap_phy = pytest.mark.skipif(
    not has_connected_erddap_phy,
    reason="Requires a live and valid core Argo dataset from Ifremer erddap server",
)
has_connected_erddap_bgc = has_connection and has_erddap and has_erddap_bgc
requires_connected_erddap_bgc = pytest.mark.skipif(
    not has_connected_erddap_bgc,
    reason="Requires a live and valid BGC Argo dataset from Ifremer erddap server",
)
has_connected_erddap_ref = has_connection and has_erddap and has_erddap_ref
requires_connected_erddap_ref = pytest.mark.skipif(
    not has_connected_erddap_ref,
    reason="Requires a live and valid Reference Argo dataset from Ifremer erddap server",
)
has_connected_erddap_index = has_connection and has_erddap and has_erddap_index
requires_connected_erddap_index = pytest.mark.skipif(
    not has_connected_erddap_index,
    reason="Requires a live and valid Argo Index from Ifremer erddap server",
)

ci_erddap_index = pytest.mark.skipif(True, reason="Tests disabled for erddap index fetcher")

###########
# ARGOVIS #
###########
has_argovis, requires_argovis = _connectskip(
    "argovis" in AVAILABLE_SOURCES, "argovis data fetcher"
)

has_connected_argovis = has_connection and has_argovis
requires_connected_argovis = pytest.mark.skipif(
    not has_connected_argovis, reason="Requires a live Argovis server"
)

############
# LOCALFTP #
############
has_localftp, requires_localftp = _connectskip(
    "localftp" in AVAILABLE_SOURCES, "the localftp data fetcher"
)
has_localftp_index, requires_localftp_index = _connectskip(
    "localftp" in AVAILABLE_INDEX_SOURCES, "the localftp index fetcher"
)

########
# PLOT #
########
has_matplotlib, requires_matplotlib = _importorskip("matplotlib")
has_seaborn, requires_seaborn = _importorskip("seaborn")
has_cartopy, requires_cartopy = _importorskip("cartopy")

############
# Fix for issues discussed here:
# - https://github.com/euroargodev/argopy/issues/63#issuecomment-742379699
# - https://github.com/euroargodev/argopy/issues/96
safe_to_fsspec_version = pytest.mark.skipif(
    version.parse(fsspec.__version__) > version.parse("0.8.3"),
    reason="fsspec version %s > 0.8.3 (https://github.com/euroargodev/argopy/issues/96)" % fsspec.__version__
)
skip_this_for_debug = pytest.mark.skipif(True, reason="Skipped temporarily for debug")


############
def safe_to_server_errors(test_func):
    """ Test fixture to make sure we don't fail because of an error from the server, not our Fault ! """

    def test_wrapper(fix):
        try:
            test_func(fix)
        except ErddapServerError as e:
            # Test is passed when something goes wrong because of the erddap server
            warnings.warn(
                "\nSomething happened on erddap server that should not: %s" % str(e.args)
            )
            pass
        except ArgovisServerError as e:
            # Test is passed when something goes wrong because of the argovis server
            warnings.warn(
                "\nSomething happened on argovis server that should not: %s" % str(e.args)
            )
            pass
        except DataNotFound as e:
            # We make sure that data requested by tests are available from API, so this must be a server side error !
            warnings.warn("\nDataNotFound, Something happened on server side with:\n\t-%s" % "\n\t-".join(list(e.path)))
            # traceback.print_tb(e.__traceback__)
            pass
        except ServerDisconnectedError as e:
            # We can't do anything about this !
            warnings.warn("\n We were disconnected from server !\n%s" % str(e.args))
            pass
        except ClientResponseError as e:
            # The server is sending back an error when creating the response
            warnings.warn("\nAnother server side error:\n%s" % str(e.args))
            pass
        except FileNotFoundError as e:
            warnings.warn("\nServer didn't return the data:\n%s" % str(e.args))
            pass
        except Exception as e:
            warnings.warn("\nUnknown server error:\n%s" % str(e.args))
            raise

    return test_wrapper

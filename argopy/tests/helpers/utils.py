""" Utilities for tests.

This module contains:
- all variables and functions used by tests to determine if they should be ran or not
- test wrapper to make safe requests to web APIs

"""

import importlib
import pytest
import fsspec
from aiohttp.client_exceptions import ServerDisconnectedError, ClientResponseError, ClientConnectorError
import ftplib
from packaging import version
import warnings
from argopy.errors import ErddapServerError, ArgovisServerError, DataNotFound, FtpPathError
from argopy.utilities import (
    list_available_data_src,
    list_available_index_src,
    isconnected,
    erddap_ds_exists,
    isAPIconnected
)
import logging


log = logging.getLogger("argopy.tests.utils")


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

has_connection, requires_connection = _connectskip(
    CONNECTED, "an internet connection"
)

##########
# ERDDAP #
##########
if CONNECTED:
    log.debug("Check which Erddap dataset are available (eg: core, bgc, ref, index)")
    DSEXISTS = erddap_ds_exists(ds="ArgoFloats")
    DSEXISTS_bgc = erddap_ds_exists(ds="ArgoFloats-bio")
    DSEXISTS_ref = erddap_ds_exists(ds="ArgoFloats-ref")
    DSEXISTS_index = erddap_ds_exists(ds="ArgoFloats-index")
else:
    DSEXISTS = False
    DSEXISTS_bgc = False
    DSEXISTS_ref = False
    DSEXISTS_index = False

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

has_connected_argovis = has_connection and has_argovis and isAPIconnected(src='argovis', data=True)
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

############
# GDAC FTP #
############
has_pyarrow, requires_pyarrow = _importorskip("pyarrow")

has_gdac, requires_gdac = _connectskip(
    "gdac" in AVAILABLE_SOURCES, "the gdac data fetcher"
)
has_gdac_index, requires_gdac_index = _connectskip(
    "gdac" in AVAILABLE_INDEX_SOURCES, "the gdac index fetcher"
)
has_connected_gdac = has_connection and has_gdac and isAPIconnected(src='gdac', data=True)
requires_connected_gdac = pytest.mark.skipif(
    not has_connected_gdac, reason="Requires a live GDAC server"# and pyarrow"
)

########
# PLOT #
########
has_matplotlib, requires_matplotlib = _importorskip("matplotlib")
has_seaborn, requires_seaborn = _importorskip("seaborn")
has_cartopy, requires_cartopy = _importorskip("cartopy")
has_ipython, requires_ipython = _importorskip("IPython")

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
def fct_safe_to_server_errors(func, *args, **kwargs):
    """Make any function safe to server error.

        This wrapper makes sure a function call won't fail because of an error from the server, not our Fault !
        This wrapper return the function results
        For a test decorator, use safe_to_server_errors instead
    """
    def func_wrapper(*args, **kwargs):
        msg, xmsg = None, None
        try:
            kw = kwargs.copy()
            if 'xfail' in kw:
                del kw['xfail']
            # log.debug(args)
            # log.debug(kw)
            return func(*args, **kw)
        except ErddapServerError as e:
            # Test is passed when something goes wrong because of the erddap server
            msg = "\nSomething happened on erddap server that should not: %s" % str(e.args)
            xmsg = "Failing because something happened on erddap server, but should work"
            pass
        except ArgovisServerError as e:
            # Test is passed when something goes wrong because of the argovis server
            msg = "\nSomething happened on argovis server that should not: %s" % str(e.args)
            xmsg = "Failing because something happened on argovis server, but should work"
            pass
        except DataNotFound as e:
            # We make sure that data requested by tests are available from API, so this must be a server side error !
            msg = "\nDataNotFound, Something happened on server side with:\n\t-%s" % str(e.args)
            xmsg = "Failing because some data were not found, but should work"
            pass
        except ServerDisconnectedError as e:
            # We can't do anything about this !
            msg = "\nWe were disconnected from server !\n%s" % str(e.args)
            xmsg = "Failing because we were disconnected from the server, but should work"
            pass
        except ClientResponseError as e:
            # The server is sending back an error when creating the response
            msg = "\nAnother server side error:\n%s" % str(e.args)
            xmsg = "Failing because an error happened on the server side, but should work"
            pass
        except ClientConnectorError as e:
            msg = "\naiohttp cannot connect to host:\n%s" % str(e.args)
            xmsg = "Failing because aiohttp cannot connect to host, but should work"
            pass
        except FileNotFoundError as e:
            msg = "\nServer didn't return the data:\n%s" % str(e.args)
            xmsg = "Failing because some file were not found, but should work"
            pass
        except FtpPathError as e:
            if 'xfail' in kwargs and kwargs['xfail']:
                # Some tests will expect this error to be raised, so we make sure it is, for pytest to catch it
                raise
            else:
                # Otherwise we pass, because this is not due to argopy
                msg = "\nCannot connect to the FTP path index file\n%s" % str(e.args)
                xmsg = "Failing because cannot connect to the FTP path index file, but should work"
                pass
        except ftplib.error_temp as e:
            # ftplib.error_temp: 421 There are too many connections from your internet address.
            # ftplib.error_temp: 426 Failure writing network stream.
            msg = "\nCannot connect to the FTP server\n%s" % str(e.args)
            xmsg = "Failing because cannot connect to the FTP server, but should work"
            pass
        except Exception as e:
            msg = "\nUnknown server error:\n%s" % str(e.args)
            raise  # Because we need to ID this for addition in this list
        finally:
            # We just document in log and warning what happened
            if msg is not None:
                warnings.warn(msg)
            if xmsg is not None:
                pytest.xfail(xmsg)

    return func_wrapper


def safe_to_server_errors(test_func, *args, **kwargs):
    """Decorator to safely execute a test
        This execution the test function, but does not return the result
    """
    def test_wrapper(*args, **kwargs):
        fct_safe_to_server_errors(test_func)(*args, **kwargs)
    return test_wrapper


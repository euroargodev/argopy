"""
This module manage options of the package

# Like always, largely inspired by xarray code:
# https://github.com/pydata/xarray/blob/cafab46aac8f7a073a32ec5aa47e213a9810ed54/xarray/core/options.py
"""

import os
import warnings
import logging
import fsspec
import importlib


try:
    import distributed

    has_distributed = True
except ModuleNotFoundError:
    has_distributed = False
    distributed = None


if importlib.util.find_spec("boto3") is not None:
    HAS_BOTO3 = True
else:
    HAS_BOTO3 = False


from .errors import OptionValueError, GdacPathError, ErddapPathError


# Define a logger
log = logging.getLogger("argopy.options")

# Define option names as seen by users and internals:
DATA_SOURCE = "src"
GDAC = "gdac"
ERDDAP = "erddap"
DATASET = "ds"
CACHE_FOLDER = "cachedir"
CACHE_EXPIRATION = "cache_expiration"
USER_LEVEL = "mode"
API_TIMEOUT = "api_timeout"
TRUST_ENV = "trust_env"
SERVER = "server"
USER = "user"
PASSWORD = "password"
ARGOVIS_API_KEY = "argovis_api_key"
PARALLEL = "parallel"
PARALLEL_DEFAULT_METHOD = "parallel_default_method"
LON = "longitude_convention"

# Define the list of available options and default values:
OPTIONS = {
    DATA_SOURCE: "erddap",
    GDAC: "https://data-argo.ifremer.fr",
    ERDDAP: "https://erddap.ifremer.fr/erddap",
    DATASET: "phy",
    CACHE_FOLDER: os.path.expanduser(os.path.sep.join(["~", ".cache", "argopy"])),
    CACHE_EXPIRATION: 86400,
    USER_LEVEL: "standard",
    API_TIMEOUT: 60,
    TRUST_ENV: False,
    SERVER: None,
    USER: os.environ.get("ERDDAP_USERNAME"),
    PASSWORD: os.environ.get("ERDDAP_PASSWORD"),
    ARGOVIS_API_KEY: "guest",  # https://argovis-keygen.colorado.edu
    PARALLEL: False,
    PARALLEL_DEFAULT_METHOD: "thread",
    LON: "180",
}
DEFAULT = OPTIONS.copy()

# Define the list of possible values
_DATA_SOURCE_LIST = frozenset(["erddap", "argovis", "gdac"])
_DATASET_LIST = frozenset(["phy", "bgc", "ref", "bgc-s", "bgc-b"])
_USER_LEVEL_LIST = frozenset(["standard", "expert", "research"])


# Define how to validate options:
def _positive_integer(value):
    return isinstance(value, int) and value > 0


def validate_gdac(this_path):
    if this_path != "-":
        return check_gdac_option(this_path, errors="raise")
    else:
        log.debug("OPTIONS['%s'] is not defined" % GDAC)
        return False


def validate_http(this_path):
    if this_path != "-":
        return check_erddap_path(this_path, errors="raise")
    else:
        log.debug("OPTIONS['%s'] is not defined" % ERDDAP)
        return False


def validate_parallel(method):
    """Possible values: True, False, 'thread', 'process', distributed.client.Client"""
    if isinstance(method, bool):
        return True
    else:
        return validate_parallel_method(method)


def validate_parallel_method(method):
    """Possible values: 'thread', 'process', distributed.client.Client"""
    if method in ["thread", "process"]:
        return True
    elif has_distributed and isinstance(method, distributed.client.Client):
        return True
    else:
        return False


_VALIDATORS = {
    DATA_SOURCE: _DATA_SOURCE_LIST.__contains__,
    GDAC: validate_gdac,
    ERDDAP: validate_http,
    DATASET: _DATASET_LIST.__contains__,
    CACHE_FOLDER: lambda x: os.access(x, os.W_OK),
    CACHE_EXPIRATION: lambda x: isinstance(x, int) and x > 0,
    USER_LEVEL: _USER_LEVEL_LIST.__contains__,
    API_TIMEOUT: lambda x: isinstance(x, int) and x > 0,
    TRUST_ENV: lambda x: isinstance(x, bool),
    SERVER: lambda x: True,
    USER: lambda x: isinstance(x, str) or x is None,
    PASSWORD: lambda x: isinstance(x, str) or x is None,
    ARGOVIS_API_KEY: lambda x: isinstance(x, str) or x is None,
    PARALLEL: validate_parallel,
    PARALLEL_DEFAULT_METHOD: validate_parallel_method,
    LON: lambda x: x in ['180', '360'],
}


def VALIDATE(key, val):
    """Return option value if validated otherwise raise an OptionValueError"""
    if key in _VALIDATORS:
        if not _VALIDATORS[key](val):
            raise OptionValueError(f"option '{key}' given an invalid value: '{val}'")
        else:
            return val
    else:
        raise ValueError(f"option '{key}' has no validation method")


def PARALLEL_SETUP(parallel):
    parallel = VALIDATE("parallel", parallel)
    if isinstance(parallel, bool):
        if parallel:
            return True, OPTIONS["parallel_default_method"]
        else:
            return False, "sequential"
    else:
        return True, parallel


class set_options:
    """Set options for argopy

    Parameters
    ----------

    ds: str, default: ``phy``
        Define the Dataset to work with: ``phy``, ``bgc`` or ``ref``

    src: str, default: ``erddap``
        Source of fetched data: ``erddap``, ``gdac``, ``argovis``

    mode: str, default: ``standard``
        User mode: ``standard``, ``expert`` or ``research``

    gdac: str, default: https://data-argo.ifremer.fr
        Default path to be used by the GDAC fetchers and Argo index stores

    erddap: str, default: https://erddap.ifremer.fr/erddap
        Default server address to be used by the data and index erddap fetchers

    cachedir: str, default: ``~/.cache/argopy``
        Absolute path to a local cache directory

    cache_expiration: int, default: 86400
        Expiration delay of cache files in seconds

    api_timeout: int, default: 60
        Time out for internet requests to web API, in seconds

    trust_env: bool, default: False
        Allow for local environment variables to be used to connect to the internet.

        Argopy will get proxies information from HTTP_PROXY / HTTPS_PROXY environment variables if this option is True
        and it can also get proxy credentials from ~/.netrc file if this file exists

    user: str, default: None
        Username to use when a simple authentication is required

    password: str, default: None
        Password to use when a simple authentication is required

    argovis_api_key: str, default: ``guest``
        The API key to use when fetching data from the `argovis` data source

        You can get a free key at https://argovis-keygen.colorado.edu

    parallel: bool, str, :class:`distributed.Client`, default: False
        Set whether to use parallelisation or not, and possibly which method to use.

            Possible values:
                - ``False``: no parallelisation is used
                - ``True``: use default method specified by the ``parallel_default_method`` option (see below)
                - any other values accepted by the ``parallel_default_method`` option (see below)

    parallel_default_method: str, :class:`distributed.Client`, default: ``thread``
        The default parallelisation method to use if the option ``parallel`` is simply set to ``True``.

            Possible values:
                - ``thread``: use `multi-threading <https://en.wikipedia.org/wiki/Multithreading_(computer_architecture)>`_ with a :class:`concurrent.futures.ThreadPoolExecutor`
                - ``process``: use `multi-processing <https://en.wikipedia.org/wiki/Multiprocessing>`_ with a :class:`concurrent.futures.ProcessPoolExecutor`
                -  :class:`distributed.Client`: Use a `Dask Cluster <https://docs.dask.org/en/stable/deploying.html>`_ `client <https://distributed.dask.org/en/latest/client.html>`_.

    longitude_convention: str, default: '180',
        The longitude convention to use when longitudes are compared.

            Possible values:
                - '180': longitude goes from -180 to 180
                - '360': longitude goes from 0 to 360

    Other Parameters
    ----------------
    server: : str, default: None
        Other than expected/default server to be uses by a function/method

        This is mostly intended to be used by unit tests.

    Examples
    --------

    You can use ``set_options`` either as a context manager for temporary setting:

    >>> import argopy
    >>> with argopy.set_options(src='gdac'):
    >>>    ds = argopy.DataFetcher().float(3901530).to_xarray()

    or to set global options (at the beginning of a script for instance):

    >>> argopy.set_options(src='gdac')

    Warns
    -----
    A DeprecationWarning can be raised when a deprecated option is set

    """

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(
                    "argument name %r is not in the set of valid options %r"
                    % (k, set(OPTIONS))
                )

            if k == CACHE_FOLDER:
                os.makedirs(v, exist_ok=True)

            VALIDATE(k, v)

            self.old[k] = OPTIONS[k]
        self._apply_update(kwargs)

    def _apply_update(self, options_dict):
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)


def reset_options():
    """Reset all options to default values"""
    set_options(**DEFAULT)


def check_erddap_path(path, errors="ignore"):
    """Check if an url points to an ERDDAP server"""
    fs = fsspec.filesystem("http", ssl=False)
    check1 = fs.exists(path + "/info/index.json")
    if check1:
        return True
    elif errors == "raise":
        raise ErddapPathError("This url is not a valid ERDDAP server:\n%s" % path)

    elif errors == "warn":
        warnings.warn("This url is not a valid ERDDAP server:\n%s" % path)
        return False
    else:
        return False


def check_gdac_option(
    path, errors: str = "ignore", ignore_knowns: bool = True
):  # noqa: C901
    """Check if a path has the expected GDAC structure

    Expected GDAC structure::

        .
        └── dac
            ├── aoml
            ├── ...
            ├── coriolis
            ├── ...
            ├── meds
            └── nmdis

    Examples::

    >>> check_gdac_path("https://data-argo.ifremer.fr")  # True
    >>> check_gdac_path("https://usgodae.org/pub/outgoing/argo") # True
    >>> check_gdac_path("ftp://ftp.ifremer.fr/ifremer/argo") # True
    >>> check_gdac_path("/home/ref-argo/gdac") # True
    >>> check_gdac_path("s3://argo-gdac-sandbox/pub") # True

    >>> check_gdac_path("https://www.ifremer.fr") # False
    >>> check_gdac_path("ftp://usgodae.org/pub/outgoing") # False

    Parameters
    ----------
    path: str
        Path name to check, including access protocol
    errors: str, default="ignore"
        Determine how check procedure error are handled: "ignore", "raise" or "warn"
    ignore_knowns: bool, default=False
        Should the checking procedure be by-passed for the internal list of known GDACs.
        Set this to True to check if a known GDACs is connected or not.

    Returns
    -------
    checked: boolean

    See also
    --------
    :class:`argopy.stores.gdacfs`, :meth:`argopy.utils.list_gdac_servers`

    """
    from .utils import (
        list_gdac_servers,
    )  # import here, otherwise raises circular import

    if path in list_gdac_servers() and ignore_knowns:
        return True
    else:

        from .stores import gdacfs  # import here, otherwise raises circular import

        try:
            fs = gdacfs(path)
        except GdacPathError:
            if errors == "raise":
                raise
            elif errors == "warn":
                warnings.warn("Can't get address info (GAIerror) on '%s'" % path)
                return False
            else:
                return False

        check1 = fs.exists("dac")
        if check1:
            return True

        elif errors == "raise":
            raise GdacPathError(
                "This path is not GDAC compliant (no legitimate sub-folder `dac`):\n%s"
                % path
            )

        elif errors == "warn":
            warnings.warn(
                "This path is not GDAC compliant (no legitimate sub-folder `dac`):\n%s"
                % path
            )
            return False

        else:
            return False

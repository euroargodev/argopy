"""
This module manage options of the package

# Like always, largely inspired by xarray code:
# https://github.com/pydata/xarray/blob/cafab46aac8f7a073a32ec5aa47e213a9810ed54/xarray/core/options.py
"""
import os
from argopy.errors import OptionValueError, FtpPathError, ErddapPathError
import warnings
import logging
import fsspec
from fsspec.core import split_protocol
from socket import gaierror
from urllib.parse import urlparse


# Define a logger
log = logging.getLogger("argopy.options")

# Define option names as seen by users:
DATA_SOURCE = "src"
FTP = "ftp"
ERDDAP = 'erddap'
DATASET = "dataset"
CACHE_FOLDER = "cachedir"
CACHE_EXPIRATION = "cache_expiration"
USER_LEVEL = "mode"
API_TIMEOUT = "api_timeout"
TRUST_ENV = "trust_env"
SERVER = "server"
USER = "user"
PASSWORD = "password"

# Define the list of available options and default values:
OPTIONS = {
    DATA_SOURCE: "erddap",
    FTP: "https://data-argo.ifremer.fr",
    ERDDAP: "https://erddap.ifremer.fr/erddap",
    DATASET: "phy",
    CACHE_FOLDER: os.path.expanduser(os.path.sep.join(["~", ".cache", "argopy"])),
    CACHE_EXPIRATION: 86400,
    USER_LEVEL: "standard",
    API_TIMEOUT: 60,
    TRUST_ENV: False,
    SERVER: None,
    USER: None,
    PASSWORD: None,
}
DEFAULT = OPTIONS.copy()

# Define the list of possible values
_DATA_SOURCE_LIST = frozenset(["erddap", "argovis", "gdac"])
_DATASET_LIST = frozenset(["phy", "bgc", "ref"])
_USER_LEVEL_LIST = frozenset(["standard", "expert", "research"])


# Define how to validate options:
def _positive_integer(value):
    return isinstance(value, int) and value > 0


def validate_ftp(this_path):
    if this_path != "-":
        return check_gdac_path(this_path, errors='raise')
    else:
        log.debug("OPTIONS['%s'] is not defined" % FTP)
        return False


def validate_http(this_path):
    if this_path != "-":
        return check_erddap_path(this_path, errors='raise')
    else:
        log.debug("OPTIONS['%s'] is not defined" % ERDDAP)
        return False


_VALIDATORS = {
    DATA_SOURCE: _DATA_SOURCE_LIST.__contains__,
    FTP: validate_ftp,
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
}


class set_options:
    """Set options for argopy

    List of options:

    - ``dataset``: Define the Dataset to work with.
        Default: ``phy``.
        Possible values: ``phy``, ``bgc`` or ``ref``.
    - ``src``: Source of fetched data.
        Default: ``erddap``.
        Possible values: ``erddap``, ``gdac``, ``argovis``
    - ``mode``: User mode.
        Default: ``standard``.
        Possible values: ``standard``, ``expert`` or ``research``.
    - ``ftp``: Default path to be used by the GDAC fetchers and Argo index stores
        Default: https://data-argo.ifremer.fr
    - ``erddap``: Default server address to be used by the data and index erddap fetchers
        Default: https://erddap.ifremer.fr/erddap
    - ``cachedir``: Absolute path to a local cache directory.
        Default: ``~/.cache/argopy``
    - ``cache_expiration``: Expiration delay of cache files in seconds.
        Default: 86400
    - ``api_timeout``: Define the time out of internet requests to web API, in seconds.
        Default: 60
    - ``trust_env``: Allow for local environment variables to be used to connect to the internet.
        Default: False.
        Argopy will get proxies information from HTTP_PROXY / HTTPS_PROXY environment variables if this option is True and it can also get proxy credentials from ~/.netrc file if this file exists.
    - ``user``/``password``: Username and password to use when a simple authentication is required.
        Default: None, None
    - ``server``: Other than expected/default server to be uses by a function/method. This is mostly intended to be used for unit testing
        Default: None


    You can use ``set_options`` either as a context manager for temporary setting:

    >>> import argopy
    >>> with argopy.set_options(src='gdac'):
    >>>    ds = argopy.DataFetcher().float(3901530).to_xarray()

    or to set global options (at the beginning of a script for instance):

    >>> argopy.set_options(src='gdac')

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
            if k in _VALIDATORS and not _VALIDATORS[k](v):
                raise OptionValueError(f"option {k!r} given an invalid value: {v!r}")
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


def check_erddap_path(path, errors='ignore'):
    """Check if an url points to an ERDDAP server"""
    fs = fsspec.filesystem('http')
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


def check_gdac_path(path, errors='ignore'):  # noqa: C901
    """ Check if a path has the expected GDAC ftp structure

        Check if a path is structured like:
        .
        └── dac
            ├── aoml
            ├── ...
            ├── coriolis
            ├── ...
            ├── meds
            └── nmdis

        Examples:
        >>> check_gdac_path("https://data-argo.ifremer.fr")  # True
        >>> check_gdac_path("ftp://ftp.ifremer.fr/ifremer/argo") # True
        >>> check_gdac_path("ftp://usgodae.org/pub/outgoing/argo") # True
        >>> check_gdac_path("/home/ref-argo/gdac") # True
        >>> check_gdac_path("https://www.ifremer.fr") # False
        >>> check_gdac_path("ftp://usgodae.org/pub/outgoing") # False

        Parameters
        ----------
        path: str
            Path name to check, including access protocol
        errors: str
            "ignore" or "raise" (or "warn")

        Returns
        -------
        checked: boolean
            True if at least one DAC folder is found under path/dac/<dac_name>
            False otherwise
    """
    # Create a file system for this path
    if split_protocol(path)[0] is None:
        fs = fsspec.filesystem('file')
    elif split_protocol(path)[0] in ['https', 'http']:
        fs = fsspec.filesystem('http')
    elif 'ftp' in split_protocol(path)[0]:
        try:
            host = urlparse(path).hostname
            port = 0 if urlparse(path).port is None else urlparse(path).port
            fs = fsspec.filesystem('ftp', host=host, port=port)
        except gaierror:
            if errors == 'raise':
                raise FtpPathError("Can't get address info (GAIerror) on '%s'" % host)
            elif errors == "warn":
                warnings.warn("Can't get address info (GAIerror) on '%s'" % host)
                return False
            else:
                return False
    else:
        raise FtpPathError("Unknown protocol for an Argo GDAC host: %s" % split_protocol(path)[0])

    # dacs = [
    #     "aoml",
    #     "bodc",
    #     "coriolis",
    #     "csio",
    #     "csiro",
    #     "incois",
    #     "jma",
    #     "kma",
    #     "kordi",
    #     "meds",
    #     "nmdis",
    # ]

    # Case 1:
    # check1 = (
    #     fs.exists(path)  # Fails on localhost for the mocked ftp server
    #     and fs.exists(fs.sep.join([path, "dac"]))
    #     # and np.any([fs.exists(fs.sep.join([path, "dac", dac])) for dac in dacs])  # Take too much time on http/ftp GDAC server
    # )
    check1 = fs.exists(fs.sep.join([path, "dac"]))
    if check1:
        return True

    elif errors == "raise":
        raise FtpPathError("This path is not GDAC compliant (no `dac` folder with legitimate sub-folder):\n%s" % path)

    elif errors == "warn":
        warnings.warn("This path is not GDAC compliant:\n%s" % path)
        return False

    else:
        return False

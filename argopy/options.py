"""
This module manage options of the package

# Like always, largely inspired by xarray code:
# https://github.com/pydata/xarray/blob/cafab46aac8f7a073a32ec5aa47e213a9810ed54/xarray/core/options.py
"""
import os
import numpy as np
from argopy.errors import OptionValueError, FtpPathError
import warnings
import logging


# Define a logger
log = logging.getLogger("argopy.options")

# Define option names as seen by users:
DATA_SOURCE = "src"
LOCAL_FTP = "local_ftp"
DATASET = "dataset"
DATA_CACHE = "cachedir"
USER_LEVEL = "mode"
API_TIMEOUT = "api_timeout"
TRUST_ENV = "trust_env"

# Define the list of available options and default values:
OPTIONS = {
    DATA_SOURCE: "erddap",
    LOCAL_FTP: "-",  # No default value
    DATASET: "phy",
    DATA_CACHE: os.path.expanduser(os.path.sep.join(["~", ".cache", "argopy"])),
    USER_LEVEL: "standard",
    API_TIMEOUT: 60,
    TRUST_ENV: False
}

# Define the list of possible values
_DATA_SOURCE_LIST = frozenset(["erddap", "localftp", "argovis"])
_DATASET_LIST = frozenset(["phy", "bgc", "ref"])
_USER_LEVEL_LIST = frozenset(["standard", "expert"])


# Define how to validate options:
def _positive_integer(value):
    return isinstance(value, int) and value > 0


def validate_ftp(this_path):
    if this_path != "-":
        return check_localftp(this_path, errors='raise')
    else:
        log.debug("OPTIONS['%s'] is not defined" % LOCAL_FTP)
        return False


_VALIDATORS = {
    DATA_SOURCE: _DATA_SOURCE_LIST.__contains__,
    LOCAL_FTP: validate_ftp,
    DATASET: _DATASET_LIST.__contains__,
    DATA_CACHE: os.path.exists,
    USER_LEVEL: _USER_LEVEL_LIST.__contains__,
    API_TIMEOUT: lambda x: isinstance(x, int) and x > 0,
    TRUST_ENV: lambda x: isinstance(x, bool)
}


class set_options:
    """Set options for argopy

    List of options:

    - ``dataset``: Define the Dataset to work with.
        Default: ``phy``.
        Possible values: ``phy``, ``bgc`` or ``ref``.
    - ``src``: Source of fetched data.
        Default: ``erddap``.
        Possible values: ``erddap``, ``localftp``, ``argovis``
    - ``local_ftp``: Absolute path to a local GDAC ftp copy.
        Default: None
    - ``cachedir``: Absolute path to a local cache directory.
        Default: ``~/.cache/argopy``
    - ``mode``: User mode.
        Default: ``standard``.
        Possible values: ``standard`` or ``expert``.
    - ``api_timeout``: Define the time out of internet requests to web API, in seconds.
        Default: 60
    - ``trust_env``: Allow for local environment variables to be used by fsspec to connect to the internet.
        Get proxies information from HTTP_PROXY / HTTPS_PROXY environment variables if this option is True (
        False by default). Also can get proxy credentials from ~/.netrc file if present.

    You can use `set_options` either as a context manager:

    >>> import argopy
    >>> with argopy.set_options(src='localftp'):
    >>>    ds = argopy.DataFetcher().float(3901530).to_xarray()

    Or to set global options:

    >>> argopy.set_options(src='localftp')

    """
    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(
                    "argument name %r is not in the set of valid options %r"
                    % (k, set(OPTIONS))
                )
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


def check_localftp(path, errors: str = "ignore"):
    """ Check if the path has the expected GDAC ftp structure

        Check if the path is structured like:
        .
        └── dac
            ├── aoml
            ├── ...
            ├── coriolis
            ├── ...
            ├── meds
            └── nmdis

        Parameters
        ----------
        path: str
            Path name to check
        errors: str
            "ignore" or "raise" (or "warn"

        Returns
        -------
        checked: boolean
            True if at least one DAC folder is found under path/dac/<dac_name>
            False otherwise
    """
    dacs = [
        "aoml",
        "bodc",
        "coriolis",
        "csio",
        "csiro",
        "incois",
        "jma",
        "kma",
        "kordi",
        "meds",
        "nmdis",
    ]

    # Case 1:
    check1 = (
        os.path.isdir(path)
        and os.path.isdir(os.path.join(path, "dac"))
        and np.any([os.path.isdir(os.path.join(path, "dac", dac)) for dac in dacs])
    )

    if check1:
        return True
    elif errors == "raise":
        # This was possible up to v0.1.3:
        check2 = os.path.isdir(path) and np.any(
            [os.path.isdir(os.path.join(path, dac)) for dac in dacs]
        )
        if check2:
            raise FtpPathError(
                "This path is no longer GDAC compliant for argopy.\n"
                "Please make sure you point toward a path with a 'dac' folder:\n%s"
                % path
            )
        else:
            raise FtpPathError("This path is not GDAC compliant (no `dac` folder with legitimate sub-folder):\n%s" % path)

    elif errors == "warn":
        # This was possible up to v0.1.3:
        check2 = os.path.isdir(path) and np.any(
            [os.path.isdir(os.path.join(path, dac)) for dac in dacs]
        )
        if check2:
            warnings.warn(
                "This path is no longer GDAC compliant for argopy. This will raise an error in the future.\n"
                "Please make sure you point toward a path with a 'dac' folder:\n%s"
                % path
            )
            return False
        else:
            warnings.warn("This path is not GDAC compliant:\n%s" % path)
            return False
    else:
        return False
